//! LGS self-play for Connect 4 — generates v2 training records from the KR-PUCT
//! latent search (docs/algorithm.md).
//!
//! Two drivers share `play_game` and the v2 record format:
//!   * `run_generate` — single game at a time (correctness-first).
//!   * `run_generate_batched` — the throughput path, mirroring the discrete
//!     `core::batch_search`: a large pool of games (>> batch size) runs across
//!     worker threads in lockstep, resolving the transposition cache inline and
//!     submitting whole *waves* of net-op misses to one central inference server
//!     that coalesces them into fixed-size batched forwards (one GPU consumer, so
//!     no concurrent-CUDA serialization). Finished games are continuously
//!     replenished from a shared budget to keep batches full and cache hits high.
//!
//! Record v2 (fixed stride `REC_F32` little-endian f32 per training example):
//!   parent planes (PLANES) | z (1) | MAX_CHILDREN × [col, visits, child planes]
//! Absent child slots are zero-filled (visits = 0). The trainer re-encodes the
//! parent and each child with the live net, so targets live in the current
//! latent space (the policy/reifier/consistency losses need that).

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Instant;

use async_trait::async_trait;
use candle_core::Device;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use playground_core::latent_mcts::{search, RootChild, SearchConfig};
use playground_core::{Game, LatentGameState, LatentModel, MixturePolicy, Outcome};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS};
use crate::lgs_net::{move_from_decoded, LgsModel, LgsServer};
use crate::mcts::{planes_of, PLANES};

/// Latent dimension (must match the LGS net / lgs_net.rs).
const LGS_DIM: usize = 64;

/// Max reified root children stored per move (Connect 4 has ≤7 columns).
pub const MAX_CHILDREN: usize = C4_COLS;
/// Fixed f32 stride of one v2 record.
pub const REC_F32: usize = PLANES + 1 + MAX_CHILDREN * (2 + PLANES);

/// One move's training target before the final outcome `z` is known.
struct MoveRec {
    parent: Connect4State,
    mover: u8, // 1 or 2, the side to move at `parent`
    children: Vec<(u8, Connect4State, u32)>, // (col, child state, merged visits)
}

/// Play one self-play game; returns per-move records with `z` filled in.
///
/// Playout Cap Randomization (KataGo): each move is a *full* (high-playout)
/// search with probability `pcr_full_prob`, else a cheap *fast* search. Both pick
/// the move and yield a value record (`z` from the game outcome), but only full
/// moves emit a policy/reifier target (the children block) — fast moves store
/// zeroed children, which the trainer weights to zero. So value trains on every
/// position, policy/reifier only on the high-quality full ones.
async fn play_game<M: LatentModel<Connect4>>(
    model: &M,
    cfg_full: &SearchConfig,
    cfg_fast: &SearchConfig,
    pcr_full_prob: f64,
    temp_moves: u32,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(Vec<(MoveRec, f32)>, u32)> {
    let mut state = Connect4::initial_state();
    let mut recs: Vec<MoveRec> = Vec::new();
    let mut reifications = 0u32;

    while !Connect4::outcome(&state).is_terminal() {
        let full = rng.gen::<f64>() < pcr_full_prob;
        let cfg = if full { cfg_full } else { cfg_fast };
        let res = search(model, &state, cfg).await?;
        reifications += res.reifications;
        if res.root_children.is_empty() {
            break; // tiny budget produced no reified child; bail safely
        }
        let sample = Connect4::ply(&state) < temp_moves;
        let pick = choose(&res.root_children, sample, rng);
        // PCR: only full (high-playout) moves contribute a policy/reifier target.
        let children = if full {
            res.root_children.iter().map(|c| (c.mv.col, c.state.clone(), c.visits)).collect()
        } else {
            Vec::new()
        };
        let mover = if state.ply % 2 == 0 { 1u8 } else { 2u8 };
        let next = res.root_children[pick].state.clone();
        recs.push(MoveRec { parent: state.clone(), mover, children });
        state = next;
    }

    let winner = match Connect4::outcome(&state) {
        Outcome::FirstWins => Some(1u8),
        Outcome::SecondWins => Some(2u8),
        _ => None,
    };
    let out = recs
        .into_iter()
        .map(|r| {
            let z = match winner {
                Some(w) => if w == r.mover { 1.0 } else { -1.0 },
                None => 0.0,
            };
            (r, z)
        })
        .collect();
    Ok((out, reifications))
}

/// Pick a root child: sample ∝ visits during the opening, else the max-visit
/// child (children are pre-sorted by visits descending).
fn choose(children: &[RootChild<Connect4>], sample: bool, rng: &mut ChaCha8Rng) -> usize {
    if !sample {
        return 0;
    }
    let total: u32 = children.iter().map(|c| c.visits).sum();
    if total == 0 {
        return 0;
    }
    let mut r = rng.gen_range(0..total);
    for (i, c) in children.iter().enumerate() {
        if r < c.visits {
            return i;
        }
        r -= c.visits;
    }
    0
}

fn push_planes(buf: &mut Vec<f32>, planes: &[f32; PLANES]) {
    buf.extend_from_slice(planes);
}

/// Serialize records to the v2 dump (u64 count, then `count` × `REC_F32` f32).
fn write_dump(out_path: &str, examples: &[(MoveRec, f32)]) -> anyhow::Result<()> {
    let mut floats: Vec<f32> = Vec::with_capacity(examples.len() * REC_F32);
    let zeros = [0f32; PLANES];
    for (rec, z) in examples {
        push_planes(&mut floats, &planes_of(&rec.parent));
        floats.push(*z);
        for slot in 0..MAX_CHILDREN {
            match rec.children.get(slot) {
                Some((col, child, visits)) => {
                    floats.push(*col as f32);
                    floats.push(*visits as f32);
                    push_planes(&mut floats, &planes_of(child));
                }
                None => {
                    floats.push(0.0);
                    floats.push(0.0);
                    push_planes(&mut floats, &zeros);
                }
            }
        }
    }
    let mut bytes = Vec::with_capacity(8 + floats.len() * 4);
    bytes.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for x in &floats {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    f.write_all(&bytes)?;
    f.flush()?;
    Ok(())
}

/// Single-game LGS self-play (correctness path). Writes a v2 dump.
#[allow(clippy::too_many_arguments)]
pub async fn run_generate(
    checkpoint: &str,
    cuda: bool,
    out_path: &str,
    n_games: usize,
    sims: u32,
    fast_sims: u32,
    pcr_full_prob: f64,
    temp_moves: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LgsModel::load_on(checkpoint, device)?);
    let cfg_full = SearchConfig { simulations: sims, c_puct, kernel_bandwidth, root_explore: true, seed: None, ..Default::default() };
    let cfg_fast = SearchConfig { simulations: fast_sims, c_puct, kernel_bandwidth, root_explore: false, seed: None, ..Default::default() };
    eprintln!(
        "c4 LGS self-play: {n_games} games, PCR full {sims}/fast {fast_sims} sims (p_full {pcr_full_prob}), h {kernel_bandwidth}, model {checkpoint} -> {out_path}"
    );
    let t0 = Instant::now();
    let mut all: Vec<(MoveRec, f32)> = Vec::new();
    let mut reifications = 0u64;
    for g in 0..n_games {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(g as u64));
        let (recs, reif) = play_game(model.as_ref(), &cfg_full, &cfg_fast, pcr_full_prob, temp_moves, &mut rng).await?;
        reifications += reif as u64;
        all.extend(recs);
    }
    write_dump(out_path, &all)?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "  wrote {} examples ({} games) in {:.1}s  ({:.1} games/s, {} reifications)",
        all.len(),
        n_games,
        dt,
        n_games as f64 / dt,
        reifications,
    );
    Ok(())
}

// ── Central-batched throughput path ───────────────────────────────────────────
//
// A large pool of games (>> batch size) runs across worker threads. Each worker
// holds `in_flight` games and drives them in lockstep (reusing the async
// latent_mcts::search): per round it polls every active game, resolves the
// encode/policy transposition cache inline, and submits a *wave* of the misses
// (+ all eval/reify) to the central inference server, which coalesces waves from
// all workers into fixed-size batched forwards (one GPU consumer). Finished games
// are replenished from a shared budget, keeping the pipeline full.

/// A net op a game yields this round. `encode`/`policy` ride the fused ops too
/// (value discarded by the caller), so every state-keyed forward is cacheable and
/// the dominant standalone `eval` (value head) is folded into `EncodeEval` (reify
/// leaf) and `PolicyEval` (node expansion). Only sampled-latent and aliased-child
/// evals remain as standalone `Eval`s.
enum Pending {
    EncodeEval(Vec<f32>),       // planes  → (lgs, value)
    Eval(Vec<f32>),             // lgs     → value
    PolicyEval(Vec<f32>),       // lgs     → (MDN, value)
    Reify(Vec<f32>),            // next-LGS → decoded board planes
}

/// The driver's reply to a yielded op.
enum OpOut {
    LgsVal(Vec<f32>, f32),
    Val(f32),
    PolVal(MixturePolicy, f32),
    Decoded(Vec<f32>),  // decoder plane logits (2*CELLS)
}

type PolOut = (usize, Vec<f32>, Vec<f32>, Vec<f32>);

/// A sharded concurrent map (key → Arc<V>). Sharding + per-key-short locks keep
/// the many worker threads from serializing on one cache mutex.
struct ShardCache<V> {
    shards: Vec<Mutex<HashMap<u64, Arc<V>>>>,
}

impl<V> ShardCache<V> {
    fn new(shards: usize) -> Self {
        Self { shards: (0..shards).map(|_| Mutex::new(HashMap::new())).collect() }
    }
    #[inline]
    fn shard(&self, key: u64) -> &Mutex<HashMap<u64, Arc<V>>> {
        &self.shards[(key % self.shards.len() as u64) as usize]
    }
    fn get(&self, key: u64) -> Option<Arc<V>> {
        self.shard(key).lock().unwrap().get(&key).cloned()
    }
    fn insert(&self, key: u64, v: Arc<V>) {
        self.shard(key).lock().unwrap().insert(key, v);
    }
}

/// Transposition cache over *real-state* fused forwards, shared across worker
/// threads. Connect 4 transposes heavily, so the same board recurs across
/// games/searches; caching removes the dominant conv (encode) and value/policy
/// forwards. `enc` is keyed by board planes → (lgs, value); `pol` by lgs →
/// (MDN, value). Standalone `Eval`/`Reify` hit unique sampled latents and stay
/// uncached. Keyed by a hash of the input floats.
struct Cache {
    enc: ShardCache<(Vec<f32>, f32)>,
    pol: ShardCache<(PolOut, f32)>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            enc: ShardCache::new(64),
            pol: ShardCache::new(64),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

/// FNV-1a over the float bit patterns (collision-safe at our scale; avoids
/// allocating a byte-vector key on the hot path).
fn hash_f32s(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Per-worker lockstep state: each slot holds at most one pending request + its
/// result. The `Mutex` only satisfies `LatentModel: Send + Sync`; it is
/// uncontended (a single worker thread owns each lane).
struct Lane {
    request: Vec<Option<Pending>>,
    result: Vec<Option<OpOut>>,
}

/// A `LatentModel` bound to one game slot: registers each op into the lane and
/// awaits the result the worker writes after the batched forward.
struct LockstepModel {
    slot: usize,
    lane: Arc<Mutex<Lane>>,
    version: String,
    /// (valid_decodes, total_decodes) shared across the run for the
    /// valid-decode-rate metric (fraction of reify calls whose decoded board was
    /// a legal single-drop successor vs a random fallback).
    decode_stats: Arc<(AtomicU64, AtomicU64)>,
}

/// Future that resolves once the worker fills `lane.result[slot]`.
struct OpFuture {
    slot: usize,
    lane: Arc<Mutex<Lane>>,
}

impl Future for OpFuture {
    type Output = OpOut;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<OpOut> {
        match self.lane.lock().unwrap().result[self.slot].take() {
            Some(r) => Poll::Ready(r),
            None => Poll::Pending,
        }
    }
}

impl LockstepModel {
    fn submit(&self, req: Pending) -> OpFuture {
        self.lane.lock().unwrap().request[self.slot] = Some(req);
        OpFuture { slot: self.slot, lane: self.lane.clone() }
    }
}

#[async_trait]
impl LatentModel<Connect4> for LockstepModel {
    async fn encode(&self, state: &Connect4State) -> anyhow::Result<LatentGameState> {
        let (lgs, _) = self.encode_eval(state).await?;
        Ok(lgs)
    }
    async fn encode_eval(&self, state: &Connect4State) -> anyhow::Result<(LatentGameState, f64)> {
        match self.submit(Pending::EncodeEval(planes_of(state).to_vec())).await {
            OpOut::LgsVal(v, val) => Ok((LatentGameState::from_f32(&v, self.version.clone()), val as f64)),
            _ => unreachable!("encode_eval reply"),
        }
    }
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        match self.submit(Pending::Eval(lgs.to_f32())).await {
            OpOut::Val(v) => Ok(v as f64),
            _ => unreachable!("eval reply"),
        }
    }
    async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy> {
        let (p, _) = self.policy_eval(lgs).await?;
        Ok(p)
    }
    async fn policy_eval(&self, lgs: &LatentGameState) -> anyhow::Result<(MixturePolicy, f64)> {
        match self.submit(Pending::PolicyEval(lgs.to_f32())).await {
            OpOut::PolVal(p, val) => Ok((p, val as f64)),
            _ => unreachable!("policy_eval reply"),
        }
    }
    async fn reify(
        &self,
        from_state: &Connect4State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<Connect4Move> {
        let _ = from_lgs; // decoder reifies from the target next-LGS alone
        let decoded = match self.submit(Pending::Reify(to_lgs.to_f32())).await {
            OpOut::Decoded(p) => p,
            _ => unreachable!("reify reply"),
        };
        let (mv, valid) = move_from_decoded(from_state, &decoded);
        self.decode_stats.1.fetch_add(1, Ordering::Relaxed);
        if valid {
            self.decode_stats.0.fetch_add(1, Ordering::Relaxed);
        }
        Ok(mv)
    }
    async fn model_version(&self) -> anyhow::Result<String> {
        Ok(self.version.clone())
    }
}

/// A waker that does nothing — we drive the futures manually in lockstep.
fn noop_waker() -> Waker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

type GameFut = Pin<Box<dyn Future<Output = anyhow::Result<(Vec<(MoveRec, f32)>, u32)>>>>;

fn make_game(
    slot: usize,
    lane: Arc<Mutex<Lane>>,
    version: String,
    cfg_full: SearchConfig,
    cfg_fast: SearchConfig,
    pcr_full_prob: f64,
    temp_moves: u32,
    seed: u64,
    decode_stats: Arc<(AtomicU64, AtomicU64)>,
) -> GameFut {
    Box::pin(async move {
        let model = LockstepModel { slot, lane, version, decode_stats };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        play_game(&model, &cfg_full, &cfg_fast, pcr_full_prob, temp_moves, &mut rng).await
    })
}

#[derive(Default)]
struct ThreadOut {
    records: Vec<(MoveRec, f32)>,
    reifications: u64,
}

/// Server batching stats (rounds = coalesced forward-sets; ops = items served).
#[derive(Default)]
struct ServerStats {
    rounds: AtomicU64,
    ops: AtomicU64,
    n_enc: AtomicU64,
    n_eval: AtomicU64,
    n_pol: AtomicU64,
    n_reif: AtomicU64,
}

/// One worker's wave of net-op *misses* for the central inference server.
#[derive(Default)]
struct Wave {
    enc_eval: Vec<Vec<f32>>, // planes
    eval: Vec<Vec<f32>>,     // lgs (sampled / aliased child)
    pol_eval: Vec<Vec<f32>>, // lgs
    reif: Vec<Vec<f32>>,     // next-LGS to decode
}

impl Wave {
    fn count(&self) -> usize {
        self.enc_eval.len() + self.eval.len() + self.pol_eval.len() + self.reif.len()
    }
    fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

#[derive(Default)]
struct WaveOut {
    enc_eval: Vec<(Vec<f32>, f32)>,
    eval: Vec<f32>,
    pol_eval: Vec<(PolOut, f32)>,
    reif: Vec<Vec<f32>>,
}

/// A submitted wave + its reply channel.
struct Job {
    wave: Wave,
    resp: std::sync::mpsc::SyncSender<WaveOut>,
}

/// Central inference server: coalesce workers' waves up to `max_batch` total
/// items, run one batched forward per op type across all of them, scatter the
/// per-wave results. One thread → one GPU consumer (no concurrent CUDA).
fn inference_server(
    rx: std::sync::mpsc::Receiver<Job>,
    server: &LgsServer,
    max_batch: usize,
    stats: &ServerStats,
) {
    // Reused across rounds (the server is serial, so its per-round marshalling is
    // on the critical path workers block on): clear-and-refill, never realloc.
    let mut flat = Vec::new();
    loop {
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => break, // all workers done
        };
        let mut total = first.wave.count();
        let mut jobs = vec![first];
        while total < max_batch {
            match rx.try_recv() {
                Ok(j) => {
                    total += j.wave.count();
                    jobs.push(j);
                }
                Err(_) => break,
            }
        }
        let mut outs: Vec<WaveOut> = (0..jobs.len()).map(|_| WaveOut::default()).collect();

        // ENCODE_EVAL (planes → lgs + value, one fused forward)
        let n: usize = jobs.iter().map(|j| j.wave.enc_eval.len()).sum();
        if n > 0 {
            flat.clear();
            for j in &jobs {
                for p in &j.wave.enc_eval {
                    flat.extend_from_slice(p);
                }
            }
            let (lgs, val) = server.encode_eval_batch(&flat, n).expect("encode_eval_batch");
            let mut off = 0;
            for (ji, j) in jobs.iter().enumerate() {
                for _ in 0..j.wave.enc_eval.len() {
                    outs[ji].enc_eval.push((lgs[off * LGS_DIM..(off + 1) * LGS_DIM].to_vec(), val[off]));
                    off += 1;
                }
            }
        }
        // EVAL (sampled / aliased-child latents)
        let n: usize = jobs.iter().map(|j| j.wave.eval.len()).sum();
        if n > 0 {
            flat.clear();
            for j in &jobs {
                for x in &j.wave.eval {
                    flat.extend_from_slice(x);
                }
            }
            let r = server.eval_batch(&flat, n).expect("eval_batch");
            let mut off = 0;
            for (ji, j) in jobs.iter().enumerate() {
                for _ in 0..j.wave.eval.len() {
                    outs[ji].eval.push(r[off]);
                    off += 1;
                }
            }
        }
        // POLICY_EVAL (lgs → MDN + value, one fused forward)
        let n: usize = jobs.iter().map(|j| j.wave.pol_eval.len()).sum();
        if n > 0 {
            flat.clear();
            for j in &jobs {
                for x in &j.wave.pol_eval {
                    flat.extend_from_slice(x);
                }
            }
            let (k, w, mu, std, val) = server.policy_eval_batch(&flat, n).expect("policy_eval_batch");
            let md = k * LGS_DIM;
            let mut off = 0;
            for (ji, j) in jobs.iter().enumerate() {
                for _ in 0..j.wave.pol_eval.len() {
                    outs[ji].pol_eval.push((
                        (
                            k,
                            w[off * k..(off + 1) * k].to_vec(),
                            mu[off * md..(off + 1) * md].to_vec(),
                            std[off * md..(off + 1) * md].to_vec(),
                        ),
                        val[off],
                    ));
                    off += 1;
                }
            }
        }
        // REIFY (decode next-LGS → board planes; move inferred CPU-side)
        let n: usize = jobs.iter().map(|j| j.wave.reif.len()).sum();
        if n > 0 {
            flat.clear();
            for j in &jobs {
                for x in &j.wave.reif {
                    flat.extend_from_slice(x);
                }
            }
            let r = server.decode_batch(&flat, n).expect("decode_batch");
            let mut off = 0;
            for (ji, j) in jobs.iter().enumerate() {
                for _ in 0..j.wave.reif.len() {
                    outs[ji].reif.push(r[off * PLANES..(off + 1) * PLANES].to_vec());
                    off += 1;
                }
            }
        }

        stats.rounds.fetch_add(1, Ordering::Relaxed);
        stats.ops.fetch_add(total as u64, Ordering::Relaxed);
        stats.n_enc.fetch_add(jobs.iter().map(|j| j.wave.enc_eval.len() as u64).sum(), Ordering::Relaxed);
        stats.n_eval.fetch_add(jobs.iter().map(|j| j.wave.eval.len() as u64).sum(), Ordering::Relaxed);
        stats.n_pol.fetch_add(jobs.iter().map(|j| j.wave.pol_eval.len() as u64).sum(), Ordering::Relaxed);
        stats.n_reif.fetch_add(jobs.iter().map(|j| j.wave.reif.len() as u64).sum(), Ordering::Relaxed);
        for (j, o) in jobs.into_iter().zip(outs) {
            let _ = j.resp.send(o);
        }
    }
}

/// One worker: drive a pool of `in_flight` games in lockstep, replenishing from
/// the shared `next_game` budget, submitting waves to the central server.
#[allow(clippy::too_many_arguments)]
fn worker(
    tx: &std::sync::mpsc::Sender<Job>,
    cache: &Cache,
    in_flight: usize,
    next_game: &AtomicUsize,
    n_games: usize,
    cfg_full: &SearchConfig,
    cfg_fast: &SearchConfig,
    pcr_full_prob: f64,
    temp_moves: u32,
    seed: u64,
    version: &str,
    decode_stats: &Arc<(AtomicU64, AtomicU64)>,
) -> ThreadOut {
    let lane = Arc::new(Mutex::new(Lane {
        request: (0..in_flight).map(|_| None).collect(),
        result: (0..in_flight).map(|_| None).collect(),
    }));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut futs: Vec<Option<GameFut>> = (0..in_flight).map(|_| None).collect();
    let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel::<WaveOut>(1);
    let mut out = ThreadOut::default();

    loop {
        // Replenish: keep the pool full while the global game budget remains.
        for slot in 0..in_flight {
            if futs[slot].is_none() {
                let g = next_game.fetch_add(1, Ordering::Relaxed);
                if g < n_games {
                    {
                        let mut l = lane.lock().unwrap();
                        l.request[slot] = None;
                        l.result[slot] = None;
                    }
                    futs[slot] = Some(make_game(
                        slot,
                        lane.clone(),
                        version.to_string(),
                        cfg_full.clone(),
                        cfg_fast.clone(),
                        pcr_full_prob,
                        temp_moves,
                        seed.wrapping_add(g as u64),
                        decode_stats.clone(),
                    ));
                }
                // else: budget exhausted; leave the slot empty.
            }
        }

        // Poll every active game once (consume last round's results; yield next op).
        let mut any = false;
        for slot in 0..in_flight {
            if let Some(fut) = futs[slot].as_mut() {
                any = true;
                if let Poll::Ready(res) = fut.as_mut().poll(&mut cx) {
                    match res {
                        Ok((recs, reif)) => {
                            out.records.extend(recs);
                            out.reifications += reif as u64;
                        }
                        Err(e) => eprintln!("lockstep game error: {e}"),
                    }
                    futs[slot] = None;
                }
            }
        }
        if !any {
            break; // no active games and the budget is exhausted
        }

        // Drain this round's ops (brief lane lock — the lane is per-worker).
        let mut reqs: Vec<(usize, Pending)> = Vec::with_capacity(in_flight);
        {
            let mut l = lane.lock().unwrap();
            for slot in 0..in_flight {
                if let Some(p) = l.request[slot].take() {
                    reqs.push((slot, p));
                }
            }
        }

        // Resolve fused encode/policy cache hits (sharded, short locks); collect
        // misses (+ all eval/reify) into a wave.
        let mut wave = Wave::default();
        let mut enc_slot: Vec<usize> = Vec::new();
        let mut enc_key: Vec<u64> = Vec::new();
        let mut eval_slot: Vec<usize> = Vec::new();
        let mut pol_slot: Vec<usize> = Vec::new();
        let mut pol_key: Vec<u64> = Vec::new();
        let mut reif_slot: Vec<usize> = Vec::new();
        let mut hits_out: Vec<(usize, OpOut)> = Vec::new();
        for (slot, p) in reqs {
            match p {
                Pending::EncodeEval(p) => {
                    let k = hash_f32s(&p);
                    if let Some(v) = cache.enc.get(k) {
                        let (lgs, val) = v.as_ref();
                        hits_out.push((slot, OpOut::LgsVal(lgs.clone(), *val)));
                        cache.hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        enc_slot.push(slot);
                        enc_key.push(k);
                        wave.enc_eval.push(p);
                        cache.misses.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Pending::PolicyEval(x) => {
                    let k = hash_f32s(&x);
                    if let Some(v) = cache.pol.get(k) {
                        let ((_, w, mu, std), val) = v.as_ref();
                        hits_out.push((slot, OpOut::PolVal(
                            MixturePolicy { dim: LGS_DIM, weights: w.clone(), means: mu.clone(), stds: std.clone() },
                            *val,
                        )));
                        cache.hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        pol_slot.push(slot);
                        pol_key.push(k);
                        wave.pol_eval.push(x);
                        cache.misses.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Pending::Eval(x) => {
                    eval_slot.push(slot);
                    wave.eval.push(x);
                }
                Pending::Reify(to) => {
                    reif_slot.push(slot);
                    wave.reif.push(to);
                }
            }
        }
        if !hits_out.is_empty() {
            let mut l = lane.lock().unwrap();
            for (slot, o) in hits_out {
                l.result[slot] = Some(o);
            }
        }

        if wave.is_empty() {
            continue; // every active game resolved via cache this round
        }
        tx.send(Job { wave, resp: resp_tx.clone() }).expect("inference server gone");
        let wo = resp_rx.recv().expect("inference response dropped");

        // Populate caches (sharded), then write results back (brief lane lock).
        for (i, &k) in enc_key.iter().enumerate() {
            cache.enc.insert(k, Arc::new(wo.enc_eval[i].clone()));
        }
        for (i, &k) in pol_key.iter().enumerate() {
            cache.pol.insert(k, Arc::new(wo.pol_eval[i].clone()));
        }
        {
            let mut l = lane.lock().unwrap();
            for (i, &slot) in enc_slot.iter().enumerate() {
                let (lgs, val) = &wo.enc_eval[i];
                l.result[slot] = Some(OpOut::LgsVal(lgs.clone(), *val));
            }
            for (i, &slot) in pol_slot.iter().enumerate() {
                let ((_, w, mu, std), val) = &wo.pol_eval[i];
                l.result[slot] = Some(OpOut::PolVal(
                    MixturePolicy { dim: LGS_DIM, weights: w.clone(), means: mu.clone(), stds: std.clone() },
                    *val,
                ));
            }
            for (i, &slot) in eval_slot.iter().enumerate() {
                l.result[slot] = Some(OpOut::Val(wo.eval[i]));
            }
            for (i, &slot) in reif_slot.iter().enumerate() {
                l.result[slot] = Some(OpOut::Decoded(wo.reif[i].clone()));
            }
        }
    }
    out
}

/// Central-batched LGS self-play. A pool of `threads × in_flight` games (kept
/// full by replenishment from `n_games`) feeds one inference server batching to
/// `max_batch`. Writes a v2 dump.
#[allow(clippy::too_many_arguments)]
pub fn run_generate_batched(
    checkpoint: &str,
    cuda: bool,
    out_path: &str,
    n_games: usize,
    in_flight: usize,
    threads: usize,
    max_batch: usize,
    sims: u32,
    fast_sims: u32,
    pcr_full_prob: f64,
    temp_moves: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let server = LgsServer::load_on(checkpoint, device)?;
    let version = server.model_version.clone();
    let cfg_full = SearchConfig { simulations: sims, c_puct, kernel_bandwidth, root_explore: true, seed: None, ..Default::default() };
    let cfg_fast = SearchConfig { simulations: fast_sims, c_puct, kernel_bandwidth, root_explore: false, seed: None, ..Default::default() };
    let pool = threads * in_flight;
    eprintln!(
        "c4 LGS self-play (central-batched): {n_games} games, pool {pool} ({threads}×{in_flight}), \
         max_batch {max_batch}, PCR full {sims}/fast {fast_sims} (p_full {pcr_full_prob}), h {kernel_bandwidth}, device {}, model {checkpoint} -> {out_path}",
        if cuda { "cuda" } else { "cpu" }
    );

    let t0 = Instant::now();
    let cache = Cache::default();
    let stats = ServerStats::default();
    let next_game = AtomicUsize::new(0);
    // (valid_decodes, total_decodes) across all workers — valid-decode-rate metric.
    let decode_stats = Arc::new((AtomicU64::new(0), AtomicU64::new(0)));
    let (tx, rx) = std::sync::mpsc::channel::<Job>();

    let outs: Vec<ThreadOut> = std::thread::scope(|scope| {
        let server_ref = &server;
        let stats_ref = &stats;
        scope.spawn(move || inference_server(rx, server_ref, max_batch, stats_ref));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let tx = tx.clone();
            let cache = &cache;
            let next_game = &next_game;
            let cfg_full = &cfg_full;
            let cfg_fast = &cfg_fast;
            let version = version.as_str();
            let decode_stats = &decode_stats;
            handles.push(scope.spawn(move || {
                worker(&tx, cache, in_flight, next_game, n_games, cfg_full, cfg_fast, pcr_full_prob, temp_moves, seed, version, decode_stats)
            }));
        }
        drop(tx); // let the server detect shutdown once workers finish
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut all: Vec<(MoveRec, f32)> = Vec::new();
    let mut reif = 0u64;
    for o in outs {
        all.extend(o.records);
        reif += o.reifications;
    }
    write_dump(out_path, &all)?;
    let dt = t0.elapsed().as_secs_f64();
    let hits = cache.hits.load(Ordering::Relaxed);
    let misses = cache.misses.load(Ordering::Relaxed);
    let hit_rate = if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 };
    let rounds = stats.rounds.load(Ordering::Relaxed).max(1);
    let ops = stats.ops.load(Ordering::Relaxed);
    let dec_valid = decode_stats.0.load(Ordering::Relaxed);
    let dec_total = decode_stats.1.load(Ordering::Relaxed);
    let dec_rate = if dec_total > 0 { 100.0 * dec_valid as f64 / dec_total as f64 } else { 0.0 };
    eprintln!(
        "  wrote {} examples ({} games) in {:.1}s  ({:.1} games/s, {} reifications)\n  \
         avg_batch {:.0}  forwards/s {:.0} ({:.1}k)  cache_hit {:.0}% (encode+policy)\n  \
         valid_decode {:.0}% ({}/{})\n  \
         ops: enc {} eval {} pol {} reif {}",
        all.len(),
        n_games,
        dt,
        n_games as f64 / dt,
        reif,
        ops as f64 / rounds as f64,
        ops as f64 / dt,
        ops as f64 / dt / 1000.0,
        100.0 * hit_rate,
        dec_rate,
        dec_valid,
        dec_total,
        stats.n_enc.load(Ordering::Relaxed),
        stats.n_eval.load(Ordering::Relaxed),
        stats.n_pol.load(Ordering::Relaxed),
        stats.n_reif.load(Ordering::Relaxed),
    );
    Ok(())
}

/// Absolute-strength eval for the LGS net (the deferred Stage 4c). From random
/// openings, run the KR-PUCT latent search and compare its chosen move + root
/// value to the perfect solver: `move_agree` is how often the net plays a
/// solver-optimal column, `value_sign_agree` how often it knows who's winning.
/// Single-thread, argmax (no temperature / root noise). Prints a human line and
/// a machine-readable `LGSEVAL <move%> <sign%> <n>` line for log scraping.
#[allow(clippy::too_many_arguments)]
pub async fn run_lgs_eval(
    checkpoint: &str,
    cuda: bool,
    positions: usize,
    opening: u32,
    sims: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    use playground_core::Solver;
    let device = crate::net::pick_device(cuda)?;
    let model = LgsModel::load_on(checkpoint, device)?;
    let cfg =
        SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let (mut move_ok, mut sign_ok, mut n) = (0usize, 0usize, 0usize);
    let mut attempts = 0usize;
    while n < positions && attempts < positions * 20 {
        attempts += 1;
        // Random opening; reject if it ends the game before `opening` plies.
        let mut state = Connect4::initial_state();
        for _ in 0..opening {
            if Connect4::outcome(&state).is_terminal() {
                break;
            }
            let moves = Connect4::legal_moves(&state);
            let mv = moves[rng.gen_range(0..moves.len())].clone();
            state = Connect4::apply(&state, &mv);
        }
        if Connect4::outcome(&state).is_terminal() {
            continue;
        }
        let res = search(&model, &state, &cfg).await?;
        let sol = Connect4::choose_move(&state);
        if res.chosen.col == sol.best_move.col {
            move_ok += 1;
        }
        if res.root_value.signum() == (sol.value as f64).signum() {
            sign_ok += 1;
        }
        n += 1;
    }
    let nn = n.max(1) as f64;
    let (mp, sp) = (100.0 * move_ok as f64 / nn, 100.0 * sign_ok as f64 / nn);
    eprintln!(
        "lgs_eval_vs_solver  move_agree {mp:.1}%  value_sign_agree {sp:.1}%  ({n} positions, {sims} sims, opening {opening})"
    );
    println!("LGSEVAL {mp:.2} {sp:.2} {n}");
    Ok(())
}

/// Probe a single position (given as a comma-separated opening move sequence):
/// print the net's chosen move, root value, per-child visits, and the solver's
/// optimal move — for manual degeneracy inspection. `OK`/`WARN` flags obvious
/// pathologies (illegal/own-loss, ignored immediate win, ignored forced block).
pub async fn run_lgs_move(
    checkpoint: &str,
    cuda: bool,
    moves: &str,
    sims: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
) -> anyhow::Result<()> {
    use playground_core::Solver;
    let device = crate::net::pick_device(cuda)?;
    let model = LgsModel::load_on(checkpoint, device)?;
    let cfg =
        SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };

    let mut state = Connect4::initial_state();
    for tok in moves.split(',').filter(|s| !s.trim().is_empty()) {
        let col: u8 = tok.trim().parse()?;
        state = Connect4::apply(&state, &Connect4Move { col });
    }
    if Connect4::outcome(&state).is_terminal() {
        println!("terminal position, no move");
        return Ok(());
    }
    let res = search(&model, &state, &cfg).await?;
    let sol = Connect4::choose_move(&state);
    let mut kids: Vec<(u8, u32, f64)> =
        res.root_children.iter().map(|c| (c.mv.col, c.visits, c.value)).collect();
    kids.sort_by(|a, b| b.1.cmp(&a.1));
    let flag = if res.chosen.col == sol.best_move.col { "OK" } else { "DIFF-from-solver" };
    eprintln!(
        "lgs_move  after [{moves}]: net plays col {} (root_value {:.3}); solver {} ({flag})\n  children (col,visits,value): {:?}",
        res.chosen.col, res.root_value, sol.best_move.col, kids
    );
    println!("MOVE {} {} {:.4}", res.chosen.col, sol.best_move.col, res.root_value);
    Ok(())
}

/// A match player: an LGS checkpoint (argmax KR-PUCT search) or the perfect solver.
enum Player {
    Lgs(LgsModel),
    Solver,
}

impl Player {
    fn load(spec: &str, device: &Device) -> anyhow::Result<Self> {
        Ok(if spec == "solver" {
            Player::Solver
        } else {
            Player::Lgs(LgsModel::load_on(spec, device.clone())?)
        })
    }
    async fn pick(&self, state: &Connect4State, cfg: &SearchConfig) -> anyhow::Result<u8> {
        use playground_core::Solver;
        Ok(match self {
            Player::Solver => Connect4::choose_move(state).best_move.col,
            Player::Lgs(m) => search(m, state, cfg).await?.chosen.col,
        })
    }
}

/// Head-to-head match between two players (LGS checkpoint paths or "solver").
/// Each game starts from a random opening (for variety); the two engines are
/// swapped between which side moves first on alternating games (colour balance).
/// Prints `MATCH <a_wins> <draws> <b_wins>` from A's perspective — the raw counts
/// the BT-ELO ladder consumes.
#[allow(clippy::too_many_arguments)]
pub async fn run_lgs_match(
    a_spec: &str,
    b_spec: &str,
    cuda: bool,
    games: usize,
    sims: u32,
    opening: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let a = Player::load(a_spec, &device)?;
    let b = Player::load(b_spec, &device)?;
    let cfg =
        SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let (mut a_wins, mut draws, mut b_wins) = (0usize, 0usize, 0usize);
    let mut g = 0usize;
    let mut attempts = 0usize;
    while g < games && attempts < games * 20 {
        attempts += 1;
        let mut state = Connect4::initial_state();
        for _ in 0..opening {
            if Connect4::outcome(&state).is_terminal() {
                break;
            }
            let moves = Connect4::legal_moves(&state);
            let mv = moves[rng.gen_range(0..moves.len())].clone();
            state = Connect4::apply(&state, &mv);
        }
        if Connect4::outcome(&state).is_terminal() {
            continue;
        }
        // A controls the side to move at the opening on even games, B on odd.
        let mut a_to_move = g % 2 == 0;
        let outcome = loop {
            let o = Connect4::outcome(&state);
            if o.is_terminal() {
                break o;
            }
            let col = if a_to_move {
                a.pick(&state, &cfg).await?
            } else {
                b.pick(&state, &cfg).await?
            };
            state = Connect4::apply(&state, &Connect4Move { col });
            a_to_move = !a_to_move;
        };
        match outcome {
            Outcome::Draw => draws += 1,
            // The last mover won; `a_to_move` has flipped to the loser's side.
            _ => {
                if !a_to_move {
                    a_wins += 1
                } else {
                    b_wins += 1
                }
            }
        }
        g += 1;
    }
    eprintln!(
        "lgs_match  {a_spec} vs {b_spec}:  {a_wins}-{draws}-{b_wins} (W-D-L, A's view; {g} games, {sims} sims)"
    );
    println!("MATCH {a_wins} {draws} {b_wins}");
    Ok(())
}
