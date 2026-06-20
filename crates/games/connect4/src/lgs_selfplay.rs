//! LGS self-play for Connect 4 — generates v2 training records from the KR-PUCT
//! latent search (docs/algorithm.md).
//!
//! Two drivers share `play_game` and the v2 record format:
//!   * `run_generate` — single game at a time (correctness-first).
//!   * `run_generate_batched` — synchronous **lockstep**: many games advance in
//!     lockstep per worker thread; each wave coalesces the games' net ops into one
//!     batched forward per op type (no channels / async overhead). This is the
//!     throughput path (we diagnosed the async-server path as CPU/overhead-bound
//!     with the GPU starved).
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Instant;

use async_trait::async_trait;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use playground_core::latent_mcts::{search, RootChild, SearchConfig};
use playground_core::{Game, LatentGameState, LatentModel, MixturePolicy, Outcome};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS};
use crate::lgs_net::{LgsModel, LgsServer};
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
async fn play_game<M: LatentModel<Connect4>>(
    model: &M,
    cfg: &SearchConfig,
    temp_moves: u32,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(Vec<(MoveRec, f32)>, u32)> {
    let mut state = Connect4::initial_state();
    let mut recs: Vec<MoveRec> = Vec::new();
    let mut reifications = 0u32;

    while !Connect4::outcome(&state).is_terminal() {
        let res = search(model, &state, cfg).await?;
        reifications += res.reifications;
        if res.root_children.is_empty() {
            break; // tiny budget produced no reified child; bail safely
        }
        let sample = Connect4::ply(&state) < temp_moves;
        let pick = choose(&res.root_children, sample, rng);
        let children = res
            .root_children
            .iter()
            .map(|c| (c.mv.col, c.state.clone(), c.visits))
            .collect();
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
    temp_moves: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LgsModel::load_on(checkpoint, device)?);
    let cfg = SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };
    eprintln!(
        "c4 LGS self-play: {n_games} games, {sims} sims/move, h {kernel_bandwidth}, model {checkpoint} -> {out_path}"
    );
    let t0 = Instant::now();
    let mut all: Vec<(MoveRec, f32)> = Vec::new();
    let mut reifications = 0u64;
    for g in 0..n_games {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(g as u64));
        let (recs, reif) = play_game(model.as_ref(), &cfg, temp_moves, &mut rng).await?;
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

// ── Lockstep throughput path ──────────────────────────────────────────────────
//
// Each worker thread drives `in_flight` game futures (reusing the async
// latent_mcts::search) in lockstep. A "wave" = poll every active game once; each
// yields exactly one net op (encode/eval/policy/reify) or completes. We collect
// the wave's ops, run one batched forward per type on the shared net, write the
// results back, and poll again. Batch size ≈ in-flight games; no channels / async
// runtime / per-op task churn. `T` threads run independent waves for parallel
// tree work (the diagnosed CPU bottleneck).

/// A net op a game yields this wave.
enum Pending {
    Encode(Vec<f32>),
    Eval(Vec<f32>),
    Policy(Vec<f32>),
    Reify(Vec<f32>, Vec<f32>),
}

/// The driver's reply to a yielded op.
enum OpOut {
    Lgs(Vec<f32>),
    Val(f32),
    Pol(MixturePolicy),
    Logits(Vec<f32>),
}

type PolOut = (usize, Vec<f32>, Vec<f32>, Vec<f32>);

/// Transposition cache over *real-state* forwards (`encode`, `policy`), shared
/// across worker threads. Connect 4 transposes heavily, so the same board state
/// recurs across games/searches; caching removes the (dominant) conv forwards.
/// `eval`/`reify` are not cached — they hit unique sampled latents and would
/// pollute the cache. Keyed by a hash of the input floats.
#[derive(Default)]
struct Cache {
    encode: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
    policy: Mutex<HashMap<u64, Arc<PolOut>>>,
    hits: AtomicU64,
    misses: AtomicU64,
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

/// Per-thread wave state: each slot holds at most one pending request + its
/// result. The `Mutex` only satisfies `LatentModel: Send + Sync`; it is
/// uncontended (a single driver thread per lane).
struct Lane {
    request: Vec<Option<Pending>>,
    result: Vec<Option<OpOut>>,
}

/// A `LatentModel` bound to one game slot: registers each op into the lane and
/// awaits the result the driver writes after the batched forward.
struct LockstepModel {
    slot: usize,
    lane: Arc<Mutex<Lane>>,
    version: String,
}

/// Future that resolves once the driver fills `lane.result[slot]`.
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
        match self.submit(Pending::Encode(planes_of(state).to_vec())).await {
            OpOut::Lgs(v) => Ok(LatentGameState::from_f32(&v, self.version.clone())),
            _ => unreachable!("encode reply"),
        }
    }
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        match self.submit(Pending::Eval(lgs.to_f32())).await {
            OpOut::Val(v) => Ok(v as f64),
            _ => unreachable!("eval reply"),
        }
    }
    async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy> {
        match self.submit(Pending::Policy(lgs.to_f32())).await {
            OpOut::Pol(p) => Ok(p),
            _ => unreachable!("policy reply"),
        }
    }
    async fn reify(
        &self,
        from_state: &Connect4State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<Connect4Move> {
        let logits = match self.submit(Pending::Reify(from_lgs.to_f32(), to_lgs.to_f32())).await {
            OpOut::Logits(l) => l,
            _ => unreachable!("reify reply"),
        };
        let legal: Vec<u8> = Connect4::legal_moves(from_state).iter().map(|m| m.col).collect();
        let best = legal
            .iter()
            .copied()
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]));
        Ok(Connect4Move { col: best.or_else(|| legal.first().copied()).unwrap_or(0) })
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
    cfg: SearchConfig,
    temp_moves: u32,
    seed: u64,
) -> GameFut {
    Box::pin(async move {
        let model = LockstepModel { slot, lane, version };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        play_game(&model, &cfg, temp_moves, &mut rng).await
    })
}

#[derive(Default)]
struct ThreadOut {
    records: Vec<(MoveRec, f32)>,
    reifications: u64,
    batches: u64,
    samples: u64,
}

/// Drive games `[game_start, game_end)` in lockstep, `in_flight` at a time.
#[allow(clippy::too_many_arguments)]
fn drive_thread(
    server: &LgsServer,
    cache: &Cache,
    in_flight: usize,
    game_start: usize,
    game_end: usize,
    cfg: &SearchConfig,
    temp_moves: u32,
    seed: u64,
    version: &str,
) -> ThreadOut {
    let lane = Arc::new(Mutex::new(Lane {
        request: (0..in_flight).map(|_| None).collect(),
        result: (0..in_flight).map(|_| None).collect(),
    }));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut futs: Vec<Option<GameFut>> = (0..in_flight).map(|_| None).collect();
    let mut next_game = game_start;
    let mut out = ThreadOut::default();

    loop {
        // Fill free slots with new games while this thread's share remains.
        for slot in 0..in_flight {
            if futs[slot].is_none() && next_game < game_end {
                {
                    let mut l = lane.lock().unwrap();
                    l.request[slot] = None;
                    l.result[slot] = None;
                }
                futs[slot] = Some(make_game(
                    slot,
                    lane.clone(),
                    version.to_string(),
                    cfg.clone(),
                    temp_moves,
                    seed.wrapping_add(next_game as u64),
                ));
                next_game += 1;
            }
        }

        // Poll every active slot once: it consumes last wave's result, runs tree
        // work, and yields its next op (Pending) or completes (Ready).
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
        if !any && next_game >= game_end {
            break;
        }
        process_wave(server, cache, &lane, in_flight, &mut out);
    }
    out
}

/// Collect the wave's pending ops, serve `encode`/`policy` from the transposition
/// cache where possible, run one batched forward per type over the misses, write
/// results back. `eval`/`reify` are always forwarded (uncacheable inputs).
fn process_wave(server: &LgsServer, cache: &Cache, lane: &Arc<Mutex<Lane>>, in_flight: usize, out: &mut ThreadOut) {
    let mut enc: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut ev: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut pol: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut rei: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
    {
        let mut l = lane.lock().unwrap();
        for slot in 0..in_flight {
            match l.request[slot].take() {
                Some(Pending::Encode(p)) => enc.push((slot, p)),
                Some(Pending::Eval(x)) => ev.push((slot, x)),
                Some(Pending::Policy(x)) => pol.push((slot, x)),
                Some(Pending::Reify(a, b)) => rei.push((slot, a, b)),
                None => {}
            }
        }
    }
    let total = enc.len() + ev.len() + pol.len() + rei.len();
    if total == 0 {
        return;
    }
    out.batches += 1;
    out.samples += total as u64;

    // ENCODE — cached by planes (real states only).
    if !enc.is_empty() {
        let mut miss: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut miss_keys: Vec<u64> = Vec::new();
        {
            let cmap = cache.encode.lock().unwrap();
            let mut l = lane.lock().unwrap();
            for (slot, p) in enc {
                let k = hash_f32s(&p);
                if let Some(v) = cmap.get(&k) {
                    l.result[slot] = Some(OpOut::Lgs(v.as_ref().clone()));
                    cache.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    miss_keys.push(k);
                    miss.push((slot, p));
                    cache.misses.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if !miss.is_empty() {
            let b = miss.len();
            let mut flat = Vec::with_capacity(b * PLANES);
            for (_, p) in &miss {
                flat.extend_from_slice(p);
            }
            let r = server.encode_batch(&flat, b).expect("encode_batch");
            let mut cmap = cache.encode.lock().unwrap();
            let mut l = lane.lock().unwrap();
            for (i, (slot, _)) in miss.into_iter().enumerate() {
                let v = r[i * LGS_DIM..(i + 1) * LGS_DIM].to_vec();
                cmap.insert(miss_keys[i], Arc::new(v.clone()));
                l.result[slot] = Some(OpOut::Lgs(v));
            }
        }
    }

    // POLICY — cached by lgs (real node states only).
    if !pol.is_empty() {
        let mut miss: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut miss_keys: Vec<u64> = Vec::new();
        {
            let cmap = cache.policy.lock().unwrap();
            let mut l = lane.lock().unwrap();
            for (slot, x) in pol {
                let k = hash_f32s(&x);
                if let Some(v) = cmap.get(&k) {
                    let (kk, w, mu, std) = v.as_ref();
                    l.result[slot] = Some(OpOut::Pol(MixturePolicy {
                        dim: LGS_DIM,
                        weights: w.clone(),
                        means: mu.clone(),
                        stds: std.clone(),
                    }));
                    let _ = kk;
                    cache.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    miss_keys.push(k);
                    miss.push((slot, x));
                    cache.misses.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if !miss.is_empty() {
            let b = miss.len();
            let mut flat = Vec::with_capacity(b * LGS_DIM);
            for (_, x) in &miss {
                flat.extend_from_slice(x);
            }
            let (k, w, mu, std) = server.policy_batch(&flat, b).expect("policy_batch");
            let md = k * LGS_DIM;
            let mut cmap = cache.policy.lock().unwrap();
            let mut l = lane.lock().unwrap();
            for (i, (slot, _)) in miss.into_iter().enumerate() {
                let wi = w[i * k..(i + 1) * k].to_vec();
                let mui = mu[i * md..(i + 1) * md].to_vec();
                let stdi = std[i * md..(i + 1) * md].to_vec();
                cmap.insert(miss_keys[i], Arc::new((k, wi.clone(), mui.clone(), stdi.clone())));
                l.result[slot] = Some(OpOut::Pol(MixturePolicy { dim: LGS_DIM, weights: wi, means: mui, stds: stdi }));
            }
        }
    }

    // EVAL — always forwarded (candidate evals hit unique sampled latents).
    if !ev.is_empty() {
        let b = ev.len();
        let mut flat = Vec::with_capacity(b * LGS_DIM);
        for (_, x) in &ev {
            flat.extend_from_slice(x);
        }
        let r = server.eval_batch(&flat, b).expect("eval_batch");
        let mut l = lane.lock().unwrap();
        for (i, (slot, _)) in ev.into_iter().enumerate() {
            l.result[slot] = Some(OpOut::Val(r[i]));
        }
    }

    // REIFY — always forwarded.
    if !rei.is_empty() {
        let b = rei.len();
        let mut from = Vec::with_capacity(b * LGS_DIM);
        let mut to = Vec::with_capacity(b * LGS_DIM);
        for (_, a, c) in &rei {
            from.extend_from_slice(a);
            to.extend_from_slice(c);
        }
        let r = server.reify_batch(&from, &to, b).expect("reify_batch");
        let mut l = lane.lock().unwrap();
        for (i, (slot, _, _)) in rei.into_iter().enumerate() {
            l.result[slot] = Some(OpOut::Logits(r[i * C4_COLS..(i + 1) * C4_COLS].to_vec()));
        }
    }
}

/// Lockstep LGS self-play across `threads` worker threads, `in_flight` games per
/// thread feeding one batched forward per op type per wave. Writes a v2 dump.
#[allow(clippy::too_many_arguments)]
pub fn run_generate_batched(
    checkpoint: &str,
    cuda: bool,
    out_path: &str,
    n_games: usize,
    in_flight: usize,
    threads: usize,
    sims: u32,
    temp_moves: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let server = Arc::new(LgsServer::load_on(checkpoint, device)?);
    let version = server.model_version.clone();
    let cfg = SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };
    eprintln!(
        "c4 LGS self-play (lockstep): {n_games} games, {threads} threads × {in_flight} in flight, \
         {sims} sims/move, h {kernel_bandwidth}, device {}, model {checkpoint} -> {out_path}",
        if cuda { "cuda" } else { "cpu" }
    );

    let t0 = Instant::now();
    let cache = Arc::new(Cache::default());
    let per = n_games.div_ceil(threads.max(1));
    let outs: Vec<ThreadOut> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let start = t * per;
            let end = ((t + 1) * per).min(n_games);
            if start >= end {
                break;
            }
            let server = server.clone();
            let cache = cache.clone();
            let cfg = cfg.clone();
            let version = version.clone();
            handles.push(scope.spawn(move || {
                drive_thread(&server, &cache, in_flight, start, end, &cfg, temp_moves, seed, &version)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut all: Vec<(MoveRec, f32)> = Vec::new();
    let (mut reif, mut batches, mut samples) = (0u64, 0u64, 0u64);
    for o in outs {
        all.extend(o.records);
        reif += o.reifications;
        batches += o.batches;
        samples += o.samples;
    }
    write_dump(out_path, &all)?;
    let dt = t0.elapsed().as_secs_f64();
    let hits = cache.hits.load(Ordering::Relaxed);
    let misses = cache.misses.load(Ordering::Relaxed);
    let hit_rate = if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 };
    eprintln!(
        "  wrote {} examples ({} games) in {:.1}s  ({:.1} games/s, {} reifications)\n  \
         avg_batch {:.0}  net-ops/s {:.0}  ({:.1}k forwards/s)  cache_hit {:.0}% (encode+policy)",
        all.len(),
        n_games,
        dt,
        n_games as f64 / dt,
        reif,
        samples as f64 / batches.max(1) as f64,
        samples as f64 / dt,
        samples as f64 / dt / 1000.0,
        100.0 * hit_rate,
    );
    Ok(())
}
