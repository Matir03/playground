//! LGS self-play for Connect 4 — generates v2 training records from the KR-PUCT
//! latent search (docs/algorithm.md). Single-threaded and correctness-first; the
//! batched / PCR throughput path is stage 3.
//!
//! Record v2 (fixed stride `REC_F32` little-endian f32 per training example):
//!   parent planes (PLANES) | z (1) | MAX_CHILDREN × [col, visits, child planes]
//! Absent child slots are zero-filled (visits = 0). The trainer re-encodes the
//! parent and each child with the live net, so targets live in the current
//! latent space (the policy/reifier/consistency losses need that).

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

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

/// Generate `n_games` of LGS self-play and write a v2 dump to `out_path`.
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
    let cfg = SearchConfig {
        simulations: sims,
        c_puct,
        kernel_bandwidth,
        seed: None,
        ..Default::default()
    };
    eprintln!(
        "c4 LGS self-play: {n_games} games, {sims} sims/move, h {kernel_bandwidth}, model {checkpoint} -> {out_path}"
    );
    let t0 = std::time::Instant::now();
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

// ── Batched throughput path ───────────────────────────────────────────────────
//
// One GPU-owning server thread coalesces encode/eval/policy/reify requests from
// many concurrent async games into batched forwards. Reuses the async
// `latent_mcts::search` unchanged; concurrency (not thread count) drives batch
// size, so a few hundred in-flight games give large GPU batches. The candle
// backend is device-agnostic — `--cuda` runs it on the GPU.

/// A batched-inference request: payload + a oneshot for the typed reply.
enum Req {
    Encode(Vec<f32>, oneshot::Sender<Vec<f32>>),
    Eval(Vec<f32>, oneshot::Sender<f32>),
    Policy(Vec<f32>, oneshot::Sender<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>),
    Reify(Vec<f32>, Vec<f32>, oneshot::Sender<Vec<f32>>),
}

/// Inference server loop (runs on its own OS thread): block for a request, then
/// **linger** briefly to accumulate a full batch (woken games re-queue their next
/// request on the tokio workers; without a linger each forward sees a near-empty
/// queue and is launch/sync-bound), run one batched forward per op type, scatter.
fn serve(
    mut rx: mpsc::UnboundedReceiver<Req>,
    server: LgsServer,
    max_batch: usize,
    linger_us: u64,
    counters: Arc<(AtomicU64, AtomicU64)>, // (batches, samples)
) {
    loop {
        let first = match rx.blocking_recv() {
            Some(r) => r,
            None => break, // all senders dropped: shut down
        };
        let mut batch = vec![first];
        let deadline = Instant::now() + Duration::from_micros(linger_us);
        while batch.len() < max_batch {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => {
                    if linger_us == 0 || Instant::now() >= deadline {
                        break;
                    }
                    std::thread::yield_now(); // spin until more arrive or deadline
                }
            }
        }
        counters.0.fetch_add(1, Ordering::Relaxed);
        counters.1.fetch_add(batch.len() as u64, Ordering::Relaxed);

        let mut enc = Vec::new();
        let mut ev = Vec::new();
        let mut pol = Vec::new();
        let mut rei = Vec::new();
        for r in batch {
            match r {
                Req::Encode(p, tx) => enc.push((p, tx)),
                Req::Eval(l, tx) => ev.push((l, tx)),
                Req::Policy(l, tx) => pol.push((l, tx)),
                Req::Reify(a, b, tx) => rei.push((a, b, tx)),
            }
        }

        if !enc.is_empty() {
            let b = enc.len();
            let mut flat = Vec::with_capacity(b * PLANES);
            for (p, _) in &enc {
                flat.extend_from_slice(p);
            }
            if let Ok(out) = server.encode_batch(&flat, b) {
                for (i, (_, tx)) in enc.into_iter().enumerate() {
                    let _ = tx.send(out[i * LGS_DIM..(i + 1) * LGS_DIM].to_vec());
                }
            }
        }
        if !ev.is_empty() {
            let b = ev.len();
            let mut flat = Vec::with_capacity(b * LGS_DIM);
            for (l, _) in &ev {
                flat.extend_from_slice(l);
            }
            if let Ok(out) = server.eval_batch(&flat, b) {
                for (i, (_, tx)) in ev.into_iter().enumerate() {
                    let _ = tx.send(out[i]);
                }
            }
        }
        if !pol.is_empty() {
            let b = pol.len();
            let mut flat = Vec::with_capacity(b * LGS_DIM);
            for (l, _) in &pol {
                flat.extend_from_slice(l);
            }
            if let Ok((k, w, mu, std)) = server.policy_batch(&flat, b) {
                let md = k * LGS_DIM;
                for (i, (_, tx)) in pol.into_iter().enumerate() {
                    let _ = tx.send((
                        k,
                        w[i * k..(i + 1) * k].to_vec(),
                        mu[i * md..(i + 1) * md].to_vec(),
                        std[i * md..(i + 1) * md].to_vec(),
                    ));
                }
            }
        }
        if !rei.is_empty() {
            let b = rei.len();
            let mut from = Vec::with_capacity(b * LGS_DIM);
            let mut to = Vec::with_capacity(b * LGS_DIM);
            for (a, c, _) in &rei {
                from.extend_from_slice(a);
                to.extend_from_slice(c);
            }
            if let Ok(out) = server.reify_batch(&from, &to, b) {
                for (i, (_, _, tx)) in rei.into_iter().enumerate() {
                    let _ = tx.send(out[i * C4_COLS..(i + 1) * C4_COLS].to_vec());
                }
            }
        }
    }
}

/// A `LatentModel` that routes each call to the batched server and awaits.
struct BatchedLatentModel {
    tx: mpsc::UnboundedSender<Req>,
    version: String,
}

impl BatchedLatentModel {
    fn gone() -> anyhow::Error {
        anyhow::anyhow!("inference server gone")
    }
}

#[async_trait]
impl LatentModel<Connect4> for BatchedLatentModel {
    async fn encode(&self, state: &Connect4State) -> anyhow::Result<LatentGameState> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Req::Encode(planes_of(state).to_vec(), tx)).map_err(|_| Self::gone())?;
        let v = rx.await.map_err(|_| Self::gone())?;
        Ok(LatentGameState::from_f32(&v, self.version.clone()))
    }

    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Req::Eval(lgs.to_f32(), tx)).map_err(|_| Self::gone())?;
        Ok(rx.await.map_err(|_| Self::gone())? as f64)
    }

    async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Req::Policy(lgs.to_f32(), tx)).map_err(|_| Self::gone())?;
        let (_k, weights, means, stds) = rx.await.map_err(|_| Self::gone())?;
        Ok(MixturePolicy { dim: LGS_DIM, weights, means, stds })
    }

    async fn reify(
        &self,
        from_state: &Connect4State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<Connect4Move> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Req::Reify(from_lgs.to_f32(), to_lgs.to_f32(), tx))
            .map_err(|_| Self::gone())?;
        let logits = rx.await.map_err(|_| Self::gone())?;
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

/// Batched LGS self-play: `concurrency` games in flight feeding one batched
/// inference server. Writes a v2 dump. This is the throughput path (`--cuda`).
#[allow(clippy::too_many_arguments)]
pub async fn run_generate_batched(
    checkpoint: &str,
    cuda: bool,
    out_path: &str,
    n_games: usize,
    concurrency: usize,
    sims: u32,
    temp_moves: u32,
    c_puct: f64,
    kernel_bandwidth: f64,
    max_batch: usize,
    linger_us: u64,
    seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let server = LgsServer::load_on(checkpoint, device)?;
    let version = server.model_version.clone();
    let (tx, rx) = mpsc::unbounded_channel::<Req>();
    let counters = Arc::new((AtomicU64::new(0), AtomicU64::new(0)));
    let cs = counters.clone();
    let handle = std::thread::spawn(move || serve(rx, server, max_batch, linger_us, cs));
    let model = Arc::new(BatchedLatentModel { tx, version });
    let cfg = SearchConfig { simulations: sims, c_puct, kernel_bandwidth, seed: None, ..Default::default() };
    eprintln!(
        "c4 LGS self-play (batched): {n_games} games, {concurrency} in flight, {sims} sims/move, \
         max_batch {max_batch}, device {}, model {checkpoint} -> {out_path}",
        if cuda { "cuda" } else { "cpu" }
    );

    let t0 = Instant::now();
    let mut set: JoinSet<anyhow::Result<(Vec<(MoveRec, f32)>, u32)>> = JoinSet::new();
    let mut next = 0usize;
    let spawn = |set: &mut JoinSet<_>, gi: usize| {
        let m = model.clone();
        let cfgc = cfg.clone();
        set.spawn(async move {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(gi as u64));
            play_game(&*m, &cfgc, temp_moves, &mut rng).await
        });
    };
    while next < n_games && set.len() < concurrency {
        spawn(&mut set, next);
        next += 1;
    }
    let mut all: Vec<(MoveRec, f32)> = Vec::new();
    let mut reifications = 0u64;
    while let Some(res) = set.join_next().await {
        let (recs, reif) = res??;
        reifications += reif as u64;
        all.extend(recs);
        if next < n_games {
            spawn(&mut set, next);
            next += 1;
        }
    }
    drop(model); // last sender → server thread exits
    let _ = handle.join();

    write_dump(out_path, &all)?;
    let dt = t0.elapsed().as_secs_f64();
    let batches = counters.0.load(Ordering::Relaxed).max(1);
    let samples = counters.1.load(Ordering::Relaxed);
    let nps = samples as f64 / dt; // net forwards (encode/eval/policy/reify) per sec
    eprintln!(
        "  wrote {} examples ({} games) in {:.1}s  ({:.1} games/s, {} reifications)\n  \
         avg_batch {:.0}  net-ops/s {:.0}  ({:.1}k forwards/s effective)",
        all.len(),
        n_games,
        dt,
        n_games as f64 / dt,
        reifications,
        samples as f64 / batches as f64,
        nps,
        nps / 1000.0,
    );
    Ok(())
}
