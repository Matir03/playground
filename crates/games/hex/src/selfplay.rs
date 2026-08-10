//! Hex self-play: the continuous AZ actor and a throughput bench.
//!
//! Each ply of a finished game becomes one training example — the canonical
//! planes, the game outcome z from that position's side-to-move point of view
//! (±1; Hex has no draws), and the MCTS visit distribution over the 169 cells as
//! the policy target. Everything is in the canonical frame (see
//! `game::fill_planes_into`), so the learner needs no notion of colour at all.
//!
//! The target is stored **dense**: 169 floats is 676 bytes, small enough that the
//! sparse encoding amazons needs (2×7200) would only add complexity here.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use playground_core::batch_search::{self, BatchModel, MvGameRecord, SearchParams};

use crate::game::{canon_index, fill_planes_into, Hex, HexMove, N};
use crate::mcts::{PLANES, POLICY_DIM};
use crate::net::LocalModel;

/// Which inference backend runs the net during self-play.
///
/// `Candle` loads `latest.safetensors` in-process; `Ort` loads the learner's
/// `model.onnx` through ONNX Runtime on the CUDA or TensorRT execution provider.
/// TensorRT with fp16 is several times faster here (tensor cores + a fused
/// engine vs candle's per-op kernels), at the cost of a one-off engine build
/// per shape, cached under `<live>/trt_cache`.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Candle,
    /// `ep` is "trt" or "cuda"; `fp16` applies to TensorRT.
    Ort { ep: &'static str, fp16: bool },
}

impl Backend {
    pub fn is_ort(self) -> bool {
        matches!(self, Backend::Ort { .. })
    }

    pub fn label(self) -> String {
        match self {
            Backend::Candle => "candle".to_string(),
            Backend::Ort { ep, fp16 } => format!("ort/{ep}/fp16={fp16}"),
        }
    }

    /// The file whose mtime signals "there are new weights to play from".
    ///
    /// `best.*` when the evaluator has promoted something, otherwise the
    /// learner's latest. That indirection *is* the gate: the actor generates
    /// data with a net that has proven itself against MoHex, so a snapshot that
    /// got worse can't feed itself worse data (see `python/hex_eval_loop.py`).
    pub fn weights_path(self, live_dir: &str) -> String {
        let (best, latest) = match self {
            Backend::Candle => ("best.safetensors", "latest.safetensors"),
            Backend::Ort { .. } => ("best.onnx", "model.onnx"),
        };
        let promoted = format!("{live_dir}/{best}");
        if Path::new(&promoted).exists() {
            promoted
        } else {
            format!("{live_dir}/{latest}")
        }
    }

    fn load(self, live_dir: &str, device: Option<candle_core::Device>) -> anyhow::Result<Arc<dyn BatchModel>> {
        let path = self.weights_path(live_dir);
        tracing::debug!(%path, "loading self-play weights");
        match self {
            Backend::Candle => Ok(Arc::new(LocalModel::load_on(&path, device.unwrap())?)),
            Backend::Ort { ep, fp16 } => Ok(Arc::new(crate::ort_net::OrtModel::new(
                &path,
                ep,
                fp16,
                &format!("{live_dir}/trt_cache"),
            )?)),
        }
    }

    /// Same, for a checkpoint given directly (the bench takes a path, not a live dir).
    fn load_file(self, path: &str, device: Option<candle_core::Device>) -> anyhow::Result<Arc<dyn BatchModel>> {
        match self {
            Backend::Candle => Ok(Arc::new(LocalModel::load_on(path, device.unwrap())?)),
            Backend::Ort { ep, fp16 } => {
                let cache = Path::new(path)
                    .parent()
                    .map(|d| d.join("trt_cache").to_string_lossy().into_owned())
                    .unwrap_or_else(|| "trt_cache".to_string());
                Ok(Arc::new(crate::ort_net::OrtModel::new(path, ep, fp16, &cache)?))
            }
        }
    }
}

/// One self-play training example.
pub struct Example {
    pub planes: Vec<f32>, // PLANES floats, canonical frame
    pub value: f32,       // game outcome z, side-to-move POV
    pub policy: Vec<f32>, // POLICY_DIM visit probabilities, canonical frame
    /// Who ends up holding each cell when the game finishes: +1 this position's
    /// mover, -1 the opponent, 0 still empty. Canonical frame, like everything else.
    ///
    /// This is KataGo's ownership target, and it exists because one ±1 outcome per
    /// position is a starvation diet for a value head — especially in Hex, where a
    /// weak net rightly sees every opening as a coin flip and therefore learns
    /// nothing from the result. Final ownership is 169 supervised numbers per
    /// position instead of one, it is available for free from the finished game,
    /// and "which of us ends up holding this region" is exactly the structure a
    /// connection game turns on.
    pub ownership: Vec<f32>,
}

/// Turn finished games into examples. The winner is whoever was *not* to move in
/// the final position (a Hex game ends the moment someone connects), or the side
/// the loser resigned to; a game stopped by the ply backstop is scored 0.
pub fn examples_from_records(records: Vec<MvGameRecord<Hex>>) -> Vec<Example> {
    let mut out = Vec::new();
    for rec in records {
        let final_state = rec.states.last().expect("a game has at least the initial state");
        let winner: Option<u8> = if let Some(loser) = rec.resigned_by {
            Some(3 - loser)
        } else if final_state.winner != 0 {
            Some(final_state.winner)
        } else {
            None // hit the ply cap, undecided
        };
        for (i, mv) in rec.moves.iter().enumerate() {
            // Cheap searches (playout cap randomization) advanced the game but
            // aren't worth training on.
            if !mv.for_training {
                continue;
            }
            let total: u32 = mv.visits.iter().sum();
            if total == 0 {
                continue;
            }
            let state = &rec.states[i];
            let mut policy = vec![0f32; POLICY_DIM];
            for (m, &v) in mv.moves.iter().zip(&mv.visits) {
                if v == 0 {
                    continue;
                }
                if let HexMove::Play { r, c } = m {
                    let cell = *r as usize * N + *c as usize;
                    policy[canon_index(state, cell)] += v as f32 / total as f32;
                }
            }
            let z = match winner {
                Some(w) => {
                    if w == mv.to_move {
                        1.0
                    } else {
                        -1.0
                    }
                }
                None => 0.0,
            };
            let mut planes = vec![0f32; PLANES];
            fill_planes_into(state, &mut planes);
            // Ownership is read off the finished board, from this mover's side.
            let mut ownership = vec![0f32; POLICY_DIM];
            for cell in 0..POLICY_DIM {
                let owner = final_state.cell_at(cell);
                if owner != 0 {
                    ownership[canon_index(state, cell)] =
                        if owner == mv.to_move { 1.0 } else { -1.0 };
                }
            }
            out.push(Example { planes, value: z, policy, ownership });
        }
    }
    out
}

/// Bytes per example in the dump: planes, value, policy, ownership.
pub const EXAMPLE_FLOATS: usize = PLANES + 1 + POLICY_DIM + POLICY_DIM;

/// Binary dump read by `hex_az_learner.py`. Header: magic u32, version u32,
/// count u64. Per example: planes[PLANES f32], value f32, policy[POLICY_DIM f32],
/// ownership[POLICY_DIM f32]. All little-endian.
///
/// The header exists so a learner never silently misreads a chunk written by a
/// different build — the ownership target changed the layout once already.
pub const DUMP_MAGIC: u32 = 0x5845_4831; // "HEX1" little-endian
pub const DUMP_VERSION: u32 = 2;

pub fn write_dump(out_path: &str, examples: &[Example]) -> anyhow::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(16 + examples.len() * EXAMPLE_FLOATS * 4);
    buf.extend_from_slice(&DUMP_MAGIC.to_le_bytes());
    buf.extend_from_slice(&DUMP_VERSION.to_le_bytes());
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in examples {
        for &x in &ex.planes {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&ex.value.to_le_bytes());
        for &p in &ex.policy {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        for &o in &ex.ownership {
            buf.extend_from_slice(&o.to_le_bytes());
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

fn chunk_seq(entry: &std::fs::DirEntry) -> Option<u64> {
    let name = entry.file_name();
    let s = name.to_str()?;
    s.strip_prefix("chunk_")?.strip_suffix(".bin")?.parse::<u64>().ok()
}

fn max_chunk_seq(buffer: &Path) -> Option<u64> {
    std::fs::read_dir(buffer).ok()?.filter_map(|e| e.ok()).filter_map(|e| chunk_seq(&e)).max()
}

/// Keep only the newest `keep` chunks — the replay buffer's sliding window.
fn prune_buffer(buffer: &Path, keep: usize) {
    let mut chunks: Vec<(u64, PathBuf)> = std::fs::read_dir(buffer)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| chunk_seq(&e).map(|s| (s, e.path())))
        .collect();
    chunks.sort_unstable_by_key(|&(s, _)| s);
    if chunks.len() > keep {
        for (_, p) in &chunks[..chunks.len() - keep] {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Continuous actor: play games forever, flush finished ones to `<live>/buffer`
/// every `chunk_secs`, prune old chunks, and hot-reload the weights whenever the
/// learner writes a newer `latest.safetensors`.
#[allow(clippy::too_many_arguments)]
pub fn run_loop(
    live_dir: &str,
    cuda: bool,
    params: SearchParams,
    concurrency: usize,
    chunk_secs: f64,
    max_chunks: usize,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
    backend: Backend,
) -> anyhow::Result<()> {
    let buffer = Path::new(live_dir).join("buffer");
    std::fs::create_dir_all(&buffer)?;
    // The file the learner rewrites, and thus the one whose mtime means "new weights".
    let weights = backend.weights_path(live_dir);

    while !Path::new(&weights).exists() {
        eprintln!("waiting for the learner's first export at {weights} ...");
        std::thread::sleep(Duration::from_secs(5));
    }
    let device = if backend.is_ort() { None } else { Some(crate::net::pick_device(cuda)?) };
    let make_model = || backend.load(live_dir, device.clone());
    let model: Mutex<Arc<dyn BatchModel>> = Mutex::new(make_model()?);
    let mut last_mtime = std::fs::metadata(&weights).and_then(|m| m.modified()).ok();
    // `weights` is re-resolved per flush (see the on_flush closure).
    let mut seq = max_chunk_seq(&buffer).map(|s| s + 1).unwrap_or(0);
    eprintln!(
        "hex continuous actor [{}]: concurrency {concurrency}, flush {chunk_secs:.0}s, batch \
         {max_batch}, {threads} threads, {} sims/move, buffer {} (keep {max_chunks})",
        backend.label(),
        if params.node_budget > 0 { params.node_budget } else { params.sims },
        buffer.display(),
    );

    let on_flush = |games: Vec<MvGameRecord<Hex>>| {
        let n_games = games.len();
        if n_games == 0 {
            eprintln!("  flush: 0 games (ramp-up / games still in flight)");
        } else {
            let black_wins = games
                .iter()
                .filter(|g| {
                    g.resigned_by.map_or_else(
                        || g.states.last().map(|s| s.winner) == Some(1),
                        |loser| loser == 2,
                    )
                })
                .count();
            let examples = examples_from_records(games);
            let tmp = buffer.join(format!("chunk_{seq:06}.bin.tmp"));
            let path = buffer.join(format!("chunk_{seq:06}.bin"));
            if write_dump(tmp.to_str().unwrap(), &examples).is_ok()
                && std::fs::rename(&tmp, &path).is_ok()
            {
                prune_buffer(&buffer, max_chunks);
                seq += 1;
            }
            eprintln!(
                "  chunk {}: {n_games} games  {} ex  {:.0} plies/game  black won {:.0}%",
                seq.saturating_sub(1),
                examples.len(),
                examples.len() as f64 / n_games as f64,
                100.0 * black_wins as f64 / n_games as f64,
            );
        }
        // The evaluator raises PAUSE so its match isn't fighting us for the GPU.
        // Parking between chunks is coarse but costs nothing: in-flight games
        // simply wait.
        let pause = Path::new(live_dir).join("PAUSE");
        if pause.exists() {
            let parked = Path::new(live_dir).join("actor.paused");
            let _ = std::fs::File::create(&parked);
            eprintln!("  paused for evaluation");
            while pause.exists() {
                std::thread::sleep(Duration::from_secs(1));
            }
            let _ = std::fs::remove_file(&parked);
            eprintln!("  resumed");
        }
        // Re-resolve the path each time: `best.*` may have appeared since.
        let weights = backend.weights_path(live_dir);
        let mt = std::fs::metadata(&weights).and_then(|m| m.modified()).ok();
        if mt != last_mtime {
            match make_model() {
                Ok(nm) => {
                    *model.lock().unwrap() = nm;
                    last_mtime = mt;
                    eprintln!("  hot-reloaded weights");
                }
                Err(e) => eprintln!("  weight reload failed: {e}"),
            }
        }
    };

    batch_search::run_actor_continuous::<Hex, _>(
        &model,
        concurrency,
        params,
        threads,
        max_batch,
        base_seed,
        chunk_secs,
        60.0,
        on_flush,
    );
    Ok(()) // unreachable: runs until killed
}

/// One-chunk throughput benchmark.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    model_path: &str,
    cuda: bool,
    concurrency: usize,
    threads: usize,
    params: SearchParams,
    max_batch: usize,
    seed: u64,
    secs: f64,
    backend: Backend,
) -> anyhow::Result<()> {
    let device = if backend.is_ort() { None } else { Some(crate::net::pick_device(cuda)?) };
    let model = backend.load_file(model_path, device)?;
    let (records, cs) = batch_search::run_selfplay_moves::<Hex>(
        &*model,
        concurrency,
        params,
        threads,
        max_batch,
        seed,
        secs,
        1.0,
    );
    let games = records.len();
    let plies: usize = records.iter().map(|r| r.moves.len()).sum();
    println!(
        "hex bench [{}]: conc {concurrency}  {games} games  {plies} plies  {} nodes  {:.2}s  \
         {:.0} nodes/s  avg_batch {:.0}  gpu_duty {:.2}",
        backend.label(),
        cs.nodes,
        cs.elapsed,
        cs.nodes as f64 / cs.elapsed,
        cs.avg_batch(),
        cs.gpu_duty(),
    );
    // Where the time goes. A node costs one forward *plus* the CPU work around
    // it, so nodes/s sits below the raw positions/s `net-bench` reports; this
    // says which side is actually the limit.
    let cpu = cs.gather_secs + cs.wait_secs + cs.apply_secs;
    if cpu > 0.0 {
        println!(
            "  cpu: descent {:.0}%  wait-for-eval {:.0}%  expand+backup {:.0}%  \
             [{:.1} core-s / {threads} workers, {:.1} wall]",
            100.0 * cs.gather_secs / cpu,
            100.0 * cs.wait_secs / cpu,
            100.0 * cs.apply_secs / cpu,
            cpu,
            cs.elapsed,
        );
    }
    let gt = cs.gpu_secs + cs.gpu_block_secs + cs.gpu_dispatch_secs;
    if gt > 0.0 {
        println!(
            "  gpu thread: forward {:.0}%  starved {:.0}%  dispatch {:.0}%  \
             [{:.1}s forward, {:.1}s wall]",
            100.0 * cs.gpu_secs / gt,
            100.0 * cs.gpu_block_secs / gt,
            100.0 * cs.gpu_dispatch_secs / gt,
            cs.gpu_secs,
            cs.elapsed,
        );
    }
    println!(
        "  evals {} ({:.0}/s)  cache hits {:.0}%  batches {}",
        cs.evals,
        cs.evals as f64 / cs.elapsed,
        100.0 * cs.cache_hit_rate(),
        cs.batches,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use playground_core::batch_search::{MvMoveRecord, SearchParams};
    use playground_core::Game;
    use rand::SeedableRng;

    /// A stand-in net: every position is even, every move equally likely. Enough to
    /// drive the real pipeline without a checkpoint.
    struct Uniform;

    impl BatchModel for Uniform {
        fn forward(&self, _planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            Ok((vec![0.0; b], vec![0.0; b * POLICY_DIM]))
        }
    }

    /// The cold-start failure this whole mechanism exists for: with a flat policy
    /// and an uninformative value head, root Dirichlet noise decides which move
    /// the search commits to — and that noise then *is* the policy target. With
    /// forced playouts + pruning the noise is explored but subtracted back out,
    /// so repeated searches of the same position agree far more often.
    #[test]
    fn pruning_takes_the_exploration_noise_out_of_the_target() {
        let base = SearchParams {
            sims: 0,
            node_budget: 200,
            c_puct: 0.6,
            dir_alpha: 0.3,
            dir_eps: 0.25,
            temp_moves: 0,
            explore_eps: 0.0,
            max_ply: 0,
            settle_plies: 0,
            resign_threshold: 0.0,
            resign_consecutive: 0,
            resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
            full_search_prob: 0.0,
            fast_node_budget: 0,
        };
        // How often do independent searches of the *same* opening agree?
        let agreement = |k: f32| {
            let params = SearchParams { forced_playout_k: k, ..base };
            let (records, _) = batch_search::run_selfplay_moves::<Hex>(
                &FlatPolicyNoisyValue, 32, params, 4, 16, 11, 25.0, 0.0,
            );
            let picks: Vec<usize> = records
                .iter()
                .filter_map(|r| r.moves.first())
                .map(|mv| (0..mv.moves.len()).max_by_key(|&i| mv.visits[i]).unwrap())
                .collect();
            let mut counts = std::collections::HashMap::new();
            for p in &picks {
                *counts.entry(*p).or_insert(0usize) += 1;
            }
            let top = counts.values().copied().max().unwrap_or(0);
            (picks.len(), counts.len(), top as f32 / picks.len().max(1) as f32)
        };
        let (n_off, distinct_off, top_off) = agreement(0.0);
        let (n_on, distinct_on, top_on) = agreement(2.0);
        println!(
            "noise in target:  {n_off} games, {distinct_off} distinct openings, most common {top_off:.2}\n             noise pruned:     {n_on} games, {distinct_on} distinct openings, most common {top_on:.2}"
        );
        assert!(n_off > 8 && n_on > 8, "not enough games to compare");
        assert!(
            top_on >= top_off,
            "pruning made the targets *less* consistent: {top_on:.2} vs {top_off:.2}"
        );
    }

    /// Stand-in for a barely-trained net: a flat policy and a value that varies
    /// with the position but means nothing (what the real value head looks like
    /// for the first few generations).
    struct FlatPolicyNoisyValue;

    impl BatchModel for FlatPolicyNoisyValue {
        fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let mut vals = vec![0.0f32; b];
            for i in 0..b {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for (k, &x) in planes[i * PLANES..(i + 1) * PLANES].iter().enumerate() {
                    if x != 0.0 {
                        h ^= k as u64;
                        h = h.wrapping_mul(0x100_0000_01b3);
                    }
                }
                vals[i] = ((h >> 40) as f32 / 8388608.0 - 1.0) * 0.9; // ±0.9
            }
            Ok((vals, vec![0.5; b * POLICY_DIM]))
        }
    }

    /// The batched engine and the sequential `mcts::run_search` implement the
    /// same PUCT; on the same model and budget they should concentrate their
    /// root visits to the same degree. They did not — the batched search spread
    /// 400 visits over ~53 moves where the sequential one put a third of them on
    /// one move — which is what sent self-play targets to near-uniform and
    /// stalled the policy loss.
    #[test]
    fn both_searches_concentrate_alike_on_the_same_model() {
        let budget = 400u32;
        let params = SearchParams {
            sims: 0,
            node_budget: budget,
            c_puct: 1.5,
            dir_alpha: 0.3,
            dir_eps: 0.0,
            temp_moves: 0,
            explore_eps: 0.0,
            max_ply: 0,
            settle_plies: 0,
            resign_threshold: 0.0,
            resign_consecutive: 0,
            resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
            full_search_prob: 0.0,
            fast_node_budget: 0,
        };
        let (records, _) =
            batch_search::run_selfplay_moves::<Hex>(&FlatPolicyNoisyValue, 2, params, 2, 2, 5, 30.0, 0.0);
        assert!(!records.is_empty(), "no game finished in 30s");
        let opening = &records[0].moves[0];
        let total: u32 = opening.visits.iter().sum();
        let batched_top = *opening.visits.iter().max().unwrap() as f32 / total as f32;
        let batched_spread = opening.visits.iter().filter(|&&v| v > 0).count();

        struct Adapter;
        impl crate::mcts::Evaluator for Adapter {
            fn evaluate(&self, state: &crate::game::HexState) -> anyhow::Result<(f32, Vec<f32>)> {
                let mut planes = vec![0f32; PLANES];
                fill_planes_into(state, &mut planes);
                let (v, p) = FlatPolicyNoisyValue.forward(&planes, 1)?;
                Ok((v[0], p))
            }
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let seq_params = crate::mcts::Params {
            sims: budget,
            c_puct: 1.5,
            dir_alpha: 0.3,
            dir_eps: 0.0,
            temp_moves: 0,
        };
        let (_m, visits, _v) =
            crate::mcts::run_search(&Adapter, &Hex::initial_state(), &seq_params, &mut rng).unwrap();
        let st: u32 = visits.iter().sum();
        let seq_top = *visits.iter().max().unwrap() as f32 / st as f32;
        let seq_spread = visits.iter().filter(|&&v| v > 0).count();

        println!(
            "batched: top {batched_top:.3} over {batched_spread} moves ({total} visits)\n             sequential: top {seq_top:.3} over {seq_spread} moves ({st} visits)"
        );
        assert!(
            batched_top > seq_top / 2.0,
            "batched search is far flatter: top {batched_top:.3} vs {seq_top:.3}"
        );
    }

    /// A model whose policy is a *function of the planes it is handed*: all the
    /// mass on the first empty cell it can see. If the engine ever pairs a
    /// position with another position's row of the batch, the spike lands on a
    /// cell that is occupied here, `expand` masks it away, and the search falls
    /// back to a flat distribution — so a misalignment shows up as visits that
    /// don't concentrate where the model asked.
    struct SpikeFirstEmpty;

    impl BatchModel for SpikeFirstEmpty {
        fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let mut policy = vec![0.0f32; b * POLICY_DIM];
            for i in 0..b {
                let row = &planes[i * PLANES..(i + 1) * PLANES];
                let first_empty = (0..POLICY_DIM)
                    .find(|&c| row[c] == 0.0 && row[POLICY_DIM + c] == 0.0)
                    .unwrap_or(0);
                policy[i * POLICY_DIM + first_empty] = 20.0; // ≫ the 0.0 elsewhere
            }
            Ok((vec![0.0; b], policy))
        }
    }

    /// Every game in a batch must be scored with *its own* row of the forward.
    #[test]
    fn batched_evaluation_pairs_each_game_with_its_own_policy_row() {
        let params = SearchParams {
            sims: 0,
            node_budget: 12,
            c_puct: 1.5,
            dir_alpha: 0.3,
            dir_eps: 0.0,
            temp_moves: 0,
            explore_eps: 0.0,
            max_ply: 0,
            settle_plies: 0,
            resign_threshold: 0.0,
            resign_consecutive: 0,
            resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
            full_search_prob: 0.0,
            fast_node_budget: 0,
        };
        // Many games in flight against a small batch, so rows are mixed.
        let (records, _stats) =
            batch_search::run_selfplay_moves::<Hex>(&SpikeFirstEmpty, 24, params, 4, 8, 7, 25.0, 0.0);
        assert!(!records.is_empty(), "no game finished in 25s");
        let (mut checked, mut agreed) = (0usize, 0usize);
        for rec in &records {
            for (i, mv) in rec.moves.iter().enumerate() {
                let state = &rec.states[i];
                // What the model would have asked for, given this position.
                let want = state.empty().iter().map(|c| canon_index(state, c)).min().unwrap();
                let best = (0..mv.moves.len()).max_by_key(|&k| mv.visits[k]).unwrap();
                let got = match mv.moves[best] {
                    HexMove::Play { r, c } => canon_index(state, r as usize * N + c as usize),
                    HexMove::Swap => usize::MAX,
                };
                checked += 1;
                agreed += usize::from(got == want);
            }
        }
        // A spike 20 logits above everything else wins every search that sees it.
        assert!(
            agreed * 100 >= checked * 99,
            "only {agreed}/{checked} moves followed the model's own policy row — \
             batch rows are being paired with the wrong games"
        );
    }

    /// The value target of every example is `winner == states[i].to_move`, so a
    /// record whose `states` and `moves` disagree about whose turn it is would
    /// train the net on inverted outcomes — invisible in the loss, fatal to the
    /// run. Check the alignment the engine actually produces.
    #[test]
    fn selfplay_records_line_up_states_with_movers() {
        let params = SearchParams {
            sims: 0,
            node_budget: 4,
            c_puct: 1.5,
            dir_alpha: 0.3,
            dir_eps: 0.0,
            temp_moves: 0,
            explore_eps: 0.0,
            max_ply: 0,
            settle_plies: 0,
            resign_threshold: 0.0,
            resign_consecutive: 0,
            resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
            full_search_prob: 0.0,
            fast_node_budget: 0,
        };
        let (records, _stats) =
            batch_search::run_selfplay_moves::<Hex>(&Uniform, 8, params, 2, 8, 1, 20.0, 0.0);
        assert!(!records.is_empty(), "no game finished in 20s");
        for rec in &records {
            assert_eq!(rec.states.len(), rec.moves.len() + 1, "one state per move, plus the last");
            for (i, mv) in rec.moves.iter().enumerate() {
                assert_eq!(
                    rec.states[i].to_move, mv.to_move,
                    "record {i}: the position and the move record disagree on the mover"
                );
                assert_eq!(rec.states[i].winner, 0, "a decided position was still searched");
                // The move actually played must be one the search considered.
                assert!(mv.moves.contains(&mv.moves[mv.played]));
            }
            let last = rec.states.last().unwrap();
            assert_ne!(last.winner, 0, "a finished game ends decided (Hex has no draws)");
            assert_eq!(last.winner, 3 - last.to_move, "the winner is whoever moved last");
        }
        // And the derived examples inherit that: the winner's positions score +1.
        let winners: Vec<u8> = records.iter().map(|r| r.states.last().unwrap().winner).collect();
        let plies: Vec<usize> = records.iter().map(|r| r.moves.len()).collect();
        let examples = examples_from_records(records);
        assert_eq!(examples.len(), plies.iter().sum::<usize>());
        let mut k = 0;
        for (g, n) in plies.iter().enumerate() {
            for i in 0..*n {
                // Mover at ply i is black when i is even; z is +1 iff that side won.
                let mover = if i % 2 == 0 { 1 } else { 2 };
                let want = if winners[g] == mover { 1.0 } else { -1.0 };
                assert_eq!(examples[k].value, want, "game {g} ply {i}");
                k += 1;
            }
        }
    }

    #[test]
    fn examples_carry_canonical_targets_and_alternating_outcomes() {
        // Black plays a1, White b1, then Black is credited with the win.
        let s0 = Hex::initial_state();
        let s1 = Hex::apply(&s0, &HexMove::Play { r: 0, c: 0 });
        let mut s2 = Hex::apply(&s1, &HexMove::Play { r: 0, c: 1 });
        s2.winner = 1;
        let rec = MvGameRecord::<Hex> {
            states: vec![s0, s1, s2],
            moves: vec![
                MvMoveRecord::<Hex> {
                    moves: vec![HexMove::Play { r: 0, c: 0 }],
                    visits: vec![10],
                    played: 0,
                    to_move: 1,
                    for_training: true,
                },
                MvMoveRecord::<Hex> {
                    moves: vec![HexMove::Play { r: 0, c: 1 }],
                    visits: vec![10],
                    played: 0,
                    to_move: 2,
                    for_training: true,
                },
            ],
            resigned_by: None,
        };
        let ex = examples_from_records(vec![rec]);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].value, 1.0, "black moved and black won");
        assert_eq!(ex[1].value, -1.0, "white moved and lost");
        // Black's target sits at the played cell; White's is transposed into the
        // canonical frame — b1 = (0,1) becomes (1,0).
        assert_eq!(ex[0].policy[0], 1.0);
        assert_eq!(ex[1].policy[N], 1.0);
        for e in &ex {
            assert!((e.policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert_eq!(e.planes.len(), PLANES);
            assert_eq!(e.ownership.len(), POLICY_DIM);
        }
        // Black owns a1 at the end and moved first, so from black's view it is +1
        // and from white's (transposed) view it is -1.
        assert_eq!(ex[0].ownership[0], 1.0);
        assert_eq!(ex[1].ownership[crate::game::transpose_index(0)], -1.0);
        // b1 is white's stone: -1 for black, +1 for white in its own frame.
        assert_eq!(ex[0].ownership[1], -1.0);
        assert_eq!(ex[1].ownership[crate::game::transpose_index(1)], 1.0);
    }

    #[test]
    fn dump_roundtrips_through_the_learner_format() {
        let dir = std::env::temp_dir().join(format!("hex-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chunk.bin");
        let ex = vec![Example {
            planes: vec![0.5; PLANES],
            value: -1.0,
            policy: (0..POLICY_DIM).map(|i| i as f32).collect(),
            ownership: vec![-1.0; POLICY_DIM],
        }];
        write_dump(path.to_str().unwrap(), &ex).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), DUMP_MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), DUMP_VERSION);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        assert_eq!(bytes.len(), 16 + EXAMPLE_FLOATS * 4);
        let f = |off: usize| f32::from_le_bytes(bytes[16 + off * 4..20 + off * 4].try_into().unwrap());
        assert_eq!(f(0), 0.5, "planes");
        assert_eq!(f(PLANES), -1.0, "value");
        assert_eq!(f(PLANES + 2), 1.0, "policy[1]");
        assert_eq!(f(PLANES + 1 + POLICY_DIM), -1.0, "ownership[0]");
        std::fs::remove_dir_all(&dir).ok();
    }
}
