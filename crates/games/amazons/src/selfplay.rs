//! Continuous self-play actor + throughput bench for Amazons, mirroring
//! `go9/selfplay.rs` but on the move-indexed engine (`run_chunk_moves`) with the
//! factored two-head policy.
//!
//! Each turn yields one training example: the position planes, the game-outcome z
//! (±1, side-to-move POV — Amazons has no draws), and the MCTS visit distribution
//! over moves projected onto the two heads' **marginals**
//! (`q[from,qmt] += visit_prob`, `a[to,amt] += visit_prob`). Targets are stored
//! **sparse** — dense 2×7200 would be ~57 KB/example.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use playground_core::batch_search::{self, BatchModel, MvGameRecord, SearchParams};

use crate::game::{self, Amazons, PLANES as PLANE_COUNT};
use crate::mcts::PLANES;
use crate::moves::move_to_indices;
use crate::net::LocalModel;

/// One self-play training example (sparse two-head policy target).
pub struct Example {
    pub planes: Vec<f32>, // PLANES floats
    pub value: f32,       // game outcome z (side-to-move POV)
    pub queen: Vec<(u32, f32)>,
    pub arrow: Vec<(u32, f32)>,
}

fn sparse(map: HashMap<u32, f32>) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = map.into_iter().collect();
    v.sort_unstable_by_key(|&(i, _)| i);
    v
}

/// Turn finished games into examples: value = ±1 from each position's side-to-move
/// POV (resignation loss, else the natural no-move-loses terminal; capped = z 0),
/// policy = visit distribution projected to the queen/arrow head marginals.
pub fn examples_from_records(records: Vec<MvGameRecord<Amazons>>) -> Vec<Example> {
    let mut out = Vec::new();
    for rec in records {
        let final_state = rec.states.last().unwrap();
        let winner: Option<u8> = if let Some(loser) = rec.resigned_by {
            Some(3 - loser)
        } else if !game::has_any_move(final_state) {
            Some(3 - final_state.to_move) // side to move can't move → loses
        } else {
            None // hit the ply backstop, undecided
        };
        for (i, mv) in rec.moves.iter().enumerate() {
            let total: u32 = mv.visits.iter().sum();
            if total == 0 {
                continue;
            }
            let mut q: HashMap<u32, f32> = HashMap::new();
            let mut a: HashMap<u32, f32> = HashMap::new();
            for (m, &v) in mv.moves.iter().zip(&mv.visits) {
                if v == 0 {
                    continue;
                }
                let prob = v as f32 / total as f32;
                let (qi, ai) = move_to_indices(m);
                *q.entry(qi as u32).or_insert(0.0) += prob;
                *a.entry(ai as u32).or_insert(0.0) += prob;
            }
            let z = match winner {
                Some(w) => if w == mv.to_move { 1.0 } else { -1.0 },
                None => 0.0,
            };
            let mut planes = vec![0f32; PLANES];
            game::fill_planes_into(&rec.states[i], &mut planes);
            out.push(Example { planes, value: z, queen: sparse(q), arrow: sparse(a) });
        }
    }
    out
}

/// Sparse binary dump read by `amazons_az_learner.py`. Header: count u64. Per
/// example: planes[PLANES f32], value f32, nq u32 then nq×(idx u32, prob f32),
/// na u32 then na×(idx u32, prob f32). All little-endian.
pub fn write_dump(out_path: &str, examples: &[Example]) -> anyhow::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in examples {
        for &x in &ex.planes {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&ex.value.to_le_bytes());
        for head in [&ex.queen, &ex.arrow] {
            buf.extend_from_slice(&(head.len() as u32).to_le_bytes());
            for &(idx, prob) in head {
                buf.extend_from_slice(&idx.to_le_bytes());
                buf.extend_from_slice(&prob.to_le_bytes());
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

fn max_chunk_seq(buffer: &Path) -> Option<u64> {
    std::fs::read_dir(buffer)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let s = name.to_str()?;
            s.strip_prefix("chunk_")?.strip_suffix(".bin")?.parse::<u64>().ok()
        })
        .max()
}

fn prune_buffer(buffer: &Path, keep: usize) {
    let mut chunks: Vec<(u64, PathBuf)> = std::fs::read_dir(buffer)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let s = e.file_name().to_str()?.strip_prefix("chunk_")?.strip_suffix(".bin")?
                .parse::<u64>().ok()?;
            Some((s, e.path()))
        })
        .collect();
    chunks.sort_unstable_by_key(|&(s, _)| s);
    if chunks.len() > keep {
        for (_, p) in &chunks[..chunks.len() - keep] {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Continuous actor: forever build a chunk of games, append to `<live>/buffer`,
/// prune. `backend` = "candle" (loads latest.safetensors) or "ort" (loads model.onnx).
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
    backend: &str,
    ep: &str,
    fp16: bool,
) -> anyhow::Result<()> {
    let latest = format!("{live_dir}/latest.safetensors");
    let onnx = format!("{live_dir}/model.onnx");
    let trt_cache = format!("{live_dir}/trt_cache");
    let buffer = Path::new(live_dir).join("buffer");
    std::fs::create_dir_all(&buffer)?;
    let ort = backend == "ort";
    let device = if ort { None } else { Some(crate::net::pick_device(cuda)?) };
    // The weights file we hot-reload from when the learner snapshots.
    let weights = if ort { onnx.clone() } else { latest.clone() };

    // Build the current model (ORT/TRT or candle) as a trait object.
    let make_model = || -> anyhow::Result<Arc<dyn BatchModel>> {
        if ort {
            Ok(Arc::new(crate::ort_net::OrtModel::new(&onnx, ep, fp16, &trt_cache)?) as Arc<dyn BatchModel>)
        } else {
            Ok(Arc::new(LocalModel::load_on(&latest, device.clone().unwrap())?) as Arc<dyn BatchModel>)
        }
    };

    while !Path::new(&weights).exists() {
        std::thread::sleep(Duration::from_secs(1)); // wait for the learner's first export
    }
    let model: Mutex<Arc<dyn BatchModel>> = Mutex::new(make_model()?);
    let mut last_mtime = std::fs::metadata(&weights).and_then(|m| m.modified()).ok();
    let mut seq = max_chunk_seq(&buffer).map(|s| s + 1).unwrap_or(0);
    eprintln!(
        "amazons continuous actor: backend {backend}{}, concurrency {concurrency}, \
         flush {chunk_secs:.0}s, batch {max_batch}, {threads} threads, buffer {} (keep {max_chunks})",
        if ort { format!("/{ep} fp16={fp16}") } else { String::new() },
        buffer.display(),
    );

    // Flush completed games to a buffer chunk every `chunk_secs`; in-flight games keep
    // running. Hot-reload weights when the learner writes a newer snapshot.
    let on_flush = |games: Vec<MvGameRecord<Amazons>>| {
        let n_games = games.len();
        if n_games == 0 {
            eprintln!("  flush: 0 games (ramp-up / long games still in flight)");
        } else {
            let resigned = games.iter().filter(|g| g.resigned_by.is_some()).count();
            let examples = examples_from_records(games);
            let tmp = buffer.join(format!("chunk_{seq:06}.bin.tmp"));
            let path = buffer.join(format!("chunk_{seq:06}.bin"));
            if write_dump(tmp.to_str().unwrap(), &examples).is_ok() && std::fs::rename(&tmp, &path).is_ok() {
                prune_buffer(&buffer, max_chunks);
                seq += 1;
            }
            eprintln!(
                "  chunk {}: {n_games} games  {} ex  {:.0} plies/game  {:.0}% resigned",
                seq.saturating_sub(1), examples.len(),
                examples.len() as f64 / n_games as f64, 100.0 * resigned as f64 / n_games as f64,
            );
        }
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

    batch_search::run_actor_continuous::<Amazons, _>(
        &model, concurrency, params, threads, max_batch, base_seed, chunk_secs, 60.0, on_flush,
    );
    Ok(()) // unreachable (runs until the process is killed)
}

/// One-chunk throughput benchmark (candle or ort).
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
    backend: &str,
    onnx: &str,
    ep: &str,
    fp16: bool,
) -> anyhow::Result<()> {
    let (records, cs) = if backend == "ort" {
        let model = crate::ort_net::OrtModel::new(onnx, ep, fp16, "data/amazons_trt_cache")?;
        batch_search::run_selfplay_moves::<Amazons>(&model, concurrency, params, threads, max_batch, seed, secs, 1.0)
    } else {
        let device = crate::net::pick_device(cuda)?;
        let model = LocalModel::load_on(model_path, device)?;
        batch_search::run_selfplay_moves::<Amazons>(&model, concurrency, params, threads, max_batch, seed, secs, 1.0)
    };
    let games = records.len();
    let plies: usize = records.iter().map(|r| r.moves.len()).sum();
    let nps = cs.nodes as f64 / cs.elapsed;
    println!(
        "amazons bench [{backend}{}]: conc {concurrency}  {games} games done  {plies} plies  {} nodes  {:.2}s  \
         {nps:.0} nodes/s  avg_batch {:.0}  gpu_duty {:.2}",
        if backend == "ort" { format!("/{ep}/fp16={fp16}") } else { String::new() },
        cs.nodes, cs.elapsed, cs.avg_batch(), cs.gpu_duty(),
    );
    // CPU worker time (summed over workers): descent (tree walk) + legal-gen
    // (legal_moves bitboard enumeration) + priors (combine + backup).
    let cpu = cs.gather_secs + cs.wait_secs + cs.apply_secs;
    println!(
        "  cpu: descent {:.0}%  legal-gen {:.0}%  priors {:.0}%  [{:.1} core-s / {} workers, {:.1} wall]  gpu {:.1}s",
        100.0 * cs.gather_secs / cpu, 100.0 * cs.wait_secs / cpu, 100.0 * cs.apply_secs / cpu,
        cpu, threads, cs.elapsed, cs.gpu_secs,
    );
    let _ = PLANE_COUNT;
    Ok(())
}
