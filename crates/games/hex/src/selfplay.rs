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

/// One self-play training example.
pub struct Example {
    pub planes: Vec<f32>, // PLANES floats, canonical frame
    pub value: f32,       // game outcome z, side-to-move POV
    pub policy: Vec<f32>, // POLICY_DIM visit probabilities, canonical frame
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
            out.push(Example { planes, value: z, policy });
        }
    }
    out
}

/// Binary dump read by `hex_az_learner.py`. Header: count u64. Per example:
/// planes[PLANES f32], value f32, policy[POLICY_DIM f32]. All little-endian.
pub fn write_dump(out_path: &str, examples: &[Example]) -> anyhow::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(8 + examples.len() * (PLANES + 1 + POLICY_DIM) * 4);
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in examples {
        for &x in &ex.planes {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&ex.value.to_le_bytes());
        for &p in &ex.policy {
            buf.extend_from_slice(&p.to_le_bytes());
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
) -> anyhow::Result<()> {
    let latest = format!("{live_dir}/latest.safetensors");
    let buffer = Path::new(live_dir).join("buffer");
    std::fs::create_dir_all(&buffer)?;
    let device = crate::net::pick_device(cuda)?;

    while !Path::new(&latest).exists() {
        eprintln!("waiting for the learner's first export at {latest} ...");
        std::thread::sleep(Duration::from_secs(5));
    }
    let make_model = || -> anyhow::Result<Arc<dyn BatchModel>> {
        Ok(Arc::new(LocalModel::load_on(&latest, device.clone())?) as Arc<dyn BatchModel>)
    };
    let model: Mutex<Arc<dyn BatchModel>> = Mutex::new(make_model()?);
    let mut last_mtime = std::fs::metadata(&latest).and_then(|m| m.modified()).ok();
    let mut seq = max_chunk_seq(&buffer).map(|s| s + 1).unwrap_or(0);
    eprintln!(
        "hex continuous actor: concurrency {concurrency}, flush {chunk_secs:.0}s, batch \
         {max_batch}, {threads} threads, {} sims/move, buffer {} (keep {max_chunks})",
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
        let mt = std::fs::metadata(&latest).and_then(|m| m.modified()).ok();
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
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let model = LocalModel::load_on(model_path, device)?;
    let (records, cs) = batch_search::run_selfplay_moves::<Hex>(
        &model,
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
        "hex bench: conc {concurrency}  {games} games  {plies} plies  {} nodes  {:.2}s  \
         {:.0} nodes/s  avg_batch {:.0}  gpu_duty {:.2}",
        cs.nodes,
        cs.elapsed,
        cs.nodes as f64 / cs.elapsed,
        cs.avg_batch(),
        cs.gpu_duty(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use playground_core::batch_search::MvMoveRecord;
    use playground_core::Game;

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
                },
                MvMoveRecord::<Hex> {
                    moves: vec![HexMove::Play { r: 0, c: 1 }],
                    visits: vec![10],
                    played: 0,
                    to_move: 2,
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
        }
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
        }];
        write_dump(path.to_str().unwrap(), &ex).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 1);
        assert_eq!(bytes.len(), 8 + (PLANES + 1 + POLICY_DIM) * 4);
        let f = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        assert_eq!(f(8), 0.5);
        assert_eq!(f(8 + PLANES * 4), -1.0);
        assert_eq!(f(8 + (PLANES + 1) * 4 + 4), 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
