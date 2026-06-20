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
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use playground_core::latent_mcts::{search, RootChild, SearchConfig};
use playground_core::{Game, Outcome};

use crate::game::{Connect4, Connect4State, C4_COLS};
use crate::lgs_net::LgsModel;
use crate::mcts::{planes_of, PLANES};

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
async fn play_game(
    model: &LgsModel,
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
        let (recs, reif) = play_game(&model, &cfg, temp_moves, &mut rng).await?;
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
