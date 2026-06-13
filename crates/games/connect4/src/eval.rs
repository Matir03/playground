//! Absolute-strength evaluation: play a checkpoint's MCTS against the perfect
//! Connect 4 solver and report the net's score. Unlike chained self-play ELO,
//! this is grounded — it measures genuine distance to perfect play. Each game
//! uses a short random opening (for variety) and alternates which side the net
//! takes; the net moves by `run_mcts` (argmax, no exploration noise), the
//! opponent by `Connect4::choose_move` (the perfect alpha-beta solver + book).
//!
//! Games run in parallel on native threads: the net's leaf evaluations are
//! coalesced into batched GPU forwards by the shared inference server (the same
//! one self-play uses), while the solver runs on the CPU. The evaluator is meant
//! to run with self-play/training PAUSED (see the actor's `pause_gate`), so it
//! gets the whole GPU and these batches stay full.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;

use playground_core::{Game, Outcome, Solver};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::game::{Connect4, Connect4Move, C4_COLS};
use crate::mcts::{run_mcts, Params};
use crate::net::LocalModel;
use crate::selfplay::{run_inference_server, BatchedEval, Req};

fn argmax(counts: &[u32; C4_COLS]) -> usize {
    let mut best = 0;
    for c in 1..C4_COLS {
        if counts[c] > counts[best] {
            best = c;
        }
    }
    best
}

/// Play one game of the net (argmax MCTS) vs the perfect solver and return the
/// net's score doubled (win=2, draw=1, loss=0) so it sums exactly as an integer.
fn play_game(
    eval: &BatchedEval,
    params: &Params,
    opening: u32,
    net_first: bool,
    seed: u64,
) -> anyhow::Result<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = Connect4::initial_state();
    for _ in 0..opening {
        if Connect4::outcome(&state).is_terminal() {
            break;
        }
        let legal = Connect4::legal_moves(&state);
        state = Connect4::apply(&state, &legal[rng.gen_range(0..legal.len())]);
    }
    while !Connect4::outcome(&state).is_terminal() {
        let net_turn = (state.ply % 2 == 0) == net_first;
        let mv = if net_turn {
            let (counts, _v, _n, _s) = run_mcts(eval, &state, params, &mut rng)?;
            Connect4Move { col: argmax(&counts) as u8 }
        } else {
            Connect4::choose_move(&state).best_move
        };
        state = Connect4::apply(&state, &mv);
    }
    let net_player = if net_first { 1u8 } else { 2 };
    Ok(match Connect4::outcome(&state) {
        Outcome::FirstWins => if net_player == 1 { 2 } else { 0 },
        Outcome::SecondWins => if net_player == 2 { 2 } else { 0 },
        _ => 1,
    })
}

/// Play `games` games of the net vs the perfect solver across `threads` threads
/// (leaf evals batched by the shared inference server); print and return the
/// net's score in [0,1] (win=1, draw=0.5, loss=0), averaged over games.
#[allow(clippy::too_many_arguments)]
pub fn run_eval_vs_solver(
    checkpoint: &str,
    cuda: bool,
    games: u32,
    params: Params,
    opening: u32,
    seed: u64,
    threads: usize,
    max_batch: usize,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LocalModel::load_on(checkpoint, device)?);
    eprintln!(
        "c4 eval vs solver: {games} games, {threads} threads, node_budget {}, opening {opening}, \
         device {}, model {checkpoint}",
        params.node_budget,
        if cuda { "cuda" } else { "cpu" },
    );

    let next = AtomicU32::new(0);
    let score2 = AtomicU64::new(0); // sum of doubled per-game scores (exact)
    let (req_tx, req_rx) = channel::<Req>();
    std::thread::scope(|scope| {
        let model_ref: &LocalModel = &model;
        scope.spawn(move || run_inference_server(req_rx, model_ref, max_batch));
        for _ in 0..threads {
            let eval = BatchedEval { req_tx: req_tx.clone() };
            let next = &next;
            let score2 = &score2;
            let params = &params;
            scope.spawn(move || loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= games {
                    break;
                }
                let net_first = g % 2 == 0; // alternate which side the net takes
                let s2 = play_game(&eval, params, opening, net_first, seed.wrapping_add(g as u64))
                    .expect("eval game failed");
                score2.fetch_add(s2, Ordering::Relaxed);
            });
        }
        drop(req_tx); // drop the spare so the server shuts down once games finish
    });

    let net_score = score2.load(Ordering::Relaxed) as f64 / 2.0 / games as f64;
    println!(
        "net_vs_solver {net_score:.4}  ({games} games, node_budget {}, opening {opening})",
        params.node_budget
    );
    Ok(net_score)
}
