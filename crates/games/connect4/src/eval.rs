//! Absolute-strength evaluation: play a checkpoint's MCTS against the perfect
//! Connect 4 solver and report the net's score. Unlike chained self-play ELO,
//! this is grounded — it measures genuine distance to perfect play. Each game
//! uses a short random opening (for variety) and alternates which side the net
//! takes; the net moves by `run_mcts` (argmax, no exploration noise), the
//! opponent by `Connect4::choose_move` (the perfect alpha-beta solver + book).

use playground_core::{Game, Outcome, Solver};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS};
use crate::mcts::{run_mcts, Params};
use crate::net::LocalModel;

fn argmax(counts: &[u32; C4_COLS]) -> usize {
    let mut best = 0;
    for c in 1..C4_COLS {
        if counts[c] > counts[best] {
            best = c;
        }
    }
    best
}

/// Play `games` games of the net vs the perfect solver; print and return the
/// net's score in [0,1] (win=1, draw=0.5, loss=0), averaged over games.
pub fn run_eval_vs_solver(
    checkpoint: &str,
    cuda: bool,
    games: u32,
    params: Params,
    opening: u32,
    seed: u64,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let model = LocalModel::load_on(checkpoint, device)?;
    let eval = |s: &Connect4State| model.policy_value(s);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    eprintln!(
        "c4 eval vs solver: {games} games, node_budget {}, opening {opening}, device {}, model {checkpoint}",
        params.node_budget,
        if cuda { "cuda" } else { "cpu" },
    );

    let mut score = 0.0f64;
    for g in 0..games {
        let net_first = g % 2 == 0; // net is the first player (player 1) on even games
        let mut state = Connect4::initial_state();
        // Short random opening for variety (same idea as the Python arena).
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
                let (counts, _v, _n, _s) = run_mcts(&eval, &state, &params, &mut rng)?;
                Connect4Move { col: argmax(&counts) as u8 }
            } else {
                Connect4::choose_move(&state).best_move
            };
            state = Connect4::apply(&state, &mv);
        }
        let net_player = if net_first { 1u8 } else { 2 };
        let s = match Connect4::outcome(&state) {
            Outcome::FirstWins => if net_player == 1 { 1.0 } else { 0.0 },
            Outcome::SecondWins => if net_player == 2 { 1.0 } else { 0.0 },
            _ => 0.5,
        };
        score += s;
    }

    let net_score = score / games as f64;
    println!("net_vs_solver {net_score:.4}  ({games} games, node_budget {}, opening {opening})",
             params.node_budget);
    Ok(net_score)
}
