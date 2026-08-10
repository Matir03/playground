//! Head-to-head match between two checkpoints for ELO tracking, on the SAME async
//! leaf-batching pipeline as self-play (`batch_search::run_eval_moves`) — self-play
//! is the special case A == B. Plays `games` games to terminal (alternating which
//! colour A plays, sampled opening via `temp_moves`, no resignation) and reports A's
//! score. Both nets run in-process via candle (`LocalModel`), batched and routed per
//! side-to-move by the engine. The evaluator should pause the actor/learner first to
//! avoid GPU contention.

use std::time::Instant;

use playground_core::batch_search::{self, SearchParams};

use crate::game::Amazons;
use crate::mcts::Params;
use crate::net::LocalModel;

/// Run a `games`-game match A vs B; print and return A's score in [0,1].
#[allow(clippy::too_many_arguments)]
pub fn run_match(
    ckpt_a: &str,
    ckpt_b: &str,
    cuda: bool,
    games: usize,
    params: Params,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let a = LocalModel::load_on(ckpt_a, device.clone())?;
    let b = LocalModel::load_on(ckpt_b, device)?;
    eprintln!(
        "amazons eval-match: {games} games, {threads} threads, node-budget {}, max-ply {}\n  A={ckpt_a}\n  B={ckpt_b}",
        params.node_budget, params.max_ply,
    );
    // Eval search knobs: no resignation (play to terminal for a clean result), keep a
    // small opening-sampling window for game diversity.
    let sp = SearchParams {
        sims: 0,
        node_budget: params.node_budget,
        c_puct: params.c_puct,
        dir_alpha: params.dir_alpha,
        dir_eps: params.dir_eps,
        temp_moves: params.temp_moves,
        explore_eps: 0.0,
        max_ply: params.max_ply,
        settle_plies: 0,
        resign_threshold: 0.0,
        resign_consecutive: 0,
        resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
    };
    let t0 = Instant::now();
    let a_score =
        batch_search::run_eval_moves::<Amazons>(&a, &b, games, sp, threads, max_batch, base_seed);
    println!("A_score {a_score:.4}  ({games} games, {:.1}s)", t0.elapsed().as_secs_f64());
    Ok(a_score)
}
