//! Two ways to rate a Hex checkpoint.
//!
//!   * [`run_match`] — **relative**: checkpoint A vs checkpoint B on the same
//!     batched pipeline self-play uses (`run_eval_moves`, i.e. self-play with
//!     A ≠ B), for ELO tracking across generations.
//!   * [`run_vs_mohex`] — **absolute**: the net, with search, against MoHex over
//!     GTP. This is the number that says whether the run is going anywhere; a
//!     gauntlet can drift upward against itself forever.

use std::time::Instant;

use playground_core::batch_search::{self, SearchParams};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::{Hex, HexState};
use crate::mcts::{self, Params};
use crate::mohex::Mohex;
use crate::net::LocalModel;

/// Play `games` games of A vs B (alternating colours) and return A's score in
/// [0,1]. Both nets run in-process via candle, batched and routed by side to move.
#[allow(clippy::too_many_arguments)]
pub fn run_match(
    ckpt_a: &str,
    ckpt_b: &str,
    cuda: bool,
    games: usize,
    node_budget: u32,
    c_puct: f32,
    temp_moves: u32,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
    backend: crate::selfplay::Backend,
) -> anyhow::Result<f64> {
    // Matches are a big slice of a training cycle — a 40-game gate on candle
    // took 40 minutes, because candle is latency-bound at these batch sizes
    // (~1.3k nodes/s against TensorRT's ~100k). Both sides load through the
    // same backend so neither gets an advantage from it.
    let device = if backend.is_ort() { None } else { Some(crate::net::pick_device(cuda)?) };
    let a = backend.load_file(ckpt_a, device.clone())?;
    let b = backend.load_file(ckpt_b, device)?;
    eprintln!(
        "hex eval-match [{}]: {games} games, {threads} threads, node-budget {node_budget}\n  \
         A={ckpt_a}\n  B={ckpt_b}",
        backend.label(),
    );
    let sp = SearchParams {
        sims: 0,
        node_budget,
        c_puct,
        dir_alpha: 0.3,
        dir_eps: 0.0, // no exploration noise when measuring strength
        temp_moves,   // a short sampled opening keeps the games from being identical
        explore_eps: 0.0,
        // Hex always terminates within 169 plies; the cap is a pure backstop.
        max_ply: 200,
        settle_plies: 0,
        resign_threshold: 0.0,
        resign_consecutive: 0,
        resign_playout_frac: 0.0,
            forced_playout_k: 0.0,
            full_search_prob: 0.0,
            fast_node_budget: 0,
    };
    let t0 = Instant::now();
    let a_score =
        batch_search::run_eval_moves::<Hex>(&*a, &*b, games, sp, threads, max_batch, base_seed);
    println!("A_score {a_score:.4}  ({games} games, {:.1}s)", t0.elapsed().as_secs_f64());
    Ok(a_score)
}

/// Play `games` games of the net (PUCT search over its own policy/value) against
/// MoHex, alternating colours. Returns the net's score in [0,1].
#[allow(clippy::too_many_arguments)]
/// One game against MoHex. Returns `(net_won, plies, mohex_resigned)`; an `Err`
/// means MoHex stopped talking (it crashes on some positions) and the game has
/// no result.
fn play_one(
    model: &LocalModel,
    params: &Params,
    mohex: &mut Mohex,
    g: u32,
    seed: u64,
) -> anyhow::Result<(bool, u16, bool)> {
    mohex.clear_board()?;
    let net_black = g % 2 == 0;
    let mut state = HexState::default();
    let mut last_synced: Option<HexState> = None;
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(g as u64));
    let mut resigned = false;

    while state.winner == 0 {
        let mv = if (state.to_move == 1) == net_black {
            let (moves, visits, _v) = mcts::run_search(model, &state, params, &mut rng)?;
            let mv = mcts::pick_move(&moves, &visits, state.ply as u32, params.temp_moves, &mut rng);
            mohex.set_position(&state, last_synced.as_ref())?;
            mohex.play(state.to_move, &mv)?;
            mv
        } else {
            mohex.set_position(&state, last_synced.as_ref())?;
            match mohex.genmove(state.to_move)? {
                Some(mv) => mv,
                None => {
                    resigned = true; // MoHex resigns only when it has proven the loss
                    break;
                }
            }
        };
        state = <Hex as playground_core::Game>::apply(&state, &mv);
        last_synced = Some(state);
    }
    let net_won = resigned || (state.winner == 1) == net_black;
    Ok((net_won, state.ply, resigned))
}

pub fn run_vs_mohex(
    ckpt: &str,
    cuda: bool,
    mohex_bin: &str,
    games: u32,
    sims: u32,
    c_puct: f32,
    mohex_playouts: u32,
    mohex_time: f64,
    seed: u64,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let model = LocalModel::load_on(ckpt, device)?;
    let params = Params { sims, c_puct, dir_alpha: 0.3, dir_eps: 0.0, temp_moves: 2 };
    let mut mohex = Mohex::new(mohex_bin, mohex_playouts, mohex_time)?;
    eprintln!(
        "hex net vs mohex: {games} games, {sims} sims/move, mohex {mohex_playouts} playouts\
         {}\n  net={ckpt}",
        if mohex_time > 0.0 { format!(" / {mohex_time}s") } else { String::new() },
    );

    let mut score = 0.0f64;
    let mut completed = 0u32;
    let mut voided = 0u32;
    for g in 0..games {
        // MoHex segfaults mid-search on some positions (rc=139, no message), and
        // it used to take the whole match with it -- eight games discarded for
        // one crash, which is why the absolute yardstick kept coming back empty.
        //
        // A crashed game has no result, so it is not scored either way: it is
        // retried once on a fresh process with a different seed (replaying the
        // same one just reproduces the crash), and voided if that dies too. The
        // denominator counts only games that finished.
        let mut attempt = 0u64;
        loop {
            match play_one(&model, &params, &mut mohex, g, seed.wrapping_add(attempt * 7919)) {
                Ok((net_won, plies, resigned)) => {
                    score += f64::from(u8::from(net_won));
                    completed += 1;
                    eprintln!(
                        "  game {g}: net={} -> {} ({plies} plies){}",
                        if g % 2 == 0 { "black" } else { "white" },
                        if net_won { "WIN" } else { "loss" },
                        if resigned { " [mohex resigned]" } else { "" },
                    );
                    break;
                }
                Err(e) => {
                    attempt += 1;
                    eprintln!("  game {g}: mohex died ({e})");
                    mohex = Mohex::new(mohex_bin, mohex_playouts, mohex_time)?;
                    if attempt >= 2 {
                        voided += 1;
                        eprintln!("  game {g}: voided after 2 attempts");
                        break;
                    }
                }
            }
        }
    }
    if completed == 0 {
        anyhow::bail!("every game crashed mohex ({voided} voided) -- no score to report");
    }
    if voided > 0 {
        eprintln!("  {voided}/{games} games voided by mohex crashes; scoring the {completed} that finished");
    }
    let net_score = score / f64::from(completed);
    println!("net_vs_mohex {net_score:.3}  ({completed} games, {sims} sims, mohex {mohex_playouts} playouts)");
    Ok(net_score)
}
