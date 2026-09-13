//! A-vs-B evaluation, and the promotion gate.
//!
//! Two measurements, kept separate on purpose (the Hex run's lesson):
//!
//! - **Relative** — a candidate against the incumbent champion. A candidate is
//!   promoted only above a threshold, and the actor generates from `best.*`, so
//!   a net that lost its match never contaminates the replay buffer.
//! - **Absolute** — against the greedy heuristic in `solver.rs`, which is this
//!   game's MoHex analogue: a fixed yardstick that does not drift as the ladder
//!   climbs.
//!
//! Colours alternate so a first-player edge cannot masquerade as strength — and
//! it *would*: the second player starts with `START_MONEY_SECOND` gold and wins
//! mutual-pass games outright under gold adjudication.

use std::path::Path;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use playground_core::sampled_mcts::{search, SampledParams};
use playground_core::{Game, Outcome, Solver};

use crate::game::{Minions, MinionsState};
use crate::net::LocalNet;
use crate::sampler::{MinionsModel, SamplerConfig};
use crate::turn::{realize, MinionsTurns, TurnSpec};

/// One side of a match.
pub enum Seat {
    Net(Box<MinionsModel>),
    /// The greedy one-ply heuristic, driven a whole turn at a time.
    Greedy,
}

impl Seat {
    /// Root evaluation from the side to move, in `[-1, 1]`.
    ///
    /// This is what the bidding phase searches over. The greedy seat has no
    /// value function, so it declines to bid and takes whatever the other side
    /// names — which is the right default for a heuristic that cannot price a
    /// handicap.
    fn root_value(&self, state: &MinionsState, params: &SampledParams) -> anyhow::Result<f64> {
        match self {
            Seat::Net(m) => {
                let r = search::<MinionsTurns, _>(&**m, state.clone(), params, 0)?;
                // `search` reports from the side to move, which at the opening
                // is yellow — the seat `bid_of` measures against.
                Ok(r.value)
            }
            Seat::Greedy => Ok(0.0),
        }
    }

    /// Returns the declaration and the search's candidate-budget accounting. The
    /// stats are the measurement that decides whether progressive widening is
    /// worth building: they say how much of the candidate set a search actually
    /// uses. A greedy seat runs no search and reports nothing.
    fn pick(
        &self,
        state: &MinionsState,
        params: &SampledParams,
        turn: u32,
    ) -> anyhow::Result<(TurnSpec, playground_core::SearchStats)> {
        match self {
            Seat::Net(m) => {
                let r = search::<MinionsTurns, _>(&**m, state.clone(), params, turn)?;
                let spec = r.candidates.get(r.chosen).map(|(s, _)| *s).unwrap_or_default();
                Ok((spec, r.stats))
            }
            // Replay the heuristic's atomic choices and read back what it did as
            // a declaration, so both seats speak the same action space.
            //
            // This used to reconstruct the turn by *diffing* the before and after
            // boards, which recovered only the kills — every move and every
            // purchase the heuristic made was silently discarded, so the greedy
            // seat passed its whole turn and any net scored 1.000 against it.
            // `from_play` tracks each unit's origin as it moves, which is the
            // only way to know which unit ended where.
            Seat::Greedy => Ok((
                TurnSpec::from_play(state, |s| {
                    if Minions::outcome(s).is_terminal() {
                        return None;
                    }
                    Some(Minions::choose_move(s).best_move)
                }),
                playground_core::SearchStats::default(),
            )),
        }
    }
}

/// One seat's bid: the starting-money handicap it believes makes the game even.
///
/// Found by binary search on the value head. `value(bid)` is the root evaluation
/// from *yellow's* seat at that handicap; it falls as blue is paid more, so
/// somewhere in `BID_MIN..=BID_MAX` it crosses zero. The search brackets that
/// crossing to adjacent integers `n` and `n+1`, then interpolates between them:
/// with `v0 = value(n) >= 0` and `v1 = value(n+1) < 0`, the even point sits at
/// `n + v0 / (v0 - v1)`. That fraction is the bid's whole content — rounding it
/// to an integer here would throw away exactly the precision the averaging step
/// needs.
///
/// Clamped rather than extrapolated at the ends: if a side thinks even the
/// widest handicap does not level the game, the honest bid is the edge of the
/// range, not a number the game cannot express.
pub fn bid_of(seat: &Seat, params: &SampledParams) -> anyhow::Result<Option<f64>> {
    // A seat with no value function cannot price a handicap, and must not be
    // *rewarded* for that. Walking the search to an edge would make it the
    // higher bidder every time, so it would take yellow — and the first move —
    // in every game. That silently handed the greedy yardstick the opening seat
    // in every match it played.
    if matches!(seat, Seat::Greedy) {
        return Ok(None);
    }
    bid_value(seat, params).map(Some)
}

fn bid_value(seat: &Seat, params: &SampledParams) -> anyhow::Result<f64> {
    let value_at = |bid: i16| -> anyhow::Result<f64> {
        let state = MinionsState::with_bid(bid);
        seat.root_value(&state, params)
    };

    let (lo, hi) = (crate::game::BID_MIN, crate::game::BID_MAX);
    let v_lo = value_at(lo)?;
    let v_hi = value_at(hi)?;
    // No crossing inside the range: the value head prefers one side throughout,
    // so bid the end that gets closest to even.
    if v_lo < 0.0 {
        return Ok(lo as f64);
    }
    if v_hi >= 0.0 {
        return Ok(hi as f64);
    }

    let (mut a, mut va, mut b) = (lo, v_lo, hi);
    while b - a > 1 {
        let mid = a + (b - a) / 2;
        let vm = value_at(mid)?;
        if vm >= 0.0 {
            a = mid;
            va = vm;
        } else {
            b = mid;
        }
    }
    let vb = value_at(b)?;
    let span = va - vb;
    let frac = if span > 1e-9 { (va / span).clamp(0.0, 1.0) } else { 0.5 };
    Ok(a as f64 + frac)
}

/// Settle the handicap and the seats from both sides' bids.
///
/// Returns `(bid, a_is_yellow)`. The average of the two bids is rounded
/// *probabilistically* — a bid of 3.7 becomes 4 seven times in ten — so that the
/// integer handicap is unbiased with respect to the real-valued one the players
/// actually named.
///
/// The higher bidder asked for blue to be paid more, i.e. claimed yellow is the
/// stronger seat; they are therefore given yellow. That is what makes
/// exaggeration self-punishing: bid too high and you take the side you just
/// argued was advantaged, at a compensation the other player helped set.
pub fn settle_bids<R: rand::Rng + ?Sized>(a_bid: f64, b_bid: f64, rng: &mut R) -> (i16, bool) {
    let mean = 0.5 * (a_bid + b_bid);
    let floor = mean.floor();
    let up = rng.gen::<f64>() < (mean - floor);
    let bid = (floor as i16) + i16::from(up);
    let bid = bid.clamp(crate::game::BID_MIN, crate::game::BID_MAX);
    // Ties go to A by seat convention; any fixed rule is fair once bids are equal.
    (bid, a_bid >= b_bid)
}

/// Play `games` games, alternating who moves first. Returns A's score in [0, 1].
/// `sigma` is `(A, B)`: the KR-PUCT kernel bandwidth each seat searches with.
///
/// Per-seat rather than shared because the match that decides whether the kernel
/// is worth adopting is flat-vs-KR **against the same checkpoint** — both seats
/// load one net and differ only in how the search reads its candidates.
/// One turn for many games at once.
///
/// The reason eval needed this: it used to play games strictly one at a time,
/// so every search ran with a GPU batch of *one* while self-play batched 128.
/// That was invisible at 100 simulations and became the dominant cost of the
/// cycle at 800 — a single 40-game match was ~600k essentially unbatched
/// expansions, around 25 minutes. Here the whole group advances together.
fn pick_many(
    seat: &Seat,
    states: &[MinionsState],
    plies: &[u32],
    params: &SampledParams,
) -> anyhow::Result<(Vec<TurnSpec>, playground_core::SearchStats)> {
    let mut stats = playground_core::SearchStats::default();
    match seat {
        Seat::Net(m) => {
            let results = playground_core::sampled_mcts::search_lockstep::<MinionsTurns, _>(
                &**m,
                states.to_vec(),
                params,
                plies,
            )?;
            let specs = results
                .into_iter()
                .map(|r| {
                    stats.merge(&r.stats);
                    r.candidates.get(r.chosen).map(|(s, _)| *s).unwrap_or_default()
                })
                .collect();
            Ok((specs, stats))
        }
        Seat::Greedy => Ok((
            states
                .iter()
                .map(|st| {
                    TurnSpec::from_play(st, |s| {
                        if Minions::outcome(s).is_terminal() {
                            return None;
                        }
                        Some(Minions::choose_move(s).best_move)
                    })
                })
                .collect(),
            stats,
        )),
    }
}

/// Eval's search: settings of its own, built by the code self-play uses.
///
/// A match keeps the configuration it has always had -- no random candidates,
/// full-depth diffusion on every turn, `SampledParams`' defaults for the rest,
/// and the most-visited turn played -- but goes through
/// `ActorConfig::search_params` like every other search, so a change to how a
/// search is built reaches eval too.
///
/// `pad_to` pads every batch to the match width, so the live count falling as
/// games finish does not set TensorRT rebuilding an engine per shape.
pub fn match_search(
    sims: u32,
    candidates: usize,
    t_steps: usize,
    pad_to: usize,
    seed: u64,
) -> (SamplerConfig, SampledParams) {
    let sampler = SamplerConfig {
        t_steps,
        t_steps_fast: t_steps,
        k_candidates: candidates,
        k_candidates_fast: candidates,
        // One draw per candidate slot, any shortfall filled.
        k_draws: candidates,
        k_draws_fast: candidates,
        random_fraction: 0.0,
        pad_to,
        seed,
        ..Default::default()
    };
    let actor = crate::selfplay::ActorConfig {
        sims,
        temp_moves: 0,
        pw_resume: false,
        ..Default::default()
    };
    (sampler, actor.search_params(&sampler, true, seed))
}

/// `render` is `(path, count)`: write a full turn-by-turn transcript of the
/// first `count` games to `path`.
///
/// Rendering lives here, inside the match, rather than in a separate replay
/// tool: both nets are already loaded and warm, so a sample costs nothing extra
/// and is guaranteed to be *the games the promotion gate actually scored* — a
/// standalone renderer has to reload both checkpoints, contend with the running
/// trainer for the GPU, and still only reproduces a different sample.
pub fn run_match(
    a: &Seat,
    b: &Seat,
    games: u32,
    params: &SampledParams,
    sigma: (f32, f32),
    render: Option<(&std::path::Path, u32)>,
) -> anyhow::Result<(f64, u32, u32, u32, f64, f64)> {
    use std::io::Write as _;
    let mut sink: Option<std::io::BufWriter<std::fs::File>> = match render {
        Some((path, n)) if n > 0 => {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            Some(std::io::BufWriter::new(std::fs::File::create(path)?))
        }
        _ => None,
    };
    let render_games = render.map_or(0, |(_, n)| n);
    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    let mut margin_sum = 0.0f64;
    let mut score_sum = 0.0f64;
    // Game length split by verdict. A side that is genuinely stronger should
    // close games out *sooner* than it loses them: a short win means the
    // necromancer fell or the graveyards were taken outright, while a long loss
    // is usually a position ground down to the turn cap. Equal lengths mean the
    // score is coming from adjudication, not from play.
    let (mut win_turns, mut loss_turns) = (0u32, 0u32);
    let mut search_stats = playground_core::SearchStats::default();
    let (mut necro, mut gy, mut adj, mut turn_sum) = (0u32, 0u32, 0u32, 0u32);
    // Both seats bid once; the handicap and the seats follow from the bids, so
    // a match is no longer decided by who happens to sit where.
    // Each seat prices its handicap with the search it will actually play with,
    // or the bid measures a net the match never uses.
    let a_bid = bid_of(a, &SampledParams { kernel_sigma: sigma.0, ..*params })?;
    let b_bid = bid_of(b, &SampledParams { kernel_sigma: sigma.1, ..*params })?;
    // Plain numbers so the eval loop can parse them; "none" for a seat that
    // declines (it has no value function to price a handicap with).
    let show = |b: Option<f64>| b.map_or("none".to_string(), |v| format!("{v:.3}"));
    eprintln!("bids: A {}  B {}", show(a_bid), show(b_bid));

    // ── set every game up, then advance them all together ────────────────
    struct Live {
        g: u32,
        state: MinionsState,
        a_first: bool,
        bid: i16,
        done: bool,
        /// Per-game transcript, buffered. Games advance in lockstep now, so
        /// writing turns straight to the sink would interleave every game's
        /// moves into an unreadable stream; each game's log is emitted whole
        /// once the match ends.
        log: String,
    }
    let mut live: Vec<Live> = (0..games)
        .map(|g| {
            // Alternate which side's bid is treated as "A's" so a systematic
            // error in one seat's valuation cannot hand it the good side.
            let mut r =
                ChaCha8Rng::seed_from_u64(params.seed ^ (g as u64).wrapping_mul(0x51_7C_C1_B7));
            let (bid, a_yellow) = match (a_bid, b_bid) {
                (Some(x), Some(y)) => {
                    if g % 2 == 0 {
                        settle_bids(x, y, &mut r)
                    } else {
                        let (bd, b_takes) = settle_bids(y, x, &mut r);
                        (bd, !b_takes)
                    }
                }
                // Only one side can bid: its number stands unopposed and the
                // seats simply alternate. Letting the silent side "win" the
                // auction would be the bug this replaced.
                (Some(x), None) => (settle_bids(x, x, &mut r).0, g % 2 == 0),
                (None, Some(y)) => (settle_bids(y, y, &mut r).0, g % 2 == 0),
                (None, None) => (crate::game::START_MONEY_SECOND as i16, g % 2 == 0),
            };
            Live {
                g,
                state: MinionsState::with_bid(bid),
                a_first: a_yellow,
                bid,
                done: false,
                log: String::new(),
            }
        })
        .collect();

    if sink.is_some() {
        for l in live.iter_mut().filter(|l| l.g < render_games) {
            use std::fmt::Write as _;
            let _ = write!(
                l.log,
                "\n===== game {} of {render_games}  |  A is {}  |  handicap {} =====\n{}\n",
                l.g + 1,
                if l.a_first { "yellow (moves first)" } else { "blue" },
                l.bid,
                l.state.render()
            );
        }
    }

    let mut round = 0u32;
    loop {
        for l in live.iter_mut() {
            if !l.done && Minions::outcome(&l.state).is_terminal() {
                l.done = true;
            }
        }
        if live.iter().all(|l| l.done) {
            break;
        }
        // Both seats move in the same round: every live game is at the same
        // side, so `a_first` alone decides which seat owns it this round.
        for want_a in [true, false] {
            let idxs: Vec<usize> = live
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    !l.done && ((l.state.side == 0) == l.a_first) == want_a
                })
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                continue;
            }
            let seat = if want_a { a } else { b };
            let states: Vec<MinionsState> = idxs.iter().map(|&i| live[i].state.clone()).collect();
            let plies: Vec<u32> = idxs.iter().map(|&i| live[i].state.turn as u32).collect();
            let mut p = *params;
            p.kernel_sigma = if want_a { sigma.0 } else { sigma.1 };
            p.seed = params.seed ^ ((round as u64) << 32) ^ u64::from(want_a);
            let (specs, st) = pick_many(seat, &states, &plies, &p)?;
            search_stats.merge(&st);

            for (&i, spec) in idxs.iter().zip(specs) {
                let prev_state = live[i].state.clone();
                let next = realize(&prev_state, &spec).next;
                // A declaration that changes nothing would loop forever; end
                // the turn regardless.
                live[i].state = if Minions::encode_state(&next) == Minions::encode_state(&prev_state)
                {
                    realize(&prev_state, &TurnSpec::empty()).next
                } else {
                    next
                };
                if sink.is_some() && live[i].g < render_games {
                    use std::fmt::Write as _;
                    let r = realize(&prev_state, &spec);
                    let mover = if want_a { "A" } else { "B" };
                    let (nw0, nw1) = (
                        crate::game::net_worth(&live[i].state, 0),
                        crate::game::net_worth(&live[i].state, 1),
                    );
                    let (turn, money, board) =
                        (live[i].state.turn, live[i].state.money, live[i].state.render());
                    let _ = write!(
                        live[i].log,
                        "turn {turn} by {mover}  gold {money:?}  net worth [{nw0}, {nw1}]  \
                         actions {}\n{board}\n",
                        r.actions.len()
                    );
                }
                if Minions::outcome(&live[i].state).is_terminal() {
                    live[i].done = true;
                }
            }
        }
        round += 1;
    }

    // ── score, once each game has finished ────────────────────────────────
    for l in live.iter() {
        let state = &l.state;
        let a_first = l.a_first;
        if l.g < render_games {
            if let Some(w) = sink.as_mut() {
                write!(w, "{}", l.log)?;
                writeln!(
                    w,
                    "outcome: {:?} after {} turns (cap {})",
                    Minions::outcome(state),
                    state.turn,
                    crate::game::MAX_TURNS
                )?;
            }
        }
        // Classify the ending before scoring it.
        turn_sum += state.turn as u32;
        if state.turn >= crate::game::MAX_TURNS {
            adj += 1;
        } else if (0..2u8).any(|sd| {
            !(0..crate::map::NUM_CELLS as u8).any(|c| {
                state
                    .piece_at(c)
                    .is_some_and(|(s2, u)| s2 == sd && matches!(u, crate::units::Unit::Necromancer))
            })
        }) {
            necro += 1;
        } else {
            gy += 1;
        }

        // Net-worth margin from A's seat, recorded whatever the verdict.
        //
        // Win/loss can be a *constant* here: the outcome at the turn cap is
        // largely decided by seat, and since the match alternates seats it then
        // scores exactly 0.5 no matter how strong either side is — which is
        // what pinned the promotion gate. Margin still separates them: a
        // stronger net loses by less.
        let (a_side, b_side) = if a_first { (0u8, 1u8) } else { (1u8, 0u8) };
        margin_sum += crate::game::net_worth(state, a_side) as f64
            - crate::game::net_worth(state, b_side) as f64;
        // Terminal score from A's seat: the margin above plus a speed term, so
        // a fast win reads higher than a grinding one that ends level.
        score_sum += crate::game::game_score(state, a_side) as f64;

        let a_won = match Minions::outcome(state) {
            Outcome::FirstWins => a_first,
            Outcome::SecondWins => !a_first,
            _ => {
                draws += 1;
                continue;
            }
        };
        if a_won {
            wins += 1;
            win_turns += state.turn as u32;
        } else {
            losses += 1;
            loss_turns += state.turn as u32;
        }
    }
    let score = (wins as f64 + 0.5 * draws as f64) / games.max(1) as f64;
    let margin = margin_sum / games.max(1) as f64;
    // Named apart from `score` above, which is the win rate. This is the mean
    // terminal `game_score` in points.
    let game_score = score_sum / games.max(1) as f64;
    // How the games ended, not just who won. Material parity with a losing
    // record means the opponent is winning on a *condition* — necromancer kill
    // or graveyard control — rather than on net worth, and the score alone
    // cannot distinguish those.
    eprintln!(
        "endings: necromancer {necro}  graveyards {gy}  adjudicated {adj}  mean turns {:.1}",
        turn_sum as f64 / games.max(1) as f64
    );
    // Reported as -1 rather than 0 when a bucket is empty, so a match A never
    // won is not read as "A wins in zero turns".
    let mean = |sum: u32, n: u32| if n == 0 { -1.0 } else { sum as f64 / n as f64 };
    eprintln!(
        "turns: win {:.1}  loss {:.1}  wins {wins}  losses {losses}",
        mean(win_turns, wins),
        mean(loss_turns, losses)
    );
    // P0 for progressive widening: how much of each node's candidate set the
    // search ever tries. Candidate generation is the dominant cost, so a high
    // waste fraction is the whole argument for growing the set with visits.
    let h = search_stats.visit_hist;
    eprintln!(
        "candidates: {} over {} nodes, {} never tried ({:.1}% waste)",
        search_stats.candidates,
        search_stats.nodes,
        search_stats.unvisited,
        100.0 * search_stats.waste()
    );
    eprintln!(
        "node visits: 1:{} 2:{} 3-4:{} 5-8:{} 9-16:{} 17-32:{} 33-64:{} 65+:{}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    );
    Ok((score, wins, losses, draws, margin, game_score))
}

/// A seat backed by a checkpoint directory or file, on whichever backend.
pub fn load_seat(
    path: &Path,
    kind: &crate::net::BackendKind,
    sampler: SamplerConfig,
) -> anyhow::Result<Seat> {
    let net: Box<dyn crate::net::Backend> = match kind {
        crate::net::BackendKind::Candle { cuda } => Box::new(LocalNet::load(path, *cuda)?),
        crate::net::BackendKind::Ort { ep, fp16 } => {
            let dir = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
            let cache = dir.join("trt_cache").to_string_lossy().into_owned();
            Box::new(crate::ort_net::OrtNet::new(dir, ep, *fp16, &cache)?)
        }
    };
    Ok(Seat::Net(Box::new(MinionsModel::new(net, sampler))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{NO_MOVE, TARGET_STAY};

    /// Building eval's search through `search_params` must not change it: the
    /// match is the one eval ran before the builder was shared.
    #[test]
    fn match_search_is_evals_historical_search() {
        let (sampler, params) = match_search(400, 16, 4, 40, 9);
        let historical = SampledParams {
            simulations: 400,
            k_candidates: 16,
            // Newer than the rest: one draw per candidate.
            k_draws: 16,
            temp_moves: 0,
            seed: 9,
            ..Default::default()
        };
        assert_eq!(format!("{params:?}"), format!("{historical:?}"));
        assert_eq!(sampler.random_fraction, 0.0, "eval draws no random candidates");
        assert_eq!((sampler.t_steps, sampler.t_steps_fast), (4, 4));
        assert_eq!((sampler.k_candidates, sampler.pad_to, sampler.seed), (16, 40, 9));
    }

    /// The higher bidder claimed yellow was stronger, so they get yellow.
    ///
    /// This is the whole incentive structure: bid too high and you are handed
    /// the side you just argued was advantaged, at a compensation your opponent
    /// helped set. Get it backwards and exaggeration becomes free.
    #[test]
    fn the_higher_bidder_takes_yellow() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let (_, a_yellow) = settle_bids(7.0, 2.0, &mut rng);
        assert!(a_yellow, "A asked more for blue, so A must take yellow");
        let (_, a_yellow) = settle_bids(2.0, 7.0, &mut rng);
        assert!(!a_yellow, "B asked more for blue, so B takes yellow");
    }

    /// The settled handicap is the average of the bids, rounded probabilistically
    /// so it is unbiased with respect to the real-valued bids.
    #[test]
    fn settled_handicap_averages_the_bids_without_bias() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        // Mean 4.0 exactly: must always land on 4.
        for _ in 0..50 {
            let (bid, _) = settle_bids(3.0, 5.0, &mut rng);
            assert_eq!(bid, 4);
        }
        // Mean 3.75: should round up about three times in four, and *only* ever
        // to 3 or 4.
        let mut ups = 0;
        let n = 4000;
        for _ in 0..n {
            let (bid, _) = settle_bids(3.5, 4.0, &mut rng);
            assert!(bid == 3 || bid == 4, "rounding must stay adjacent, got {bid}");
            if bid == 4 {
                ups += 1;
            }
        }
        let frac = ups as f64 / n as f64;
        assert!((frac - 0.75).abs() < 0.03, "expected ~0.75 rounded up, got {frac:.3}");
    }

    /// Bids outside the expressible range are clamped, not extrapolated.
    #[test]
    fn settled_handicap_stays_inside_the_bid_range() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let (hi, _) = settle_bids(100.0, 100.0, &mut rng);
        assert_eq!(hi, crate::game::BID_MAX);
        let (lo, _) = settle_bids(-100.0, -100.0, &mut rng);
        assert_eq!(lo, crate::game::BID_MIN);
    }

    /// The greedy seat cannot price a handicap, so it declines to bid — and a
    /// flat-zero value must still yield a bid inside the range rather than a
    /// panic or an extrapolation.
    /// A seat that cannot price a handicap must *decline*, not bid an edge.
    ///
    /// Regression: it walked the search to `BID_MAX`, which made it the higher
    /// bidder in every match — and since the higher bidder takes yellow, the
    /// greedy yardstick was silently handed the opening seat in every game it
    /// ever played.
    #[test]
    fn a_seat_without_a_value_function_declines_to_bid() {
        let params = SampledParams::default();
        assert_eq!(bid_of(&Seat::Greedy, &params).expect("greedy answers"), None);
    }

    /// The greedy seat must actually play its turn.
    ///
    /// Regression: the declaration was reconstructed by diffing the boards, which
    /// recovered kills only — every move and purchase was dropped, so greedy
    /// passed every turn and *any* net scored 1.000 against it. Since greedy is
    /// the promotion anchor whenever there is no champion, that silently made the
    /// gate meaningless.
    #[test]
    fn greedy_seat_declares_the_moves_it_made() {
        let params = SampledParams::default();
        let mut state = Minions::initial_state();
        let (mut moved, mut spawned) = (0u32, 0u32);
        for turn in 0..12u32 {
            if Minions::outcome(&state).is_terminal() {
                break;
            }
            let (spec, _) = Seat::Greedy.pick(&state, &params, turn).expect("greedy picks");
            moved += spec.moves.iter().filter(|&&m| m != NO_MOVE).count() as u32;
            spawned += spec.spawns.iter().filter(|&&x| x != 0).count() as u32;
            let next = realize(&state, &spec).next;
            assert_ne!(
                Minions::encode_state(&next),
                Minions::encode_state(&state),
                "greedy turn {turn} changed nothing — the seat is passing"
            );
            state = next;
        }
        assert!(moved > 0, "greedy never declared a move");
        assert!(spawned > 0, "greedy never declared a spawn — its purchases are lost");
    }

    /// And what it declares has to survive realization, or the seat is still
    /// weaker than the heuristic it stands for.
    #[test]
    fn greedy_declarations_are_realizable() {
        let params = SampledParams::default();
        let mut state = Minions::initial_state();
        let (mut declared, mut done) = (0u32, 0u32);
        for turn in 0..12u32 {
            if Minions::outcome(&state).is_terminal() {
                break;
            }
            let (spec, _) = Seat::Greedy.pick(&state, &params, turn).expect("greedy picks");
            let r = realize(&state, &spec);
            declared += r.fidelity.declared_moves;
            done += r.fidelity.done_moves;
            let _ = TARGET_STAY;
            state = r.next;
        }
        assert!(declared > 0);
        let f = done as f64 / declared as f64;
        assert!(f > 0.9, "greedy move fidelity {f:.3} — the seat loses what it chose");
    }
}
