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
                let r = search::<MinionsTurns, _>(&**m, state.clone(), &m.tune_params(params), 0)?;
                // `search` reports from the side to move, which at the opening
                // is yellow — the seat `bid_of` measures against.
                Ok(r.value)
            }
            Seat::Greedy => Ok(0.0),
        }
    }

    /// Returns the declaration and the search's candidate-budget accounting.
    /// A greedy seat runs no search and reports nothing.
    ///
    /// Test-only since match play moved to `pick_many`, which drives both
    /// seats' searches in lockstep so they can be batched. The tests still on
    /// this path all use `Seat::Greedy`, which never searches.
    #[cfg(test)]
    fn pick(
        &self,
        state: &MinionsState,
        params: &SampledParams,
        turn: u32,
    ) -> anyhow::Result<(TurnSpec, playground_core::SearchStats)> {
        match self {
            Seat::Net(m) => {
                let r =
                    search::<MinionsTurns, _>(&**m, state.clone(), &m.tune_params(params), turn)?;
                let spec = r
                    .candidates
                    .get(r.chosen)
                    .map(|(s, _)| *s)
                    .unwrap_or_default();
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
    let frac = if span > 1e-9 {
        (va / span).clamp(0.0, 1.0)
    } else {
        0.5
    };
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
    want_search: bool,
) -> anyhow::Result<(
    Vec<TurnSpec>,
    playground_core::SearchStats,
    Vec<Option<RootSearch>>,
)> {
    let mut stats = playground_core::SearchStats::default();
    match seat {
        Seat::Net(m) => {
            let tuned = m.tune_params(params);
            // A best-move seat draws once per node; see `tune_params`.
            let results = if want_search {
                playground_core::sampled_mcts::search_lockstep_analyzed::<MinionsTurns, _>(
                    &**m,
                    states.to_vec(),
                    &tuned,
                    plies,
                )?
            } else {
                playground_core::sampled_mcts::search_lockstep::<MinionsTurns, _>(
                    &**m,
                    states.to_vec(),
                    &tuned,
                    plies,
                )?
                .into_iter()
                .map(|r| (r, Vec::new()))
                .collect()
            };
            let mut roots = Vec::with_capacity(results.len());
            let specs = results
                .into_iter()
                .enumerate()
                .map(|(j, (r, tree))| {
                    stats.merge(&r.stats);
                    roots.push(
                        tree.first()
                            .map(|root| root_search(&states[j], root, r.chosen, r.value)),
                    );
                    r.candidates
                        .get(r.chosen)
                        .map(|(s, _)| *s)
                        .unwrap_or_default()
                })
                .collect();
            Ok((specs, stats, roots))
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
            vec![None; states.len()],
        )),
    }
}

/// Round to four decimals. These are display numbers; full `f32` precision
/// would roughly double the recording for digits no one reads.
fn r4(x: f32) -> f32 {
    (x * 1e4).round() / 1e4
}

/// The root of one search, shaped like `AnalyzeNode`/`AnalyzeEdge` so the
/// recorded view can render it with the same component as live analysis.
///
/// Root only, and capped: the full tree for a match this size would be tens of
/// megabytes, and the analysis panel shows the root's candidate list anyway.
fn root_search(
    state: &MinionsState,
    root: &playground_core::sampled_mcts::TreeNode<MinionsTurns>,
    chosen: usize,
    value: f64,
) -> RootSearch {
    /// Edges kept, by visits. The search draws 24; the tail is all single
    /// visits and carries no information worth the bytes.
    const TOP_K: usize = 16;
    let mut idx: Vec<usize> = (0..root.edges.len()).collect();
    // `chosen` indexes the entry order, which `export_tree` preserves, so the
    // flag is set before sorting rather than by matching moves back up.
    idx.sort_by_key(|&i| std::cmp::Reverse(root.edges[i].visits));
    idx.truncate(TOP_K);
    RootSearch {
        mover: String::new(),
        total_visits: root.total_visits,
        value: Some(r4(value as f32)),
        nn_value: root.nn_value.map(r4),
        edges: idx
            .into_iter()
            .map(|i| {
                let e = &root.edges[i];
                RootEdge {
                    label: crate::net_seat::label_turn(state, &e.mv),
                    visits: e.visits,
                    forced: e.forced,
                    draws: e.draws,
                    explore: e.explore,
                    prior: r4(e.prior),
                    energy: r4(e.energy),
                    value: e.value.map(r4),
                    prior_value: e.prior_value.map(r4),
                    chosen: i == chosen,
                }
            })
            .collect(),
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
/// `GameState.payload` as hex — the exact bytes `Minions::encode_state` emits,
/// which is what the web board already decodes.
///
/// Hex rather than base64 because no base64 crate is in the tree, and the
/// doubling costs about half a megabyte on an 80-game match. Not worth a
/// dependency, and not worth a fourth hand-rolled copy of the state format.
fn hex_state(s: &MinionsState) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(266);
    for b in Minions::encode_state(s) {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// One root candidate, mirroring `AnalyzeEdge`.
#[derive(serde::Serialize, Clone)]
struct RootEdge {
    label: String,
    visits: u32,
    forced: u32,
    draws: u32,
    explore: bool,
    prior: f32,
    energy: f32,
    value: Option<f32>,
    prior_value: Option<f32>,
    /// The turn actually played from this position.
    chosen: bool,
}

/// One turn's search at its root, mirroring `AnalyzeNode`.
#[derive(serde::Serialize, Clone)]
struct RootSearch {
    /// Which seat searched this turn: "a" or "b".
    mover: String,
    total_visits: u32,
    value: Option<f32>,
    nn_value: Option<f32>,
    edges: Vec<RootEdge>,
}

/// One game: how it was set up, how it ended, and every position along the way.
#[derive(serde::Serialize)]
struct GameRecord {
    game: u32,
    /// Settled handicap, and whether seat A took yellow (and so moved first).
    bid: i16,
    a_first: bool,
    turns: u16,
    /// From A's seat: "a", "b" or "draw".
    result: String,
    /// Index 0 is the opening position; each later entry is the position after
    /// one applied turn. One entry per turn, which is the unit the move list
    /// groups by -- positions inside a turn are never offered to anything.
    ///
    /// `states.len()` is NOT always `turns + 1`. A game ends the instant a
    /// necromancer dies, part-way through a turn, so the final applied turn
    /// need not advance the counter. Measured on the first two recorded games:
    /// 30 turns/31 states, but 16 turns/18 states. Walk the list; do not index
    /// it by turn number.
    states: Vec<String>,
    /// The search behind each turn, parallel to `states[1..]`. Empty unless
    /// `--record-search` was given.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    search: Vec<RootSearch>,
}

/// A whole match, self-contained: the viewer needs no engine to replay it.
#[derive(serde::Serialize)]
struct MatchRecord {
    games: u32,
    a_wins: u32,
    b_wins: u32,
    draws: u32,
    a_score: f64,
    /// Free-form description of what each seat was, written by the caller.
    a_label: String,
    b_label: String,
    records: Vec<GameRecord>,
}

pub fn run_match(
    a: &Seat,
    b: &Seat,
    games: u32,
    params: (&SampledParams, &SampledParams),
    render: Option<(&std::path::Path, u32)>,
    record: Option<&std::path::Path>,
    record_search: bool,
) -> anyhow::Result<(f64, u32, u32, u32, f64, f64)> {
    // Each seat searches with its own params (sims, kernel, exploration); the
    // match's seed is A's.
    let (params_a, params_b) = params;
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
    // Started before the bids, not after: bidding runs a real search on each
    // seat, so a profile that skipped it would undercount the very work an A/B
    // is trying to price.
    let t_match = std::time::Instant::now();
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
    // Per seat too, for what only one seat may be doing (resampling).
    let mut seat_stats = [playground_core::SearchStats::default(); 2];
    let (mut necro, mut gy, mut adj, mut turn_sum) = (0u32, 0u32, 0u32, 0u32);
    // Both seats bid once; the handicap and the seats follow from the bids, so
    // a match is no longer decided by who happens to sit where.
    // Each seat prices its handicap with the search it will actually play with,
    // or the bid measures a net the match never uses.
    let a_bid = bid_of(a, params_a)?;
    let b_bid = bid_of(b, params_b)?;
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
        /// The position after each turn, hex-encoded. Kept for every game
        /// regardless of `render_games`: the transcript is a human sample, but
        /// a recording the viewer replays should be the whole match.
        states: Vec<String>,
        /// The search behind each turn, parallel to `states[1..]`.
        search: Vec<RootSearch>,
    }
    let mut live: Vec<Live> = (0..games)
        .map(|g| {
            // Alternate which side's bid is treated as "A's" so a systematic
            // error in one seat's valuation cannot hand it the good side.
            let mut r =
                ChaCha8Rng::seed_from_u64(params_a.seed ^ (g as u64).wrapping_mul(0x51_7C_C1_B7));
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
            let st = MinionsState::with_bid(bid);
            Live {
                g,
                states: if record.is_some() {
                    vec![hex_state(&st)]
                } else {
                    Vec::new()
                },
                search: Vec::new(),
                state: st,
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
                if l.a_first {
                    "yellow (moves first)"
                } else {
                    "blue"
                },
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
                .filter(|(_, l)| !l.done && ((l.state.side == 0) == l.a_first) == want_a)
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                continue;
            }
            let seat = if want_a { a } else { b };
            let states: Vec<MinionsState> = idxs.iter().map(|&i| live[i].state.clone()).collect();
            let plies: Vec<u32> = idxs.iter().map(|&i| live[i].state.turn as u32).collect();
            let mut p = if want_a { *params_a } else { *params_b };
            p.seed = params_a.seed ^ ((round as u64) << 32) ^ u64::from(want_a);
            let (specs, st, roots) = pick_many(seat, &states, &plies, &p, record_search)?;
            search_stats.merge(&st);
            seat_stats[usize::from(!want_a)].merge(&st);

            // `j` indexes this round's live games, which is what `roots` is
            // parallel to; `i` indexes `live`.
            for (j, (&i, spec)) in idxs.iter().zip(specs).enumerate() {
                let prev_state = live[i].state.clone();
                let next = realize(&prev_state, &spec).next;
                // A declaration that changes nothing would loop forever; end
                // the turn regardless.
                live[i].state =
                    if Minions::encode_state(&next) == Minions::encode_state(&prev_state) {
                        realize(&prev_state, &TurnSpec::empty()).next
                    } else {
                        next
                    };
                if record.is_some() {
                    let h = hex_state(&live[i].state);
                    live[i].states.push(h);
                    // The mover is known here, not inside the search.
                    if let Some(mut rs) = roots.get(j).cloned().flatten() {
                        rs.mover = if want_a { "a" } else { "b" }.to_string();
                        live[i].search.push(rs);
                    }
                }
                if sink.is_some() && live[i].g < render_games {
                    use std::fmt::Write as _;
                    let r = realize(&prev_state, &spec);
                    let mover = if want_a { "A" } else { "B" };
                    let (nw0, nw1) = (
                        crate::game::net_worth(&live[i].state, 0),
                        crate::game::net_worth(&live[i].state, 1),
                    );
                    let (turn, money, board) = (
                        live[i].state.turn,
                        live[i].state.money,
                        live[i].state.render(),
                    );
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
    let mut records: Vec<GameRecord> = Vec::new();
    for l in live.iter() {
        let state = &l.state;
        let a_first = l.a_first;
        // Built here, before the draw arm below `continue`s -- a recording that
        // silently dropped drawn games would misnumber every game after one.
        if record.is_some() {
            let result = match Minions::outcome(state) {
                Outcome::FirstWins => {
                    if a_first {
                        "a"
                    } else {
                        "b"
                    }
                }
                Outcome::SecondWins => {
                    if a_first {
                        "b"
                    } else {
                        "a"
                    }
                }
                _ => "draw",
            };
            records.push(GameRecord {
                game: l.g,
                bid: l.bid,
                a_first,
                turns: state.turn,
                result: result.to_string(),
                states: l.states.clone(),
                search: l.search.clone(),
            });
        }
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
    // Per-seat phase profile. Self-play reports this through `report_wall`; a
    // head-to-head needs it too, because a seat can buy strength with time and
    // the score alone will not say so. Absolute milliseconds as well as shares:
    // what an A/B wants to know is what the extra work *cost*, and a percentage
    // of measured work cannot answer that — a seat doing twice the work reports
    // the same 100%. Both seats share one wall clock (games run in lockstep),
    // so compare the measured totals, not the wall time.
    let match_secs = t_match.elapsed().as_secs_f64();
    eprintln!("match wall: {match_secs:.1}s over {games} games");
    for (seat, s) in ["A", "B"].iter().zip([a, b]) {
        if let Seat::Net(m) = s {
            let p = &m.phases;
            let g = |c: &std::sync::atomic::AtomicU64| {
                c.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
            };
            let measured = g(&p.features_ns)
                + g(&p.trunk_ns)
                + g(&p.decl_write_ns)
                + g(&p.diffusion_ns)
                + g(&p.commit_ns)
                + g(&p.realize_ns)
                + g(&p.energy_ns)
                + g(&p.playout_ns)
                + g(&p.last_stage_ns);
            eprintln!(
                "phases ({seat}): measured {measured:.0}ms, last-stage {:.0}ms — {}",
                g(&p.last_stage_ns),
                p.report_wall(match_secs, 1)
            );
        }
    }
    for (seat, st) in ["A", "B"].iter().zip(&seat_stats) {
        if st.resamples > 0 {
            eprintln!(
                "resamples ({seat}): {} requests ({} at roots), {} new candidates; played a resampled turn in {} of {} turns",
                st.resamples, st.resamples_root, st.resample_new, st.chosen_resampled, st.searches
            );
        }
    }
    if let Some(path) = record {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let rec = MatchRecord {
            games,
            a_wins: wins,
            b_wins: losses,
            draws,
            a_score: score,
            // Self-describing: what actually differs between the seats in
            // these experiments is the value rule, so the recording carries it
            // rather than making the viewer be told out of band.
            a_label: format!("{:?}/{:?}", params_a.value_backup, params_a.value_visits),
            b_label: format!("{:?}/{:?}", params_b.value_backup, params_b.value_visits),
            records,
        };
        // Written whole at the end rather than streamed: a half-written file
        // from an interrupted match decodes as corrupt, and the viewer has no
        // way to tell that from a bug in the format.
        std::fs::write(path, serde_json::to_vec(&rec)?)?;
        eprintln!("recorded {} games to {}", rec.records.len(), path.display());
    }
    Ok((score, wins, losses, draws, margin, game_score))
}

/// A seat backed by a checkpoint directory or file, on whichever backend.
pub fn load_seat(
    path: &Path,
    kind: &crate::net::BackendKind,
    sampler: SamplerConfig,
    // Candidate-ordering net for `SamplerConfig::last_stage_samples`. Per
    // seat, not shared: the evaluator owns scratch, and a match runs both
    // seats against one checkpoint where only this differs.
    nnue: Option<crate::nnue::Evaluator>,
) -> anyhow::Result<Seat> {
    let net: Box<dyn crate::net::Backend> = match kind {
        crate::net::BackendKind::Candle { cuda } => Box::new(LocalNet::load(path, *cuda)?),
        crate::net::BackendKind::Ort { ep, fp16 } => {
            let dir = if path.is_dir() {
                path
            } else {
                path.parent().unwrap_or(path)
            };
            let cache = dir.join("trt_cache").to_string_lossy().into_owned();
            Box::new(crate::ort_net::OrtNet::new(dir, ep, *fp16, &cache)?)
        }
    };
    Ok(Seat::Net(Box::new(
        MinionsModel::new(net, sampler).with_nnue(nnue),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{NO_MOVE, TARGET_STAY};

    /// Diagnostic, run by hand: a best-move seat against a diffusion seat on
    /// the same trunk. Prints each seat's root value across the bid range, then
    /// walks a game the diffusion seat plays against itself and reports, per
    /// position, where the head's own turn stands in the best seat's root --
    /// prior rank, share of the chosen visits, whether it is played -- and
    /// whether it is the turn the diffusion seat's search plays.
    ///
    /// `MINIONS_DIAG_DIRS=<best dir>,<diffusion dir> cargo test --release
    /// best_head_diagnostic -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn best_head_diagnostic() {
        let dirs =
            std::env::var("MINIONS_DIAG_DIRS").expect("MINIONS_DIAG_DIRS=<best>,<diffusion>");
        let (best_dir, dif_dir) = dirs.split_once(',').expect("two directories");
        // Padded to eval's width so the cached TensorRT engines serve it.
        let (sampler, params) = match_search(400, 16, 4, 40, 11);
        let kind = crate::net::BackendKind::parse("ort", "trt", true, false).unwrap();
        let best = load_seat(Path::new(best_dir), &kind, sampler, None).unwrap();
        let dif = load_seat(Path::new(dif_dir), &kind, sampler, None).unwrap();
        for (name, seat) in [("best", &best), ("diffusion", &dif)] {
            let curve: Vec<String> = (crate::game::BID_MIN..=crate::game::BID_MAX)
                .map(|b| {
                    let v = seat
                        .root_value(&MinionsState::with_bid(b), &params)
                        .unwrap();
                    format!("{b}:{v:+.3}")
                })
                .collect();
            println!(
                "{name}: bid {:?}  root value by handicap {}",
                bid_of(seat, &params).unwrap(),
                curve.join(" ")
            );
        }
        let (Seat::Net(bm), Seat::Net(dm)) = (&best, &dif) else {
            unreachable!("both seats are nets")
        };
        let bp = bm.tune_params(&params);
        let mut state = MinionsState::with_bid(3);
        let (mut rows, mut played_n, mut same_n, mut top_n) = (0, 0, 0, 0);
        let mut share_sum = 0.0;
        while rows < 30 && !Minions::outcome(&state).is_terminal() {
            let turn = state.turn as u32;
            let (br, tree) = playground_core::sampled_mcts::search_analyzed::<MinionsTurns, _>(
                &**bm,
                state.clone(),
                &bp,
                turn,
            )
            .unwrap();
            let dr = search::<MinionsTurns, _>(&**dm, state.clone(), &params, turn).unwrap();
            let dif_turn = dr.candidates[dr.chosen].0;
            let key = |s: &TurnSpec| Minions::encode_state(&realize(&state, s).next);
            let root = &tree[0];
            // The head's turn is the root's one drawn candidate; fills have 0.
            let Some(head) = root.edges.iter().find(|e| e.draws > 0) else {
                println!("turn {turn}: no drawn candidate at the root");
                break;
            };
            let rank = root.edges.iter().filter(|e| e.prior > head.prior).count();
            let chosen: u32 = root.edges.iter().map(|e| e.visits - e.forced).sum();
            let share = f64::from(head.visits - head.forced) / f64::from(chosen.max(1));
            let played = key(&br.candidates[br.chosen].0) == key(&head.mv);
            let same = key(&head.mv) == key(&dif_turn);
            println!(
                "turn {turn:2}: head prior rank {rank:2} of {:2} (p {:.3}), chosen-visit share \
                 {share:.2}, played {played:5}, = diffusion's pick {same:5} | root value best \
                 {:+.3} diffusion {:+.3}",
                root.edges.len(),
                head.prior,
                br.value,
                dr.value
            );
            rows += 1;
            played_n += usize::from(played);
            same_n += usize::from(same);
            top_n += usize::from(rank == 0);
            share_sum += share;
            state = realize(&state, &dif_turn).next;
        }
        println!(
            "{rows} positions: head turn has the top prior {top_n}, is played {played_n}, is \
             the diffusion seat's pick {same_n}; mean chosen-visit share {:.2}",
            share_sum / f64::from(rows.max(1))
        );
    }

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
        assert_eq!(
            sampler.random_fraction, 0.0,
            "eval draws no random candidates"
        );
        assert_eq!((sampler.t_steps, sampler.t_steps_fast), (4, 4));
        assert_eq!(
            (sampler.k_candidates, sampler.pad_to, sampler.seed),
            (16, 40, 9)
        );
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
            assert!(
                bid == 3 || bid == 4,
                "rounding must stay adjacent, got {bid}"
            );
            if bid == 4 {
                ups += 1;
            }
        }
        let frac = ups as f64 / n as f64;
        assert!(
            (frac - 0.75).abs() < 0.03,
            "expected ~0.75 rounded up, got {frac:.3}"
        );
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
        assert_eq!(
            bid_of(&Seat::Greedy, &params).expect("greedy answers"),
            None
        );
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
            let (spec, _) = Seat::Greedy
                .pick(&state, &params, turn)
                .expect("greedy picks");
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
        assert!(
            spawned > 0,
            "greedy never declared a spawn — its purchases are lost"
        );
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
            let (spec, _) = Seat::Greedy
                .pick(&state, &params, turn)
                .expect("greedy picks");
            let r = realize(&state, &spec);
            declared += r.fidelity.declared_moves;
            done += r.fidelity.done_moves;
            let _ = TARGET_STAY;
            state = r.next;
        }
        assert!(declared > 0);
        let f = done as f64 / declared as f64;
        assert!(
            f > 0.9,
            "greedy move fidelity {f:.3} — the seat loses what it chose"
        );
    }
}
