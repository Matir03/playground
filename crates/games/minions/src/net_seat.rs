//! The trained net as an interactive seat, for playing it in the web UI.
//!
//! The UI asks for one *atomic* move at a time (`Move`, `Attack`, `Buy`,
//! `Spawn`, `EndPhase`), but the net's action space is a whole **turn** — a
//! `TurnSpec` scored by the sampled search. Bridging the two naively (re-run the
//! search for every atomic move) would pay a full 200-simulation search 10-25
//! times per turn.
//!
//! So a search runs once per turn: its chosen `TurnSpec` is realized into the
//! atomic sequence the engine would have played, and that sequence is handed out
//! one action per call. The plan is keyed by the exact state it was made from,
//! and re-planned whenever the caller's state is not the one the plan expects —
//! so a takeback, a new game, or any desync silently re-plans instead of
//! replaying a stale turn.
//!
//! Wired in as [`Solver::choose_move_external`] rather than `EnginePlayer::Nn`
//! deliberately: the `Nn` path drives a *per-move* policy through the Python
//! model server, which cannot represent a turn-level net. This path loads the
//! ONNX/safetensors weights in-process, exactly as self-play does.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;

use playground_core::sampled_mcts::{drive, search, Search, TreeNode};
use playground_core::{Game, SearchOutcome, SolverBudget};

use crate::game::{Minions, MinionsMove, MinionsState};
use crate::sampler::{MinionsModel, SamplerConfig};
use crate::turn::{realize, MinionsTurns, TurnSpec};

/// The best-first step search as the seat instead (`serve --steps-net`),
/// see [`configure_steps`].
static STEPS: Mutex<Option<StepsSeat>> = Mutex::new(None);

/// A step-search seat (docs/minions-steps.md): per turn a fresh best-first
/// search for a fixed time (the UI's `max_millis`, else `secs`), then the
/// principal variation's turn, handed out one action at a time like the net's.
#[derive(Clone)]
pub struct StepsSeat {
    pub eval: std::sync::Arc<dyn crate::bfs::Evaluator>,
    pub params: crate::bfs::Params,
    pub secs: f64,
    /// Expansion cap per turn (the UI's `max_visits` overrides it).
    pub max_expansions: u64,
}

/// The step-search seat, if `serve --steps-net` configured one.
pub(crate) fn steps_seat() -> Option<StepsSeat> {
    STEPS.lock().unwrap().clone()
}

pub fn configure_steps(seat: StepsSeat) {
    *STEPS.lock().unwrap() = Some(seat);
    *PLAN.lock().unwrap() = None;
}

/// Where the weights live, and how to run them. Set once at startup by `serve`.
static CONFIG: Mutex<Option<NetSeatConfig>> = Mutex::new(None);
/// The model, built on first use so `serve` starts instantly even without a GPU.
static MODEL: Mutex<Option<Box<MinionsModel>>> = Mutex::new(None);
/// The turn currently being handed out, one action at a time.
static PLAN: Mutex<Option<Plan>> = Mutex::new(None);
/// The search `serve` started with, as JSON: what the UI's reset restores.
static DEFAULT: Mutex<Option<String>> = Mutex::new(None);
/// Searches run, so each draws its own RNG stream (temperature picks and
/// forced-quota draws) the way self-play's do.
static SEARCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Analysis sessions: searches kept alive between requests, so a reader can
/// step one, force visits down lines of their choosing, and add turns of their
/// own. Few, least recently used out first: each holds a whole tree.
static SESSIONS: Mutex<Option<Sessions>> = Mutex::new(None);
const MAX_SESSIONS: usize = 8;

#[derive(Default)]
struct Sessions {
    next: u64,
    clock: u64,
    live: HashMap<u64, Session>,
}

struct Session {
    root: MinionsState,
    search: Search<MinionsTurns>,
    /// Which of self-play's searches this is: every expansion and widen the
    /// session makes calls the model the way that search does.
    full: bool,
    used: u64,
}

#[derive(Clone)]
pub struct NetSeatConfig {
    pub dir: PathBuf,
    pub kind: crate::net::BackendKind,
    /// Self-play's recorded search settings (see `search_config`).
    pub actor: crate::selfplay::ActorConfig,
    pub sampler: SamplerConfig,
}

struct Plan {
    /// The state the next queued action applies to. Encoded rather than cloned
    /// so the comparison is exactly the engine's own notion of state identity.
    expect: Vec<u8>,
    actions: VecDeque<MinionsMove>,
    /// Root value of the search that produced this turn, repeated on every
    /// action replayed from it.
    ///
    /// The search runs once per *turn* and the seat hands back one action per
    /// call, so all but the first action of a turn come from the queue with no
    /// new search behind them. Reporting 0.0 for those made the UI read 0.00
    /// for most of a turn — and since a turn's *last* action is always a
    /// replay, anything keeping the latest value saw nothing else.
    value: f64,
}

/// Called by `serve` when `--net-dir` is given.
///
/// Draws a fresh sampler seed unless the caller pinned one, so two engine runs
/// do not replay the same game. Within a run, `MinionsModel::expand_batch`
/// already advances its own stream per expansion.
pub fn configure(mut cfg: NetSeatConfig) {
    if cfg.sampler.seed == SamplerConfig::default().seed {
        cfg.sampler.seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
    }
    configure_inner(cfg)
}

fn configure_inner(cfg: NetSeatConfig) {
    *DEFAULT.lock().unwrap() = Some(crate::search_config::to_json(&cfg.actor, &cfg.sampler));
    *CONFIG.lock().unwrap() = Some(cfg);
    *MODEL.lock().unwrap() = None;
    *PLAN.lock().unwrap() = None;
}

fn search_seed(cfg: &NetSeatConfig) -> u64 {
    let n = SEARCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    cfg.sampler.seed ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn current_search() -> anyhow::Result<playground_core::SearchConfigJson> {
    let cfg = CONFIG
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no net configured"))?;
    let default = DEFAULT
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no starting search recorded"))?;
    Ok(playground_core::SearchConfigJson {
        current: crate::search_config::to_json(&cfg.actor, &cfg.sampler),
        default,
    })
}

/// Replace the search the seat and analysis run, or restore the one `serve`
/// started with. It applies from the next search on: a loaded model keeps its
/// weights and takes the new sampler settings, and the seat's planned turn is
/// dropped so it re-plans under them.
fn set_search(json: Option<&str>) -> anyhow::Result<playground_core::SearchConfigJson> {
    // MODEL before CONFIG, the order analysis holds them in (`aux_for` reads
    // CONFIG with the model held), so the two cannot deadlock.
    let mut model = MODEL.lock().unwrap();
    {
        let mut guard = CONFIG.lock().unwrap();
        let cfg = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no net configured"))?;
        let text = match json {
            Some(t) => t.to_owned(),
            None => DEFAULT
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("no starting search recorded"))?,
        };
        let (mut actor, mut sampler) = (cfg.actor.clone(), cfg.sampler);
        crate::search_config::apply_json(&text, &mut actor, &mut sampler)?;
        cfg.actor = actor;
        cfg.sampler = sampler;
        if let Some(m) = model.as_mut() {
            m.cfg = sampler;
        }
    }
    *PLAN.lock().unwrap() = None;
    drop(model);
    current_search()
}

pub fn is_configured() -> bool {
    CONFIG.lock().unwrap().is_some() || STEPS.lock().unwrap().is_some() || crate::lattice_ui::is_configured()
}

/// Plan a whole turn with the step search, then return its first action.
fn plan_steps_turn(
    state: &MinionsState,
    budget: &SolverBudget,
    seat: StepsSeat,
) -> anyhow::Result<SearchOutcome<MinionsMove>> {
    let secs = budget
        .max_millis
        .map(|ms| f64::from(ms) / 1000.0)
        .unwrap_or(seat.secs);
    let cap = budget
        .max_visits
        .map(u64::from)
        .unwrap_or(seat.max_expansions);
    let salt = SEARCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ^ Minions::encode_state(state)
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let t = crate::stepsp::search_turn(state, seat.eval.as_ref(), &seat.params, cap, secs, salt)?;
    let mut actions: VecDeque<MinionsMove> = crate::steps::turn_actions(state, &t.done)?.into();
    let first = actions
        .pop_front()
        .ok_or_else(|| anyhow::anyhow!("the searched turn was empty"))?;
    // The value shown is the side to move's win probability as -1..1.
    let value = 2.0 * f64::from(t.value) - 1.0;
    let next_state = Minions::apply(state, &first);
    *PLAN.lock().unwrap() = Some(Plan {
        expect: Minions::encode_state(&next_state),
        actions,
        value,
    });
    Ok(SearchOutcome {
        best_move: first,
        value,
        nodes: t.expansions,
    })
}

/// Plan a whole turn from `state`, then return its first action.
fn plan_turn(
    state: &MinionsState,
    budget: &SolverBudget,
) -> anyhow::Result<SearchOutcome<MinionsMove>> {
    let steps = STEPS.lock().unwrap().clone();
    if let Some(seat) = steps {
        return plan_steps_turn(state, budget, seat);
    }
    if crate::lattice_ui::is_configured() {
        // The lattice player (`serve --lattice-net`): tier-1 search over
        // turns, tier 2 over each turn's lattice; the UI's max_visits
        // overrides its tier-1 visits.
        let (actions, value) = crate::lattice_ui::plan(state, budget.max_visits)?;
        let mut actions: VecDeque<MinionsMove> = actions.into();
        let first = actions
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("the searched turn was empty"))?;
        let next_state = Minions::apply(state, &first);
        *PLAN.lock().unwrap() = Some(Plan {
            expect: Minions::encode_state(&next_state),
            actions,
            value,
        });
        return Ok(SearchOutcome {
            best_move: first,
            value,
            nodes: 0,
        });
    }
    let cfg = CONFIG
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no net configured"))?;

    let mut guard = MODEL.lock().unwrap();
    if guard.is_none() {
        let seat = crate::eval::load_seat(&cfg.dir, &cfg.kind, cfg.sampler.clone(), None)?;
        match seat {
            crate::eval::Seat::Net(m) => *guard = Some(m),
            crate::eval::Seat::Greedy => anyhow::bail!("load_seat returned the greedy seat"),
        }
    }
    let model = guard.as_ref().expect("model built above");

    // Self-play's full search under the engine's current settings -- the ones
    // self-play recorded unless the UI changed them -- and its move choice,
    // temperature included: for the first `temp_moves` plies the turn is a
    // pick in proportion to chosen visits, exactly as in self-play. Set
    // `temp_moves` to 0 for the most-visited turn every time.
    //
    // (Sampling once handed back a null turn -- "the net passes on turn 1" --
    // because it counted forced visits. It counts chosen visits only now.)
    let mut params = cfg
        .actor
        .search_params(&cfg.sampler, true, search_seed(&cfg));
    if let Some(v) = budget.max_visits {
        params.simulations = v;
    }

    let result = search::<MinionsTurns, _>(&**model, state.clone(), &params, Minions::ply(state))?;
    // Falling back to `TurnSpec::default()` would be a null turn, so an empty
    // candidate set is an error instead.
    let spec: TurnSpec = result
        .candidates
        .get(result.chosen)
        .map(|(s, _)| *s)
        .ok_or_else(|| anyhow::anyhow!("search returned no candidates"))?;

    let realized = realize(state, &spec);
    let mut actions: VecDeque<MinionsMove> = realized.actions.into_iter().collect();
    let first = actions
        .pop_front()
        .ok_or_else(|| anyhow::anyhow!("the realized turn was empty"))?;

    // Remember the state the *next* action applies to.
    let next_state = Minions::apply(state, &first);
    *PLAN.lock().unwrap() = Some(Plan {
        expect: Minions::encode_state(&next_state),
        actions,
        value: result.value,
    });

    Ok(SearchOutcome {
        best_move: first,
        value: result.value,
        nodes: params.simulations as u64,
    })
}

/// One atomic move from the net, replaying the current turn's plan when it
/// still applies and re-planning when it does not.
pub fn choose(
    state: &MinionsState,
    budget: SolverBudget,
) -> Option<anyhow::Result<SearchOutcome<MinionsMove>>> {
    if !is_configured() {
        return None;
    }
    let here = Minions::encode_state(state);
    {
        let mut plan = PLAN.lock().unwrap();
        if let Some(p) = plan.as_mut() {
            if p.expect == here {
                if let Some(mv) = p.actions.pop_front() {
                    let next = Minions::apply(state, &mv);
                    p.expect = Minions::encode_state(&next);
                    // `nodes: 0` is honest — no search ran for this action —
                    // but the value is the planned turn's, not zero.
                    return Some(Ok(SearchOutcome {
                        best_move: mv,
                        value: p.value,
                        nodes: 0,
                    }));
                }
            }
        }
        // Stale or exhausted: drop it and plan afresh below.
        *plan = None;
    }
    Some(plan_turn(state, &budget))
}

#[cfg(test)]
mod tests {
    use super::*;
    use playground_core::Solver;

    /// A turn played action by action becomes a turn the tree can hold: its
    /// declaration realizes to the position returned, which is where the
    /// actions led unless `differs` says otherwise, and half a turn or an
    /// illegal action is refused.
    #[test]
    fn a_played_turn_becomes_a_tree_turn() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let enc = |s: &MinionsState| Minions::encode_state(s);
        let live = |s: &MinionsState| Minions::outcome(s) == playground_core::Outcome::InProgress;
        let mut from = Minions::initial_state();
        let (mut turns, mut exact) = (0, 0);
        while turns < 40 && live(&from) {
            let mut cur = from.clone();
            let mut moves = Vec::new();
            while cur.side == from.side && live(&cur) {
                let legal = Minions::legal_moves(&cur);
                let mv = legal[rng.gen_range(0..legal.len())].clone();
                moves.push(Minions::encode_move(&mv));
                cur = Minions::apply(&cur, &mv);
            }
            let (spec, next, differs) = turn_from_moves(&from, &moves).unwrap();
            assert_eq!(enc(&realize(&from, &spec).next), enc(&next));
            if !differs {
                exact += 1;
                assert_eq!(enc(&next), enc(&cur));
            }
            assert!(
                turn_from_moves(&from, &moves[..moves.len() - 1]).is_err(),
                "the last action is what ended the turn"
            );
            turns += 1;
            from = next;
        }
        assert!(
            turns >= 10 && exact > 0,
            "{turns} turns, {exact} read back exactly"
        );
        assert!(
            turn_from_moves(&from, &[vec![9, 9, 9]]).is_err(),
            "not a move"
        );
    }

    /// Is the value head emitting NaN, or is the NaN made during backup?
    #[test]
    fn value_head_is_finite() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            return;
        };
        let d = std::path::Path::new(&dir);
        let cache = d.join("trt_cache").to_string_lossy().into_owned();
        let ort = match crate::ort_net::OrtNet::new(d, "cuda", false, &cache) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e})");
                return;
            }
        };
        use crate::net::Backend;
        let mut st = Minions::initial_state();
        for step in 0..4 {
            let planes: Vec<f32> = Minions::to_features(&st)
                .iter()
                .map(|&b| b as f32)
                .collect();
            let out = ort.trunk(&planes, 1).unwrap();
            let nan_h = out.h.iter().filter(|x| !x.is_finite()).count();
            println!(
                "step {step}: value={:?} finite={} | trunk non-finite {} of {}",
                out.value,
                out.value.iter().all(|v| v.is_finite()),
                nan_h,
                out.h.len()
            );
            let legal = Minions::legal_moves(&st);
            if legal.is_empty() {
                break;
            }
            st = Minions::apply(&st, &legal[0]);
        }
    }

    /// The same position must not always produce the same turn.
    ///
    /// `expand_batch` used to pass `batch_seq: 0`, collapsing the RNG to a
    /// constant, so the sampler proposed identical candidates every time and —
    /// with argmax selection — the net replayed one game move for move.
    #[test]
    fn opening_turns_vary_between_games() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping variety test");
            return;
        };
        configure(NetSeatConfig {
            dir: dir.into(),
            kind: crate::net::BackendKind::Ort {
                ep: "trt".into(),
                fp16: true,
            },
            actor: crate::selfplay::ActorConfig {
                sims: 64,
                ..Default::default()
            },
            sampler: SamplerConfig {
                k_candidates: 12,
                t_steps: 4,
                ..Default::default()
            },
        });
        let budget = SolverBudget {
            max_millis: None,
            max_visits: None,
        };

        let mut seen = std::collections::HashSet::new();
        for _ in 0..6 {
            *PLAN.lock().unwrap() = None;
            let mut state = Minions::initial_state();
            let mut turn = Vec::new();
            loop {
                let mv = Minions::choose_move_external(&state, budget)
                    .expect("configured")
                    .expect("search ok")
                    .best_move;
                turn.push(format!("{mv:?}"));
                state = Minions::apply(&state, &mv);
                if matches!(mv, MinionsMove::EndPhase) || Minions::outcome(&state).is_terminal() {
                    break;
                }
            }
            seen.insert(turn.join("|"));
        }
        println!("distinct opening turns over 6 games: {}", seen.len());
        assert!(
            seen.len() > 1,
            "every opening turn was identical over 6 games — the sampler's RNG is pinned again"
        );
    }

    /// The net must not pass on its opening turn.
    ///
    /// It did, and the cause was subtle: `SampledParams::temp_moves` defaults to
    /// 8, so for the first eight plies the search returns a *visit-proportional
    /// sample* rather than the argmax — right for self-play diversity, wrong
    /// against a human. Forced exploration always puts a null `TurnSpec` in the
    /// candidate set, so that sample could legitimately land on "do nothing".
    /// Observed directly: argmax had 32 of 64 visits while the sampled pick had 4.
    #[test]
    fn net_seat_does_not_pass_on_the_opening_turn() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping opening-turn test");
            return;
        };
        // The most-visited turn: the seat now follows the configured
        // temperature, as self-play does, and this is about the argmax.
        configure(NetSeatConfig {
            dir: dir.into(),
            kind: crate::net::BackendKind::Ort {
                ep: "trt".into(),
                fp16: true,
            },
            actor: crate::selfplay::ActorConfig {
                sims: 64,
                temp_moves: 0,
                ..Default::default()
            },
            sampler: SamplerConfig {
                k_candidates: 12,
                t_steps: 4,
                ..Default::default()
            },
        });
        let budget = SolverBudget {
            max_millis: None,
            max_visits: None,
        };

        // Repeat: the old bug was stochastic, so one clean turn proves nothing.
        for attempt in 0..5 {
            *PLAN.lock().unwrap() = None;
            let mut state = Minions::initial_state();
            let mut actions = 0;
            while !Minions::outcome(&state).is_terminal() {
                let mv = Minions::choose_move_external(&state, budget)
                    .expect("configured")
                    .expect("search ok")
                    .best_move;
                state = Minions::apply(&state, &mv);
                actions += 1;
                if matches!(mv, MinionsMove::EndPhase) {
                    break;
                }
            }
            assert!(
                actions > 1,
                "attempt {attempt}: the net passed its opening turn \
                 ({actions} action, EndPhase only) — temp_moves sampling is back"
            );
        }
    }

    /// One search per turn, and every action it hands back is legal.
    ///
    /// The plan cache is the subtle part: it must hand out the turn's actions in
    /// order, and re-plan the moment the caller's state is not the one it
    /// expects. Needs `MINIONS_ORT_DIR` and a GPU.
    #[test]
    fn net_seat_plays_a_legal_turn_from_one_search() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping net seat test");
            return;
        };
        configure(NetSeatConfig {
            dir: dir.into(),
            kind: crate::net::BackendKind::Ort {
                ep: "cuda".into(),
                fp16: false,
            },
            actor: crate::selfplay::ActorConfig {
                sims: 16,
                ..Default::default()
            },
            sampler: SamplerConfig {
                k_candidates: 8,
                t_steps: 2,
                ..Default::default()
            },
        });

        let mut state = Minions::initial_state();
        let budget = SolverBudget {
            max_millis: None,
            max_visits: None,
        };
        let mut acted = 0;
        // Every action replayed from a plan must carry that plan's value.
        // `nodes > 0` marks the one action a search actually ran for; the rest
        // come from the queue. They used to report a hardcoded 0.0, so a UI
        // showing the newest value read 0.00 for all but the first action of
        // every turn — and a turn's last action is always a replay.
        let mut planned_value: Option<f64> = None;
        for _ in 0..40 {
            if Minions::outcome(&state).is_terminal() {
                break;
            }
            let out = match Minions::choose_move_external(&state, budget) {
                Some(Ok(o)) => o,
                Some(Err(e)) => panic!("net seat failed: {e}"),
                None => panic!("net seat reported unconfigured after configure()"),
            };
            let legal = Minions::legal_moves(&state);
            assert!(
                legal.contains(&out.best_move),
                "net seat returned an illegal move {:?}; legal = {} moves",
                out.best_move,
                legal.len()
            );
            if out.nodes > 0 {
                planned_value = Some(out.value);
            } else {
                // NaN-tolerant: this test runs the CUDA EP, whose trunk output
                // is all-NaN (see the `--ep trt` note in the render path), so a
                // plain `==` would fail on NaN != NaN and hide the real check.
                // Carrying a NaN through is still the right behaviour — what
                // must not happen is a replay inventing 0.0.
                let planned = planned_value.expect("a replay before any search");
                assert!(
                    out.value == planned || (out.value.is_nan() && planned.is_nan()),
                    "a replayed action must report the planned turn's value \
                     ({planned}), not a placeholder; got {}",
                    out.value
                );
            }
            state = Minions::apply(&state, &out.best_move);
            acted += 1;
        }
        assert!(
            acted > 1,
            "expected the seat to play several actions, got {acted}"
        );

        // Analysis must see the same net without disturbing the seat: a tree
        // with the numbers the search selected on, and the aux heads if the
        // exported net has them.
        {
            use playground_core::Analyzable;
            let st = Minions::initial_state();
            let opts = playground_core::AnalysisOptions {
                candidates: Some(8),
                ..Default::default()
            };
            let a = Minions::analyze(&st, budget, opts)
                .expect("configured")
                .expect("analysis");
            assert!(!a.nodes.is_empty(), "a search analysis must return a tree");
            assert_eq!(a.nodes[0].parent, None, "node 0 must be the root");
            assert!(
                a.nodes[0].total_visits > 0,
                "the root must have been visited"
            );
            let chosen = a.chosen.expect("a completed search chose a move");
            assert!(
                chosen < a.nodes[0].edges.len(),
                "chosen indexes past the root's candidates"
            );
            for e in &a.nodes[0].edges {
                assert_eq!(e.mv.len(), 3 * crate::map::NUM_CELLS, "turn encoding size");
                // Walking the variation must reproduce what the search searched.
                let next = Minions::apply_encoded_turn(&st, &e.mv).expect("turn applies");
                assert_ne!(
                    Minions::encode_state(&next),
                    Minions::encode_state(&st),
                    "a turn that changes nothing would make the tree unwalkable"
                );
            }

            // Eval-only must skip the search entirely, not return a stub tree.
            let ev = Minions::analyze(
                &st,
                budget,
                playground_core::AnalysisOptions {
                    eval_only: true,
                    ..Default::default()
                },
            )
            .expect("configured")
            .expect("analysis");
            assert!(ev.nodes.is_empty(), "eval_only must not build a tree");
            assert!(ev.chosen.is_none(), "eval_only has no move to choose");
        }

        // A state the plan does not expect must force a re-plan, not a stale replay.
        let fresh = Minions::initial_state();
        let out = Minions::choose_move_external(&fresh, budget)
            .expect("configured")
            .expect("ok");
        assert!(
            Minions::legal_moves(&fresh).contains(&out.best_move),
            "after a desync the seat must re-plan and return a move legal *here*"
        );
    }
}

// ── Analysis ──────────────────────────────────────────────────────────────────
//
// Looking without playing. The interactive seat above answers "what do you
// play"; this answers "what do you see", which is a different question and has
// to leave the game alone to be worth asking.

/// The `aux.onnx` session, built on first use.
///
/// Separate from the seat's backend because the auxiliary heads are their own
/// graph — self-play never runs them, so the backend has no reason to hold one.
static AUX: Mutex<Option<ort::session::Session>> = Mutex::new(None);

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Run the auxiliary heads, or `None` if this net has no `aux.onnx`.
///
/// A missing graph is expected rather than exceptional: nets exported before
/// the aux graph existed are still perfectly playable, and the view degrades to
/// hiding those panels.
/// Must match `SCORE_SCALE` in python/src/playground/model_server/minions_net.py.
/// The head is trained on `score / SCORE_SCALE`; this undoes it so every reader
/// gets plain points. A silent disagreement here would only ever show up as an
/// analysis panel whose numbers are off by a constant factor.
const SCORE_SCALE: f32 = 150.0;

fn aux_for(state: &MinionsState) -> Option<playground_core::AnalysisAux> {
    let cfg = CONFIG.lock().unwrap().clone()?;
    let (ep, fp16) = match &cfg.kind {
        crate::net::BackendKind::Ort { ep, fp16 } => (ep.clone(), *fp16),
        // Candle has no ONNX graphs to run.
        crate::net::BackendKind::Candle { .. } => return None,
    };
    let path = cfg.dir.join("aux.onnx");
    if !path.exists() {
        return None;
    }

    let mut guard = AUX.lock().unwrap();
    if guard.is_none() {
        let cache = cfg.dir.join("trt_cache");
        let sess = crate::ort_net::build(&path, &ep, fp16, &cache.to_string_lossy()).ok()?;
        *guard = Some(sess);
    }
    let sess = guard.as_mut()?;

    let mut planes = vec![0f32; crate::game::PLANES * crate::map::NUM_CELLS];
    for (o, b) in planes.iter_mut().zip(Minions::to_features(state)) {
        *o = b as f32;
    }
    let n = crate::map::N as i64;
    let input = ort::value::TensorRef::from_array_view((
        vec![1i64, crate::game::PLANES as i64, n, n],
        planes.as_slice(),
    ))
    .ok()?;
    let out = sess.run(ort::inputs!["planes" => input]).ok()?;

    let take = |name: &str| -> Option<Vec<f32>> {
        Some(out.get(name)?.try_extract_tensor::<f32>().ok()?.1.to_vec())
    };
    let hex = take("hex")?;
    let necro = take("necro")?;
    let logz = take("logz")?;
    // Optional: a checkpoint exported before the score head existed has no such
    // output, and analysis should still work against it rather than going dark.
    let score = take("score")
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    let cells = crate::map::NUM_CELLS;
    let horizon = necro.len() / 2;
    // Sigmoids here, not in each consumer: the heads emit logits, and a UI that
    // applied its own would be one refactor away from disagreeing with the next
    // reader about what the numbers mean.
    Some(playground_core::AnalysisAux {
        hex_control: [
            hex[..cells].iter().copied().map(sigmoid).collect(),
            hex[cells..2 * cells].iter().copied().map(sigmoid).collect(),
        ],
        necro_hazard: [
            necro[..horizon].iter().copied().map(sigmoid).collect(),
            necro[horizon..2 * horizon]
                .iter()
                .copied()
                .map(sigmoid)
                .collect(),
        ],
        log_z: logz.first().copied().unwrap_or(0.0),
        // Trained on score / SCORE_SCALE, so undo that here and hand every
        // consumer plain points.
        score: score * SCORE_SCALE,
    })
}

/// Encode a turn so a client can hand it back to walk a variation.
pub fn encode_turn(spec: &TurnSpec) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 * crate::map::NUM_CELLS);
    out.extend_from_slice(&spec.moves);
    out.extend_from_slice(&spec.targets);
    out.extend_from_slice(&spec.spawns);
    out
}

/// Inverse of [`encode_turn`].
pub fn decode_turn(bytes: &[u8]) -> anyhow::Result<TurnSpec> {
    let c = crate::map::NUM_CELLS;
    if bytes.len() != 3 * c {
        anyhow::bail!("turn encoding is {} bytes, expected {}", bytes.len(), 3 * c);
    }
    let mut spec = TurnSpec::default();
    spec.moves.copy_from_slice(&bytes[..c]);
    spec.targets.copy_from_slice(&bytes[c..2 * c]);
    spec.spawns.copy_from_slice(&bytes[2 * c..]);
    Ok(spec)
}

/// Render a turn as the sequence of actions it realizes to.
pub(crate) fn label_turn(state: &MinionsState, spec: &TurnSpec) -> String {
    let actions = realize(state, spec).actions;
    if actions.is_empty() {
        return "pass".to_string();
    }
    actions
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

impl playground_core::Analyzable for Minions {
    fn analyze(
        state: &MinionsState,
        budget: SolverBudget,
        opts: playground_core::AnalysisOptions,
    ) -> Option<anyhow::Result<playground_core::Analysis>> {
        if let Some(seat) = steps_seat() {
            return Some(if opts.eval_only {
                crate::steps_analysis::eval_only(&seat, state)
            } else {
                crate::steps_analysis::analyze(&seat, state, budget)
            });
        }
        if !is_configured() {
            return None;
        }
        Some(analyze_inner(state, budget, opts))
    }

    fn apply_encoded_turn(state: &MinionsState, turn: &[u8]) -> anyhow::Result<MinionsState> {
        if let Some(board) = crate::steps_analysis::apply(turn) {
            return board;
        }
        Ok(realize(state, &decode_turn(turn)?).next)
    }

    fn commit_encoded_turn(state: &MinionsState, turn: &[u8]) -> anyhow::Result<MinionsState> {
        if let Some(seat) = steps_seat() {
            if let Some(next) = crate::steps_analysis::commit(&seat, turn) {
                return next;
            }
        }
        Self::apply_encoded_turn(state, turn)
    }

    // The sampled-AZ search's settings; the step seat has none of these.
    fn search_config() -> Option<anyhow::Result<playground_core::SearchConfigJson>> {
        CONFIG.lock().unwrap().is_some().then(current_search)
    }

    fn set_search_config(
        json: Option<&str>,
    ) -> Option<anyhow::Result<playground_core::SearchConfigJson>> {
        CONFIG.lock().unwrap().is_some().then(|| set_search(json))
    }

    fn session(
        op: playground_core::SessionOp<MinionsState>,
    ) -> Option<anyhow::Result<playground_core::SessionResult<MinionsState>>> {
        if let Some(seat) = steps_seat() {
            return Some(crate::steps_analysis::session(&seat, op));
        }
        is_configured().then(|| session_inner(op))
    }
}

/// The seat's model, loaded on first use. Takes the held guard, so callers keep
/// the MODEL-before-CONFIG lock order `set_search` relies on.
fn ensure_model<'a>(
    guard: &'a mut Option<Box<MinionsModel>>,
    cfg: &NetSeatConfig,
) -> anyhow::Result<&'a MinionsModel> {
    if guard.is_none() {
        match crate::eval::load_seat(&cfg.dir, &cfg.kind, cfg.sampler.clone(), None)? {
            crate::eval::Seat::Net(m) => *guard = Some(m),
            crate::eval::Seat::Greedy => anyhow::bail!("load_seat returned the greedy seat"),
        }
    }
    Ok(guard.as_deref().expect("model built above"))
}

fn session_inner(
    op: playground_core::SessionOp<MinionsState>,
) -> anyhow::Result<playground_core::SessionResult<MinionsState>> {
    use playground_core::sampled_mcts::SampledModel;
    use playground_core::SessionOp;
    let cfg = CONFIG
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no net configured"))?;
    // MODEL, then SESSIONS: one order everywhere, so the two cannot deadlock.
    let mut guard = MODEL.lock().unwrap();
    let model = ensure_model(&mut guard, &cfg)?;
    let mut sessions = SESSIONS.lock().unwrap();
    let store = sessions.get_or_insert_with(Sessions::default);
    store.clock += 1;
    let now = store.clock;
    let mut added = None;
    let mut note = None;

    let id = match op {
        SessionOp::Start {
            root,
            simulations,
            fast,
        } => {
            let full = !fast;
            let params = cfg
                .actor
                .search_params(&cfg.sampler, full, search_seed(&cfg));
            let mut search = Search::new(root.clone(), &params, Minions::ply(&root));
            search.set_remaining(if simulations == 0 {
                params.simulations
            } else {
                simulations
            });
            drive(&crate::sampler::AsSearch { model, full }, &mut search)?;
            if store.live.len() >= MAX_SESSIONS {
                let oldest = store
                    .live
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k);
                if let Some(k) = oldest {
                    store.live.remove(&k);
                }
            }
            store.next += 1;
            let session = Session {
                root,
                search,
                full,
                used: now,
            };
            store.live.insert(store.next, session);
            store.next
        }
        SessionOp::Step { id, simulations } => {
            let s = session_mut(store, id)?;
            s.search.add_simulations(simulations);
            drive(
                &crate::sampler::AsSearch {
                    model,
                    full: s.full,
                },
                &mut s.search,
            )?;
            id
        }
        SessionOp::Force { id, line } => {
            let s = session_mut(store, id)?;
            let specs = line
                .iter()
                .map(|b| decode_turn(b))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let entries = s.search.resolve_line(&specs)?;
            s.search.force_line(&entries)?;
            drive(
                &crate::sampler::AsSearch {
                    model,
                    full: s.full,
                },
                &mut s.search,
            )?;
            id
        }
        SessionOp::Add { id, at, moves } => {
            let s = session_mut(store, id)?;
            let specs = at
                .iter()
                .map(|b| decode_turn(b))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let node = s.search.node_at(&s.search.resolve_line(&specs)?)?;
            let from = s.search.state_of(node).clone();
            let (spec, next, differs) = turn_from_moves(&from, &moves)?;
            let (_, energy) = model.score_turns(&from, &[spec])?;
            let v = match <MinionsTurns as playground_core::SampledGame>::terminal_value(&next) {
                Some(v) => v,
                None => model.expand(&next, 0)?.0,
            };
            // From the side to move at `from`, as every edge value is.
            let eval = if next.side != from.side { -v } else { v };
            let (i, new) = s.search.add_candidate(node, spec, energy[0], Some(eval))?;
            // The entry as the tree holds it: when the turn merged into an
            // existing candidate, that candidate's encoding, not this one's.
            added = Some(encode_turn(s.search.candidate(node, i)));
            note = match (new, differs) {
                (false, _) => Some(
                    "that turn reaches a position the node already holds, so it is that candidate"
                        .to_string(),
                ),
                (true, true) => Some(
                    "added as the realizer reads it, which differs from how it was played: a \
                     purchase with no spawn, or damage that kills nothing, is not part of a turn"
                        .to_string(),
                ),
                (true, false) => None,
            };
            id
        }
    };

    let s = session_mut(store, id)?;
    s.used = now;
    let tree = s.search.export_tree();
    Ok(playground_core::SessionResult {
        id,
        root: s.root.clone(),
        analysis: playground_core::Analysis {
            value: s.search.root_value(),
            aux: aux_for(&s.root),
            nodes: analysis_nodes(&s.root, &tree),
            chosen: (!tree[0].edges.is_empty()).then(|| s.search.chosen()),
        },
        added,
        note,
    })
}

fn session_mut(store: &mut Sessions, id: u64) -> anyhow::Result<&mut Session> {
    store
        .live
        .get_mut(&id)
        .ok_or_else(|| anyhow::anyhow!("analysis session {id} has expired; start a new one"))
}

/// The turn a sequence of atomic moves plays from `from`, as the search holds
/// turns: the declaration `TurnSpec::from_play` reads back, realized. Also the
/// position that reaches, and whether it differs from where the moves
/// themselves led -- a purchase without a spawn, or damage that kills nothing,
/// is not part of any declaration.
fn turn_from_moves(
    from: &MinionsState,
    moves: &[Vec<u8>],
) -> anyhow::Result<(TurnSpec, MinionsState, bool)> {
    let played = moves
        .iter()
        .map(|b| Minions::decode_move(b))
        .collect::<anyhow::Result<Vec<MinionsMove>>>()?;
    let in_turn = |s: &MinionsState| {
        s.side == from.side && Minions::outcome(s) == playground_core::Outcome::InProgress
    };
    let mut cur = from.clone();
    for (i, mv) in played.iter().enumerate() {
        anyhow::ensure!(
            in_turn(&cur),
            "action {} comes after the turn has ended",
            i + 1
        );
        let legal = Minions::legal_moves(&cur)
            .iter()
            .any(|m| Minions::encode_move(m) == Minions::encode_move(mv));
        anyhow::ensure!(legal, "action {} ({mv}) is not legal there", i + 1);
        cur = Minions::apply(&cur, mv);
    }
    anyhow::ensure!(
        !in_turn(&cur),
        "the turn is not finished: end it before adding it"
    );
    let mut queue = played.iter().copied();
    let spec = TurnSpec::from_play(from, |_| queue.next());
    let r = realize(from, &spec);
    let differs = Minions::encode_state(&r.next) != Minions::encode_state(&cur);
    Ok((r.spec, r.next, differs))
}

/// The flattened tree as the analysis view reads it: every edge labelled with
/// the actions its turn realizes to, from the position it is played in.
fn analysis_nodes(
    root: &MinionsState,
    tree: &[TreeNode<MinionsTurns>],
) -> Vec<playground_core::AnalysisNode> {
    // Labels need the state each turn is played from, so walk the tree
    // alongside the states rather than labelling from the root throughout.
    let mut states: Vec<Option<MinionsState>> = vec![None; tree.len()];
    states[0] = Some(root.clone());
    for i in 0..tree.len() {
        let here = match &states[i] {
            Some(s) => s.clone(),
            // Unreachable in a well-formed tree; skip rather than panic on one.
            None => continue,
        };
        for e in &tree[i].edges {
            if let Some(c) = e.child {
                if c < states.len() && states[c].is_none() {
                    states[c] = Some(realize(&here, &e.mv).next);
                }
            }
        }
    }

    tree.iter()
        .enumerate()
        .map(|(i, n)| {
            let here = states[i].clone().unwrap_or_else(|| root.clone());
            playground_core::AnalysisNode {
                parent: n.parent,
                total_visits: n.total_visits,
                value: n.value,
                terminal: n.terminal,
                term_value: n.term_value,
                prior_value: n.nn_value,
                edges: n
                    .edges
                    .iter()
                    .map(|e| playground_core::AnalysisEdge {
                        mv: encode_turn(&e.mv),
                        label: label_turn(&here, &e.mv),
                        prior: e.prior,
                        energy: e.energy,
                        visits: e.visits,
                        value: e.value,
                        child: e.child,
                        explore: e.explore,
                        forced: e.forced,
                        draws: e.draws,
                        prior_value: e.prior_value,
                        manual: e.manual,
                    })
                    .collect(),
                kernel: n.kernel.clone(),
            }
        })
        .collect()
}

fn analyze_inner(
    state: &MinionsState,
    budget: SolverBudget,
    opts: playground_core::AnalysisOptions,
) -> anyhow::Result<playground_core::Analysis> {
    let playground_core::AnalysisOptions {
        candidates,
        eval_only,
        kernel_sigma,
        fast,
    } = opts;
    let cfg = CONFIG
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no net configured"))?;

    let mut guard = MODEL.lock().unwrap();
    let model = ensure_model(&mut guard, &cfg)?;

    let aux = aux_for(state);

    if eval_only {
        // `k = 0` is the value-only expansion: trunk and value head, no
        // candidate generation. Walking a variation asks for this on every
        // click, and generating candidates there would cost the diffusion head
        // for nothing.
        use playground_core::sampled_mcts::SampledModel;
        let (v, _) = model.expand(state, 0)?;
        return Ok(playground_core::Analysis {
            value: v as f64,
            aux,
            nodes: Vec::new(),
            chosen: None,
        });
    }

    // One of self-play's two searches, under the engine's current settings
    // (see `search_config`), with the model called as the self-play pool
    // calls it for that kind. `chosen` is the turn self-play would play here,
    // temperature included. A request may still override the budget, the
    // candidate count and the kernel.
    let full = !fast;
    let base = cfg
        .actor
        .search_params(&cfg.sampler, full, search_seed(&cfg));
    let mut params = playground_core::sampled_mcts::SampledParams {
        k_candidates: candidates.unwrap_or(base.k_candidates),
        // Above 0 the kernel is live and every node exports its matrix.
        kernel_sigma: kernel_sigma.unwrap_or(base.kernel_sigma),
        ..base
    };
    if let Some(v) = budget.max_visits {
        params.simulations = v;
    }

    let (result, tree) = playground_core::sampled_mcts::search_analyzed::<MinionsTurns, _>(
        &crate::sampler::AsSearch { model, full },
        state.clone(),
        &params,
        Minions::ply(state),
    )?;

    let nodes = analysis_nodes(state, &tree);

    Ok(playground_core::Analysis {
        value: result.value,
        aux,
        nodes,
        chosen: Some(result.chosen),
    })
}
