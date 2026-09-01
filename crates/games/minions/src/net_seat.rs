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

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use playground_core::{sampled_mcts::search, Game, SearchOutcome, SolverBudget};

use crate::game::{Minions, MinionsMove, MinionsState};
use crate::sampler::{MinionsModel, SamplerConfig};
use crate::turn::{realize, MinionsTurns, TurnSpec};

/// Where the weights live, and how to run them. Set once at startup by `serve`.
static CONFIG: Mutex<Option<NetSeatConfig>> = Mutex::new(None);
/// The model, built on first use so `serve` starts instantly even without a GPU.
static MODEL: Mutex<Option<Box<MinionsModel>>> = Mutex::new(None);
/// The turn currently being handed out, one action at a time.
static PLAN: Mutex<Option<Plan>> = Mutex::new(None);

#[derive(Clone)]
pub struct NetSeatConfig {
    pub dir: PathBuf,
    pub kind: crate::net::BackendKind,
    pub sims: u32,
    pub sampler: SamplerConfig,
}

struct Plan {
    /// The state the next queued action applies to. Encoded rather than cloned
    /// so the comparison is exactly the engine's own notion of state identity.
    expect: Vec<u8>,
    actions: VecDeque<MinionsMove>,
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
    *CONFIG.lock().unwrap() = Some(cfg);
    *MODEL.lock().unwrap() = None;
    *PLAN.lock().unwrap() = None;
}

pub fn is_configured() -> bool {
    CONFIG.lock().unwrap().is_some()
}

/// Plan a whole turn from `state`, then return its first action.
fn plan_turn(state: &MinionsState, budget: &SolverBudget) -> anyhow::Result<SearchOutcome<MinionsMove>> {
    let cfg = CONFIG.lock().unwrap().clone().ok_or_else(|| anyhow::anyhow!("no net configured"))?;

    let mut guard = MODEL.lock().unwrap();
    if guard.is_none() {
        let seat = crate::eval::load_seat(&cfg.dir, &cfg.kind, cfg.sampler.clone())?;
        match seat {
            crate::eval::Seat::Net(m) => *guard = Some(m),
            crate::eval::Seat::Greedy => anyhow::bail!("load_seat returned the greedy seat"),
        }
    }
    let model = guard.as_ref().expect("model built above");

    let mut params = playground_core::sampled_mcts::SampledParams {
        k_candidates: cfg.sampler.k_candidates,
        // Play the argmax, never a temperature sample.
        //
        // `temp_moves` defaults to 8, so in self-play the first eight plies pick
        // a visit-proportional candidate to explore. Against a human that is
        // just weak play — and because forced exploration puts a *null* turn in
        // the candidate set, sampling could hand back a turn that does nothing
        // at all. That is exactly what "the net passes on turn 1" was.
        temp_moves: 0,
        ..Default::default()
    };
    // A human is waiting, so honour whatever budget the UI asked for.
    params.simulations = budget.max_visits.unwrap_or(cfg.sims);

    let result = search::<MinionsTurns, _>(&**model, state.clone(), &params, Minions::ply(state))?;
    // Belt and braces: with `temp_moves: 0` the search already returns the
    // most-visited candidate, but pick it explicitly so a future default cannot
    // quietly reintroduce sampled play here. Falling back to `TurnSpec::default()`
    // would be a null turn, so an empty candidate set is an error instead.
    let spec: TurnSpec = result
        .candidates
        .iter()
        .max_by_key(|(_, visits)| *visits)
        .map(|(s, _)| *s)
        .ok_or_else(|| anyhow::anyhow!("search returned no candidates"))?;

    let realized = realize(state, &spec);
    let mut actions: VecDeque<MinionsMove> = realized.actions.into_iter().collect();
    let first = actions
        .pop_front()
        .ok_or_else(|| anyhow::anyhow!("the realized turn was empty"))?;

    // Remember the state the *next* action applies to.
    let next_state = Minions::apply(state, &first);
    *PLAN.lock().unwrap() = Some(Plan { expect: Minions::encode_state(&next_state), actions });

    Ok(SearchOutcome { best_move: first, value: result.value, nodes: params.simulations as u64 })
}

/// One atomic move from the net, replaying the current turn's plan when it
/// still applies and re-planning when it does not.
pub fn choose(state: &MinionsState, budget: SolverBudget) -> Option<anyhow::Result<SearchOutcome<MinionsMove>>> {
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
                    return Some(Ok(SearchOutcome { best_move: mv, value: 0.0, nodes: 0 }));
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

    /// Is the value head emitting NaN, or is the NaN made during backup?
    #[test]
    fn value_head_is_finite() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else { return };
        let d = std::path::Path::new(&dir);
        let cache = d.join("trt_cache").to_string_lossy().into_owned();
        let ort = match crate::ort_net::OrtNet::new(d, "cuda", false, &cache) {
            Ok(o) => o,
            Err(e) => { eprintln!("ORT unavailable ({e})"); return; }
        };
        use crate::net::Backend;
        let mut st = Minions::initial_state();
        for step in 0..4 {
            let planes: Vec<f32> =
                Minions::to_features(&st).iter().map(|&b| b as f32).collect();
            let out = ort.trunk(&planes, 1).unwrap();
            let nan_h = out.h.iter().filter(|x| !x.is_finite()).count();
            println!(
                "step {step}: value={:?} finite={} | trunk non-finite {} of {}",
                out.value, out.value.iter().all(|v| v.is_finite()), nan_h, out.h.len()
            );
            let legal = Minions::legal_moves(&st);
            if legal.is_empty() { break; }
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
            kind: crate::net::BackendKind::Ort { ep: "trt".into(), fp16: true },
            sims: 64,
            sampler: SamplerConfig { k_candidates: 12, t_steps: 4, ..Default::default() },
        });
        let budget = SolverBudget { max_millis: None, max_visits: None };

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
        configure(NetSeatConfig {
            dir: dir.into(),
            kind: crate::net::BackendKind::Ort { ep: "trt".into(), fp16: true },
            sims: 64,
            sampler: SamplerConfig { k_candidates: 12, t_steps: 4, ..Default::default() },
        });
        let budget = SolverBudget { max_millis: None, max_visits: None };

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
            kind: crate::net::BackendKind::Ort { ep: "cuda".into(), fp16: false },
            sims: 16,
            sampler: SamplerConfig { k_candidates: 8, t_steps: 2, ..Default::default() },
        });

        let mut state = Minions::initial_state();
        let budget = SolverBudget { max_millis: None, max_visits: None };
        let mut acted = 0;
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
            state = Minions::apply(&state, &out.best_move);
            acted += 1;
        }
        assert!(acted > 1, "expected the seat to play several actions, got {acted}");

        // A state the plan does not expect must force a re-plan, not a stale replay.
        let fresh = Minions::initial_state();
        let out = Minions::choose_move_external(&fresh, budget).expect("configured").expect("ok");
        assert!(
            Minions::legal_moves(&fresh).contains(&out.best_move),
            "after a desync the seat must re-plan and return a move legal *here*"
        );
    }
}
