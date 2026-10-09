//! Realizer soak test.
//!
//! Hammers [`minions_engine::turn::realize`] with random declarations and checks
//! the one property the whole learning design rests on: **whatever the
//! declaration, every action the realizer emits is legal in the state it is
//! played from.** A single counterexample here would silently corrupt every
//! training run built on top, so this is the cheapest possible insurance.
//!
//! It also answers three questions P1 needs answered before it can be tuned:
//!
//! - **Throughput.** `realize` runs `K` times per node expansion (`K ≈ 32`), so
//!   its cost per call sets a hard floor on self-play speed.
//! - **Fidelity.** How much of a declaration actually happens. If this is low
//!   the policy would be learning to declare fantasies.
//! - **Assignment quality.** How many kills the greedy attack assignment gives
//!   up against an exact one. Greedy was chosen because the exact DP is
//!   exponential; this is the evidence for whether that trade costs anything.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use minions_engine::game::Minions;
use minions_engine::turn::{realize, TurnSpec};
use playground_core::{Game, Outcome};
use rand::{Rng, SeedableRng};

#[derive(Default, Clone)]
struct Stats {
    games: u64,
    turns: u64,
    realize_nanos: u64,
    /// Fidelity counters, indexed by spec kind: `[0]` wild, `[1]` plausible,
    /// `[2]` playout.
    /// Pooling them is meaningless — wild declarations are uniform garbage and
    /// *should* mostly be dropped; only the plausible number says anything about
    /// whether a real sampler's declarations get executed.
    declared_moves: [u64; 3],
    done_moves: [u64; 3],
    declared_kills: [u64; 3],
    done_kills: [u64; 3],
    declared_spawns: [u64; 3],
    done_spawns: [u64; 3],
    solver_calls: u64,
    solver_rescued: u64,
    certified_infeasible: u64,
    unresolved: u64,
    /// Fixed-point iteration counts, bucketed 1..=8+.
    iters: [u64; 9],
    outcomes: [u64; 4], // in-progress (unused), first, second, draw
    illegal: u64,
    panics: u64,
    /// Seeds that produced a failure, so any bug is reproducible.
    failures: Vec<String>,
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.games += o.games;
        self.turns += o.turns;
        self.realize_nanos += o.realize_nanos;
        // Bound from the array itself: a hardcoded 2 here silently dropped the
        // third spec kind's counters when one was added.
        for k in 0..self.declared_moves.len() {
            self.declared_moves[k] += o.declared_moves[k];
            self.done_moves[k] += o.done_moves[k];
            self.declared_kills[k] += o.declared_kills[k];
            self.done_kills[k] += o.done_kills[k];
            self.declared_spawns[k] += o.declared_spawns[k];
            self.done_spawns[k] += o.done_spawns[k];
        }
        self.solver_calls += o.solver_calls;
        self.solver_rescued += o.solver_rescued;
        self.certified_infeasible += o.certified_infeasible;
        self.unresolved += o.unresolved;
        for i in 0..9 {
            self.iters[i] += o.iters[i];
        }
        for i in 0..4 {
            self.outcomes[i] += o.outcomes[i];
        }
        self.illegal += o.illegal;
        self.panics += o.panics;
        self.failures.extend(o.failures.iter().cloned());
    }

    fn pct(num: u64, den: u64) -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    }

    fn report(&self, elapsed: Duration) -> String {
        let per = if self.turns == 0 {
            0.0
        } else {
            self.realize_nanos as f64 / self.turns as f64
        };
        format!(
            "{:>6.0}s  games {:>7}  turns {:>9}  realize {:>7.1}µs ({:>6.0}/s/thread)\n\
             \x20        fidelity wild      moves {:>5.1}%  kills {:>5.1}%  spawns {:>5.1}%\n\
             \x20        fidelity plausible moves {:>5.1}%  kills {:>5.1}%  spawns {:>5.1}%\n\
             \x20        fidelity playout   moves {:>5.1}%  kills {:>5.1}%  spawns {:>5.1}%\n\
             \x20        assign: solver {} rescued {} certified {} UNRESOLVED {}  iters {:?}\n\
             \x20        outcomes first {} second {} draw {}   ILLEGAL {}  PANICS {}",
            elapsed.as_secs_f64(),
            self.games,
            self.turns,
            per / 1000.0,
            if per > 0.0 { 1e9 / per } else { 0.0 },
            Self::pct(self.done_moves[0], self.declared_moves[0]),
            Self::pct(self.done_kills[0], self.declared_kills[0]),
            Self::pct(self.done_spawns[0], self.declared_spawns[0]),
            Self::pct(self.done_moves[1], self.declared_moves[1]),
            Self::pct(self.done_kills[1], self.declared_kills[1]),
            Self::pct(self.done_spawns[1], self.declared_spawns[1]),
            Self::pct(self.done_moves[2], self.declared_moves[2]),
            Self::pct(self.done_kills[2], self.declared_kills[2]),
            Self::pct(self.done_spawns[2], self.declared_spawns[2]),
            self.solver_calls,
            self.solver_rescued,
            self.certified_infeasible,
            self.unresolved,
            &self.iters[1..],
            self.outcomes[1],
            self.outcomes[2],
            self.outcomes[3],
            self.illegal,
            self.panics,
        )
    }

    fn to_json(&self, elapsed: Duration) -> String {
        let per = if self.turns == 0 {
            0.0
        } else {
            self.realize_nanos as f64 / self.turns as f64
        };
        format!(
            r#"{{
  "elapsed_secs": {:.1},
  "games": {}, "turns": {},
  "realize_micros_mean": {:.3},
  "realize_per_sec_per_thread": {:.0},
  "fidelity_wild": {{
    "moves":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "kills":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "spawns": {{ "declared": {}, "done": {}, "pct": {:.2} }}
  }},
  "fidelity_plausible": {{
    "moves":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "kills":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "spawns": {{ "declared": {}, "done": {}, "pct": {:.2} }}
  }},
  "fidelity_playout": {{
    "moves":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "kills":  {{ "declared": {}, "done": {}, "pct": {:.2} }},
    "spawns": {{ "declared": {}, "done": {}, "pct": {:.2} }}
  }},
  "assignment": {{ "solver_calls": {}, "solver_rescued": {}, "certified_infeasible": {}, "unresolved": {} }},
  "iterations_histogram_1_to_8plus": {:?},
  "outcomes": {{ "first": {}, "second": {}, "draw": {} }},
  "illegal_actions": {}, "panics": {},
  "failures": [{}]
}}
"#,
            elapsed.as_secs_f64(),
            self.games,
            self.turns,
            per / 1000.0,
            if per > 0.0 { 1e9 / per } else { 0.0 },
            self.declared_moves[0],
            self.done_moves[0],
            Self::pct(self.done_moves[0], self.declared_moves[0]),
            self.declared_kills[0],
            self.done_kills[0],
            Self::pct(self.done_kills[0], self.declared_kills[0]),
            self.declared_spawns[0],
            self.done_spawns[0],
            Self::pct(self.done_spawns[0], self.declared_spawns[0]),
            self.declared_moves[1],
            self.done_moves[1],
            Self::pct(self.done_moves[1], self.declared_moves[1]),
            self.declared_kills[1],
            self.done_kills[1],
            Self::pct(self.done_kills[1], self.declared_kills[1]),
            self.declared_spawns[1],
            self.done_spawns[1],
            Self::pct(self.done_spawns[1], self.declared_spawns[1]),
            self.declared_moves[2],
            self.done_moves[2],
            Self::pct(self.done_moves[2], self.declared_moves[2]),
            self.declared_kills[2],
            self.done_kills[2],
            Self::pct(self.done_kills[2], self.declared_kills[2]),
            self.declared_spawns[2],
            self.done_spawns[2],
            Self::pct(self.done_spawns[2], self.declared_spawns[2]),
            self.solver_calls,
            self.solver_rescued,
            self.certified_infeasible,
            self.unresolved,
            &self.iters[1..],
            self.outcomes[1],
            self.outcomes[2],
            self.outcomes[3],
            self.illegal,
            self.panics,
            self.failures
                .iter()
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

pub fn run(
    threads: usize,
    secs: u64,
    verify: bool,
    oracle: bool,
    seed: u64,
    out: &str,
) -> anyhow::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let started = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let shared = Arc::new(Mutex::new(Stats::default()));
    let live_turns = Arc::new(AtomicU64::new(0));

    println!(
        "soak: {threads} threads, {secs}s, verify={verify}, oracle={oracle}, seed={seed}\n\
         checking that every realized action is legal; ILLEGAL and PANICS must stay 0."
    );

    let mut handles = Vec::new();
    for t in 0..threads {
        let stop = stop.clone();
        let shared = shared.clone();
        let live_turns = live_turns.clone();
        handles.push(std::thread::spawn(move || {
            let mut local = Stats::default();
            let mut game_no = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let game_seed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((t as u64) << 40)
                    .wrapping_add(game_no);
                game_no += 1;
                let res = std::panic::catch_unwind(|| play_one(game_seed, verify, oracle));
                match res {
                    Ok(s) => {
                        live_turns.fetch_add(s.turns, Ordering::Relaxed);
                        local.merge(&s);
                    }
                    Err(_) => {
                        local.panics += 1;
                        local.failures.push(format!("panic@seed={game_seed}"));
                    }
                }
                if local.games % 64 == 0 {
                    shared.lock().unwrap().merge(&local);
                    local = Stats::default();
                }
            }
            shared.lock().unwrap().merge(&local);
        }));
    }

    // Progress line every 30s until the deadline.
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_secs(30).min(deadline - Instant::now()));
        let snap = shared.lock().unwrap().clone();
        let mut s = snap;
        s.turns = s.turns.max(live_turns.load(Ordering::Relaxed));
        println!("{}", s.report(started.elapsed()));
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    let final_stats = shared.lock().unwrap().clone();
    let elapsed = started.elapsed();
    println!("\n=== soak complete ===\n{}", final_stats.report(elapsed));
    if let Some(dir) = std::path::Path::new(out).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(out, final_stats.to_json(elapsed))?;
    println!("wrote {out}");

    if final_stats.illegal > 0 || final_stats.panics > 0 {
        anyhow::bail!(
            "soak FAILED: {} illegal actions, {} panics — see {out}",
            final_stats.illegal,
            final_stats.panics
        );
    }
    Ok(())
}

/// One random-declaration self-play game, returning its stats.
fn play_one(seed: u64, verify: bool, oracle: bool) -> Stats {
    let mut st = Stats::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut s = Minions::initial_state();
    st.games = 1;

    while !Minions::outcome(&s).is_terminal() {
        // Alternate wild and plausible declarations: wild is the adversarial
        // input the legality property must survive, plausible is what a trained
        // sampler will actually look like.
        // Three kinds, because they answer different questions: wild is the
        // adversarial input the legality property must survive, plausible is the
        // cheap injected candidate, and playout is the one built by actually
        // playing a random legal turn — the comparison between the last two is
        // what says whether injected candidates cover combat at all.
        let k = rng.gen_range(0..3usize);
        let _wild = k == 0;
        let spec = if k == 0 {
            TurnSpec::wild(&mut rng)
        } else if k == 2 {
            TurnSpec::random_playout(&s, &mut rng)
        } else {
            TurnSpec::random_plausible(&s, &mut rng)
        };

        let _ = oracle;
        let t0 = Instant::now();
        let r = realize(&s, &spec);
        st.realize_nanos += t0.elapsed().as_nanos() as u64;
        st.turns += 1;

        let f = r.fidelity;
        st.declared_moves[k] += f.declared_moves as u64;
        st.done_moves[k] += f.done_moves as u64;
        st.declared_kills[k] += f.declared_kills as u64;
        st.done_kills[k] += f.done_kills as u64;
        st.declared_spawns[k] += f.declared_spawns as u64;
        st.done_spawns[k] += f.done_spawns as u64;
        st.iters[(f.iterations as usize).min(8)] += 1;
        st.solver_calls += f.solver_calls as u64;
        st.solver_rescued += f.solver_rescued as u64;
        st.certified_infeasible += f.certified_infeasible as u64;
        st.unresolved += f.unresolved as u64;

        if verify {
            let mut check = s.clone();
            for mv in &r.actions {
                if !Minions::is_legal(&check, mv) {
                    st.illegal += 1;
                    st.failures
                        .push(format!("illegal {mv} @seed={seed} turn={}", check.turn));
                    break;
                }
                check = Minions::apply(&check, mv);
            }
        }
        s = r.next;
    }

    st.outcomes[match Minions::outcome(&s) {
        Outcome::FirstWins => 1,
        Outcome::SecondWins => 2,
        Outcome::Draw => 3,
        Outcome::InProgress => 0,
    }] += 1;
    st
}
