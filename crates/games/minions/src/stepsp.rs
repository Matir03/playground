//! Self-play for the best-first step search: games are exploration, and the
//! training data is every search node descended into at least `min_visits`
//! times, labelled with its backed-up value.
//!
//! Record layout ([`REC_BYTES`] bytes, little endian), after a 16-byte chunk
//! header `b"MSTP", version u32, count u32, rec_bytes u32`:
//!
//! | off | type | field |
//! |---|---|---|
//! | 0 | u8 | active count |
//! | 1 | u8 | stage (0 fates, 1 moves, 2 spawns) |
//! | 2 | u16 | descents (saturating) |
//! | 4 | f32 | target: backed-up value for the mover |
//! | 8 | f32 | static value at creation |
//! | 12 | u16 | turn |
//! | 14 | u16 | reserved |
//! | 16 | u16 × 96 | active sparse indices (`plane * 49 + cell`) |
//! | 208 | f32 × 16 | scalars |

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::SeedableRng;

use crate::bfs::{Evaluator, Params, SearchStats, Tree};
use crate::steps::{Stage, StepState, View, MAX_ACTIVE, N_SCALARS};

pub const REC_BYTES: usize = 272;
static DEAD_SHOWN: AtomicU64 = AtomicU64::new(0);
pub const VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct Config {
    pub workers: usize,
    pub params: Params,
    /// Leaf expansions per move.
    pub expansions: u64,
    /// Wall-clock cap per move, seconds (0 = none).
    pub move_secs: f64,
    pub min_visits: u32,
    pub temp: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub opening_max: u32,
    pub opening_decay: f32,
    /// Stop once this many records are written.
    pub target_records: u64,
    pub chunk_records: usize,
    pub seed: u64,
}

pub fn encode_record(v: &View, target: f32, prior: f32, visits: u32, out: &mut Vec<u8>) {
    let mut act = Vec::with_capacity(MAX_ACTIVE);
    let mut sc = [0f32; N_SCALARS];
    v.features(&mut act, &mut sc);
    let start = out.len();
    out.push(act.len().min(MAX_ACTIVE) as u8);
    out.push(match v.stage {
        Stage::Fates => 0,
        Stage::Moves => 1,
        Stage::Spawns => 2,
        Stage::Terminal => 3,
    });
    out.extend_from_slice(&(visits.min(u16::MAX as u32) as u16).to_le_bytes());
    out.extend_from_slice(&target.to_le_bytes());
    out.extend_from_slice(&prior.to_le_bytes());
    out.extend_from_slice(&v.board.turn.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    for i in 0..MAX_ACTIVE {
        let x = act.get(i).copied().unwrap_or(0);
        out.extend_from_slice(&x.to_le_bytes());
    }
    for x in sc {
        out.extend_from_slice(&x.to_le_bytes());
    }
    debug_assert_eq!(out.len() - start, REC_BYTES);
}

struct Writer {
    dir: PathBuf,
    tag: String,
    seq: u32,
    buf: Vec<u8>,
    n: usize,
    per_chunk: usize,
}

impl Writer {
    fn push(&mut self, v: &View, target: f32, prior: f32, visits: u32) -> anyhow::Result<()> {
        encode_record(v, target, prior, visits, &mut self.buf);
        self.n += 1;
        if self.n >= self.per_chunk {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        if self.n == 0 {
            return Ok(());
        }
        let name = format!("chunk_{}_{:05}.bin", self.tag, self.seq);
        let tmp = self.dir.join(format!(".{name}.tmp"));
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(b"MSTP")?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.n as u32).to_le_bytes())?;
        f.write_all(&(REC_BYTES as u32).to_le_bytes())?;
        f.write_all(&self.buf)?;
        drop(f);
        std::fs::rename(&tmp, self.dir.join(&name))?;
        self.seq += 1;
        self.buf.clear();
        self.n = 0;
        Ok(())
    }
}

#[derive(Default)]
pub struct Counters {
    pub games: AtomicU64,
    pub moves: AtomicU64,
    pub records: AtomicU64,
    pub expansions: AtomicU64,
    pub evals: AtomicU64,
    pub dead: AtomicU64,
    pub nodes_at_move: AtomicU64,
    pub search_ns: AtomicU64,
    pub eval_ns: AtomicU64,
    pub expand_ns: AtomicU64,
    pub select_ns: AtomicU64,
    pub backup_ns: AtomicU64,
    pub decisive: AtomicU64,
    pub turns: AtomicU64,
    pub stuck: AtomicU64,
    pub starved: AtomicU64,
    pub walk_ns: AtomicU64,
}

impl Counters {
    fn add_stats(&self, s: &SearchStats) {
        self.expansions.fetch_add(s.expansions, Ordering::Relaxed);
        self.evals.fetch_add(s.evals, Ordering::Relaxed);
        self.dead.fetch_add(s.dead, Ordering::Relaxed);
        self.eval_ns.fetch_add(s.eval_ns, Ordering::Relaxed);
        self.expand_ns.fetch_add(s.expand_ns, Ordering::Relaxed);
        self.select_ns.fetch_add(s.select_ns, Ordering::Relaxed);
        self.backup_ns.fetch_add(s.backup_ns, Ordering::Relaxed);
    }
}

/// Record every node of `tree` outside the subtree of `keep` (all nodes when
/// `keep` is `None`) with enough descents.
fn record_nodes(
    tree: &Tree,
    keep: Option<u32>,
    min_visits: u32,
    w: &mut Writer,
) -> anyhow::Result<u64> {
    let mut inside = vec![false; tree.nodes.len()];
    if let Some(k) = keep {
        // Children are always created after their parent, so one forward pass
        // marks the whole subtree.
        inside[k as usize] = true;
        for i in (k as usize + 1)..tree.nodes.len() {
            let p = tree.nodes[i].parent;
            if p != u32::MAX && inside[p as usize] {
                inside[i] = true;
            }
        }
    }
    let mut n = 0;
    for id in tree.trained_nodes(min_visits) {
        if inside[id as usize] {
            continue;
        }
        let nd = &tree.nodes[id as usize];
        let v = tree.state_of(id).view();
        w.push(&v, nd.value, nd.prior, nd.visits)?;
        n += 1;
    }
    Ok(n)
}

fn play_game(
    cfg: &Config,
    eval: &dyn Evaluator,
    rng: &mut rand_chacha::ChaCha8Rng,
    w: &mut Writer,
    c: &Counters,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    use rand::Rng;
    let (start, _) = crate::selfplay::random_opening(cfg.opening_max, cfg.opening_decay, rng);
    let salt: u64 = rng.gen();
    let mut tree = Tree::new(StepState::begin(&start, salt), cfg.params.clone(), eval)?;
    let mut turns = 0u64;
    while !tree.root.is_terminal() && !tree.is_solved() {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let t = Instant::now();
        let before = tree.stats.expansions;
        let mut starved = 0;
        while tree.stats.expansions - before < cfg.expansions {
            if cfg.move_secs > 0.0 && t.elapsed().as_secs_f64() > cfg.move_secs {
                break;
            }
            let got = tree.iterate(eval)?;
            if got == 0 {
                break;
            }
            // The PV runs into decided positions (terminal lines) and the
            // candidate pool dries up: stop rather than trickle tiny batches.
            if got * 4 < cfg.params.heads {
                starved += 1;
                if starved >= 8 {
                    c.starved.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            } else {
                starved = 0;
            }
        }
        c.search_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if t.elapsed().as_secs_f64() > 10.0 {
            let st = &tree.stats;
            eprintln!(
                "slow move {:.1}s: iters {} expansions {} evals {} dead {} terminal-leaves {} nodes {} | select {:.1}s expand {:.1}s eval {:.1}s backup {:.1}s | turn {} units {}",
                t.elapsed().as_secs_f64(), st.iterations, st.expansions, st.evals, st.dead, st.terminal_leaves, tree.nodes.len(),
                st.select_ns as f64 * 1e-9, st.expand_ns as f64 * 1e-9, st.eval_ns as f64 * 1e-9, st.backup_ns as f64 * 1e-9,
                tree.root.base.turn, tree.root.friends.len() + tree.root.enemies.len(),
            );
        }
        c.nodes_at_move
            .fetch_add(tree.nodes.len() as u64, Ordering::Relaxed);
        // Play one whole turn: steps until the move passes or the game ends.
        // A pick that lands on a dead end is backed out of: expanding it marks
        // it dead, and the walk resumes from its parent.
        let tp = Instant::now();
        let (mut walk_exp, mut walk_steps) = (0u32, 0u32);
        let mut n = 0u32;
        let next = loop {
            walk_steps += 1;
            if tree.kind(n) == 0 {
                walk_exp += 1;
            }
            tree.ensure_expanded(n, eval)?;
            if tree.kind(n) == 3
                && std::env::var("STEPS_DEBUG_DEAD").is_ok()
                && DEAD_SHOWN.fetch_add(1, Ordering::Relaxed) < 6
            {
                eprintln!("dead in walk: {}", tree.state_of(n).debug_dead());
            }
            if tree.kind(n) == 3 {
                if n == 0 {
                    break None;
                }
                n = tree.nodes[n as usize].parent;
                continue;
            }
            let Some(ch) =
                tree.pick_child(n, cfg.temp, (cfg.dirichlet_alpha, cfg.dirichlet_eps), rng)
            else {
                break None;
            };
            if tree.nodes[ch as usize].flip || tree.kind(ch) == 2 {
                break Some(ch);
            }
            n = ch;
        };
        if next.is_none() {
            c.stuck.fetch_add(1, Ordering::Relaxed);
        }
        c.walk_ns
            .fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if tp.elapsed().as_secs_f64() > 5.0 {
            eprintln!(
                "slow walk {:.1}s: {walk_steps} steps, {walk_exp} single expansions, turn {}",
                tp.elapsed().as_secs_f64(),
                tree.root.base.turn
            );
        }
        let Some(next) = next else { break };
        let rec = record_nodes(&tree, Some(next), cfg.min_visits, w)?;
        c.records.fetch_add(rec, Ordering::Relaxed);
        c.add_stats(&tree.stats);
        tree = tree.reroot(next);
        c.moves.fetch_add(1, Ordering::Relaxed);
        turns += 1;
    }
    let rec = record_nodes(&tree, None, cfg.min_visits, w)?;
    c.records.fetch_add(rec, Ordering::Relaxed);
    c.add_stats(&tree.stats);
    c.games.fetch_add(1, Ordering::Relaxed);
    c.turns.fetch_add(turns, Ordering::Relaxed);
    if tree.root.is_terminal() && tree.root.base.winner != 3 {
        c.decisive.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Run `cfg.workers` self-play threads until `cfg.target_records` are written.
pub fn run(cfg: Config, eval: Arc<dyn Evaluator>, out: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let c = Arc::new(Counters::default());
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for wi in 0..cfg.workers {
        let (cfg, eval, c, stop, out) = (
            cfg.clone(),
            eval.clone(),
            c.clone(),
            stop.clone(),
            out.to_path_buf(),
        );
        handles.push(std::thread::spawn(move || -> anyhow::Result<()> {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                cfg.seed ^ (wi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut w = Writer {
                dir: out,
                tag: format!("s{}w{}", cfg.seed, wi),
                seq: 0,
                buf: Vec::new(),
                n: 0,
                per_chunk: cfg.chunk_records,
            };
            while !stop.load(Ordering::Relaxed) {
                play_game(&cfg, eval.as_ref(), &mut rng, &mut w, &c, &stop)?;
                if c.records.load(Ordering::Relaxed) >= cfg.target_records {
                    stop.store(true, Ordering::Relaxed);
                }
            }
            w.flush()
        }));
    }
    let t0 = Instant::now();
    let mut last = (Instant::now(), 0u64, 0u64, 0u64, 0u64);
    while !handles.iter().all(|h| h.is_finished()) {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if last.0.elapsed().as_secs() >= 60 {
            let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
            let dt = last.0.elapsed().as_secs_f64();
            let (e, v, r, m) = (g(&c.expansions), g(&c.evals), g(&c.records), g(&c.moves));
            let tot =
                (g(&c.select_ns) + g(&c.expand_ns) + g(&c.eval_ns) + g(&c.backup_ns)).max(1) as f64;
            println!(
                "[{:.0}s] games {} moves {} ({:.2}/s) | expansions {:.0}/s evals {:.0}/s | records {} ({:.0}/s) | dead {} | nodes/move {:.0} | select {:.0}% expand {:.0}% eval {:.0}% backup {:.0}% | decisive {} stuck {} starved {} avg turns {:.1} | search {:.0}s walk {:.0}s",
                t0.elapsed().as_secs_f64(),
                g(&c.games),
                m,
                (m - last.4) as f64 / dt,
                (e - last.1) as f64 / dt,
                (v - last.2) as f64 / dt,
                r,
                (r - last.3) as f64 / dt,
                g(&c.dead),
                g(&c.nodes_at_move) as f64 / m.max(1) as f64,
                100.0 * g(&c.select_ns) as f64 / tot,
                100.0 * g(&c.expand_ns) as f64 / tot,
                100.0 * g(&c.eval_ns) as f64 / tot,
                100.0 * g(&c.backup_ns) as f64 / tot,
                g(&c.decisive),
                g(&c.stuck),
                g(&c.starved),
                g(&c.turns) as f64 / g(&c.games).max(1) as f64,
                g(&c.search_ns) as f64 * 1e-9,
                g(&c.walk_ns) as f64 * 1e-9,
            );
            last = (Instant::now(), e, v, r, m);
        }
    }
    for h in handles {
        h.join().map_err(|_| anyhow::anyhow!("worker panicked"))??;
    }
    println!(
        "done: {} records, {} games",
        c.records.load(Ordering::Relaxed),
        c.games.load(Ordering::Relaxed)
    );
    Ok(())
}

/// Play one turn greedily: at every step, evaluate all options with `eval`
/// and take the best for the mover. Returns the position after the turn.
pub fn greedy_turn(
    real: &crate::game::MinionsState,
    eval: &dyn Evaluator,
    salt: u64,
) -> anyhow::Result<crate::game::MinionsState> {
    let mut s = StepState::begin(real, salt);
    let f0 = s.flipped;
    while !s.is_terminal() && s.flipped == f0 {
        let opts = s.options();
        if opts.is_empty() {
            anyhow::bail!("dead end in a greedy turn");
        }
        if opts.len() == 1 {
            s.apply(opts[0]);
            continue;
        }
        let kids: Vec<StepState> = opts
            .iter()
            .map(|&v| {
                let mut c = s.clone();
                c.apply(v);
                c
            })
            .collect();
        let live: Vec<usize> = (0..kids.len())
            .filter(|&i| !kids[i].is_terminal())
            .collect();
        let views: Vec<View> = live.iter().map(|&i| kids[i].view()).collect();
        let vals = if views.is_empty() {
            Vec::new()
        } else {
            eval.eval(&views)?
        };
        let mut q = vec![0f32; kids.len()];
        for (j, &i) in live.iter().enumerate() {
            q[i] = vals[j];
        }
        for (i, k) in kids.iter().enumerate() {
            if k.is_terminal() {
                q[i] = k.terminal_value();
            }
            if k.flipped != f0 {
                q[i] = 1.0 - q[i]; // the child is the opponent's to move
            }
        }
        let best = (0..kids.len())
            .max_by(|&a, &b| q[a].total_cmp(&q[b]))
            .expect("options");
        s = kids.into_iter().nth(best).expect("best");
    }
    Ok(s.real_base())
}

/// A side's bid: the handicap (money to the second player, `BID_MIN..=BID_MAX`)
/// at which its own evaluator rates the opening even for the first player,
/// found by binary search on the (decreasing) first-player value.
pub fn bid(eval: &dyn Evaluator, salt: u64) -> anyhow::Result<i16> {
    use crate::game::{BID_MAX, BID_MIN};
    let v = |b: i16| -> anyhow::Result<f32> {
        let s = StepState::begin(&crate::game::MinionsState::with_bid(b), salt);
        Ok(eval.eval(&[s.view()])?[0])
    };
    let (mut lo, mut hi) = (BID_MIN, BID_MAX);
    if v(lo)? < 0.5 {
        return Ok(lo);
    }
    if v(hi)? >= 0.5 {
        return Ok(hi);
    }
    // Invariant: v(lo) >= 0.5 > v(hi).
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if v(mid)? >= 0.5 {
            lo = mid
        } else {
            hi = mid
        }
    }
    Ok(lo)
}

/// Greedy match between two evaluators. Game pairs share a start (a random
/// opening, or the standard opening with a bid when `opening_max` is 0) with
/// seats swapped. Returns A's score and the per-game results.
pub fn greedy_match(
    a: &dyn Evaluator,
    b: &dyn Evaluator,
    pairs: usize,
    opening_max: u32,
    opening_decay: f32,
    seed: u64,
) -> anyhow::Result<(f64, [u32; 3])> {
    use playground_core::{Game, Outcome};
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut wdl = [0u32; 3]; // A wins, B wins, draws
    let mut score = 0f64;
    let mut first_wins = 0u32;
    let mut split_pairs = 0u32;
    for _ in 0..pairs {
        let start = if opening_max == 0 {
            None
        } else {
            Some(crate::selfplay::random_opening(opening_max, opening_decay, &mut rng).0)
        };
        let salt: u64 = rng.gen();
        let mut pair_a = 0u32;
        for a_first in [true, false] {
            let mut real = match &start {
                Some(s) => s.clone(),
                None => {
                    let (ba, bb) = (bid(a, salt)?, bid(b, salt)?);
                    // The two bids' average (rounded toward zero) is the handicap.
                    crate::game::MinionsState::with_bid((ba + bb) / 2)
                }
            };
            let mut turns = 0;
            while !crate::game::Minions::outcome(&real).is_terminal() && turns < 200 {
                let a_moves = (real.side == 0) == a_first;
                real = greedy_turn(&real, if a_moves { a } else { b }, salt ^ turns as u64)?;
                turns += 1;
            }
            let res = crate::game::Minions::outcome(&real);
            if res == Outcome::FirstWins {
                first_wins += 1;
            }
            let a_seat_first = a_first;
            let (aw, bw) = match res {
                Outcome::FirstWins => (a_seat_first, !a_seat_first),
                Outcome::SecondWins => (!a_seat_first, a_seat_first),
                _ => (false, false),
            };
            if aw {
                wdl[0] += 1;
                pair_a += 1;
                score += 1.0;
            } else if bw {
                wdl[1] += 1;
            } else {
                wdl[2] += 1;
                score += 0.5;
            }
        }
        if pair_a == 1 {
            split_pairs += 1;
        }
    }
    eprintln!(
        "first seat won {first_wins} of {} games; {split_pairs} of {pairs} pairs split 1-1 by seat",
        2 * pairs
    );
    Ok((score / (2 * pairs) as f64, wdl))
}
