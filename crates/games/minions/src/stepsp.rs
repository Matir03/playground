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
    /// Search with Descent ([`Tree::descent_wave`], `params.heads` descents in
    /// flight) instead of best-first.
    pub descent: bool,
    /// Walk each turn with [`Tree::pick_ordinal`] (ordinal distribution +
    /// completed action selection) instead of descents with temperature and
    /// Dirichlet noise.
    pub ordinal: bool,
    /// Keep each eligible node (and terminal) as a record with this
    /// probability: the trees produce far more records than the learner can
    /// take, so a uniform sample spreads a generation over more games.
    pub record_frac: f64,
    /// Sampling weight exponent on visits (see `record_nodes`).
    pub record_alpha: f64,
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
    // Game outcome for this record's mover, patched in at game end
    // (`Writer::finish_game`): see [`outcome_code`]. 0 = unknown.
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

/// Byte offset of the outcome field in a record.
pub const OUTCOME_AT: usize = 14;

/// The outcome field: bits 0-1 the result for the record's mover (1 loss,
/// 2 draw, 3 win; 0 unknown), bits 2-15 the turns from the record's turn to
/// the end of the game.
pub fn outcome_code(mover: u8, winner: u8, turn: u16, end_turn: u16) -> u16 {
    let r = match winner {
        3 => 2,
        w if w == mover + 1 => 3,
        1 | 2 => 1,
        _ => 0,
    };
    if r == 0 {
        return 0;
    }
    r | (end_turn.saturating_sub(turn).min(0x3fff) << 2)
}

struct Writer {
    dir: PathBuf,
    tag: String,
    seq: u32,
    buf: Vec<u8>,
    n: usize,
    per_chunk: usize,
    /// This game's records so far, held until its outcome is known, with
    /// each record's real mover (0 = first player).
    game: Vec<u8>,
    movers: Vec<u8>,
}

impl Writer {
    fn push(&mut self, v: &View, flipped: bool, target: f32, prior: f32, visits: u32) {
        encode_record(v, target, prior, visits, &mut self.game);
        self.movers.push(u8::from(flipped));
    }

    /// Stamp the held records with the game's result (`winner` as in
    /// `MinionsState`, 0 = unfinished) and queue them for writing.
    fn finish_game(&mut self, winner: u8, end_turn: u16) -> anyhow::Result<()> {
        let game = std::mem::take(&mut self.game);
        for (i, rec) in game.chunks_exact(REC_BYTES).enumerate() {
            let at = self.buf.len();
            self.buf.extend_from_slice(rec);
            let turn = u16::from_le_bytes([rec[12], rec[13]]);
            let code = outcome_code(self.movers[i], winner, turn, end_turn);
            self.buf[at + OUTCOME_AT..at + OUTCOME_AT + 2].copy_from_slice(&code.to_le_bytes());
            self.n += 1;
            if self.n >= self.per_chunk {
                self.flush()?;
            }
        }
        self.movers.clear();
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

/// Record the nodes of `tree` outside the subtree of `keep` (all nodes when
/// `keep` is `None`): every node with at least `min_visits` descents, and
/// every terminal node once per distinct position (many step orders reach
/// the same end). Each is kept with probability `frac * w / mean(w)` (capped
/// at 1), `w = max(visits, 1)^alpha`, so the expected count stays `frac` of
/// the eligible nodes while well-visited nodes are favoured (`alpha = 0` is
/// uniform).
fn record_nodes(
    tree: &Tree,
    keep: Option<u32>,
    min_visits: u32,
    frac: f64,
    alpha: f64,
    rng: &mut rand_chacha::ChaCha8Rng,
    w: &mut Writer,
) -> anyhow::Result<u64> {
    use rand::Rng;
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
    let mut cand: Vec<u32> = tree
        .trained_nodes(min_visits)
        .into_iter()
        .filter(|&id| !inside[id as usize])
        .collect();
    let mut seen = std::collections::HashSet::new();
    for id in tree.terminal_nodes() {
        if !inside[id as usize] && seen.insert(tree.state_of(id).base.key()) {
            cand.push(id);
        }
    }
    let weight = |id: u32| f64::from(tree.nodes[id as usize].visits.max(1)).powf(alpha);
    let mean = if cand.is_empty() {
        1.0
    } else {
        cand.iter().map(|&id| weight(id)).sum::<f64>() / cand.len() as f64
    };
    let mut n = 0;
    for id in cand {
        if frac < 1.0 && rng.gen::<f64>() >= frac * weight(id) / mean {
            continue;
        }
        let nd = &tree.nodes[id as usize];
        let st = tree.state_of(id);
        w.push(&st.view(), st.flipped, nd.value, nd.prior, nd.visits);
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
            return w.finish_game(0, 0);
        }
        let t = Instant::now();
        let before = tree.stats.expansions;
        let mut starved = 0;
        while tree.stats.expansions - before < cfg.expansions {
            if cfg.move_secs > 0.0 && t.elapsed().as_secs_f64() > cfg.move_secs {
                break;
            }
            let got = if cfg.descent {
                tree.descent_wave(eval, cfg.params.heads)?
            } else {
                tree.iterate(eval)?
            };
            if got == 0 {
                break;
            }
            if cfg.descent {
                continue;
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
        tree.clear_descents();
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
            let pick = if cfg.ordinal {
                tree.pick_ordinal(n, rng)
            } else {
                tree.pick_child(n, cfg.temp, (cfg.dirichlet_alpha, cfg.dirichlet_eps), rng)
            };
            let Some(ch) = pick else {
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
        let rec = record_nodes(
            &tree,
            Some(next),
            cfg.min_visits,
            cfg.record_frac,
            cfg.record_alpha,
            rng,
            w,
        )?;
        c.records.fetch_add(rec, Ordering::Relaxed);
        c.add_stats(&tree.stats);
        tree = tree.reroot(next);
        c.moves.fetch_add(1, Ordering::Relaxed);
        turns += 1;
    }
    let rec = record_nodes(
        &tree,
        None,
        cfg.min_visits,
        cfg.record_frac,
        cfg.record_alpha,
        rng,
        w,
    )?;
    let end = tree.root.real_base();
    w.finish_game(end.winner, end.turn)?;
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
    if std::env::var("STEPS_EXACT_DUP").is_ok() {
        let _ = crate::steps::EXACT_DUP.set(Default::default());
    }
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
                game: Vec::new(),
                movers: Vec::new(),
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
            let x = &crate::steps::EXACT_STATS;
            println!(
                "  exact search: found {} impossible {} out-of-budget {} | {:.0} thread-s",
                g(&x[0]),
                g(&x[1]),
                g(&x[2]),
                g(&x[3]) as f64 * 1e-9
            );
            let z = &crate::steps::EXACT_OUTCOME;
            println!(
                "  exact by outcome: found {:.0}s {:.0} nodes/call, impossible {:.0}s {:.0} nodes/call, out-of-budget {:.0}s | repeated queries {}",
                g(&z[0][0]) as f64 * 1e-9, g(&z[0][1]) as f64 / g(&x[0]).max(1) as f64,
                g(&z[1][0]) as f64 * 1e-9, g(&z[1][1]) as f64 / g(&x[1]).max(1) as f64,
                g(&z[2][0]) as f64 * 1e-9, g(&z[3][0])
            );
            let y = &crate::steps::EXACT_SITES;
            println!(
                "  exact by site: fates {} calls {:.0}s, moves {} calls {:.0}s, complete {} calls {:.0}s",
                g(&y[0][0]), g(&y[0][1]) as f64 * 1e-9,
                g(&y[1][0]), g(&y[1][1]) as f64 * 1e-9,
                g(&y[2][0]), g(&y[2][1]) as f64 * 1e-9
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
        let kids: Vec<StepState> = opts.iter().map(|&v| s.child(v)).collect();
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

/// A turn chosen by [`search_turn`].
pub struct SearchedTurn {
    /// The real position after the turn.
    pub next: crate::game::MinionsState,
    /// The turn's declaration, for [`crate::steps::turn_actions`].
    pub done: crate::turn::TurnSpec,
    /// The root's backed-up value for the mover.
    pub value: f32,
    pub expansions: u64,
}

/// One turn by search: a fresh tree searched for `expansions` leaf
/// expansions (or `secs` seconds if positive, or until the candidate pool
/// starves), then the principal variation played to the end of the turn.
pub fn search_turn(
    real: &crate::game::MinionsState,
    eval: &dyn Evaluator,
    params: &Params,
    expansions: u64,
    secs: f64,
    salt: u64,
) -> anyhow::Result<SearchedTurn> {
    let t = Instant::now();
    let mut tree = Tree::new(StepState::begin(real, salt), params.clone(), eval)?;
    let mut starved = 0;
    while !tree.is_solved() && tree.stats.expansions < expansions {
        if secs > 0.0 && t.elapsed().as_secs_f64() > secs {
            break;
        }
        let got = tree.iterate(eval)?;
        if got == 0 {
            break;
        }
        if got * 4 < params.heads {
            starved += 1;
            if starved >= 8 {
                break;
            }
        } else {
            starved = 0;
        }
    }
    let mut n = 0u32;
    loop {
        tree.ensure_expanded(n, eval)?;
        if tree.kind(n) == 3 {
            anyhow::ensure!(n != 0, "dead root in a search turn");
            n = tree.nodes[n as usize].parent;
            continue;
        }
        let Some(ch) = tree.best_child(n) else {
            anyhow::bail!("no live child in a search turn");
        };
        if tree.nodes[ch as usize].flip || tree.kind(ch) == 2 {
            let fin = tree.state_of(ch);
            return Ok(SearchedTurn {
                next: fin.real_base(),
                done: fin.done,
                value: tree.nodes[0].value,
                expansions: tree.stats.expansions,
            });
        }
        n = ch;
    }
}

/// How one side plays its turns in a match.
#[derive(Clone)]
pub struct Player {
    pub eval: Arc<dyn Evaluator>,
    /// 0 = greedy (static values only); otherwise [`search_turn`] with this
    /// many expansions per turn.
    pub expansions: u64,
    pub params: Params,
}

impl Player {
    fn turn(
        &self,
        real: &crate::game::MinionsState,
        salt: u64,
    ) -> anyhow::Result<crate::game::MinionsState> {
        if self.expansions == 0 {
            greedy_turn(real, self.eval.as_ref(), salt)
        } else {
            search_turn(
                real,
                self.eval.as_ref(),
                &self.params,
                self.expansions,
                0.0,
                salt,
            )
            .map(|t| t.next)
        }
    }
}

/// Match between two players. Game pairs share a start (a random opening, or
/// the standard opening with a bid when `opening_max` is 0) with seats
/// swapped; pairs run on `threads` threads. Returns A's score and the
/// per-game results (A wins, B wins, draws).
pub fn play_match(
    a: &Player,
    b: &Player,
    pairs: usize,
    opening_max: u32,
    opening_decay: f32,
    seed: u64,
    threads: usize,
) -> anyhow::Result<(f64, [u32; 3])> {
    use playground_core::{Game, Outcome};
    use rand::Rng;
    use std::sync::Mutex;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    // Starts drawn up front so results do not depend on the thread count.
    let starts: Vec<(Option<crate::game::MinionsState>, u64)> = (0..pairs)
        .map(|_| {
            let start = (opening_max != 0)
                .then(|| crate::selfplay::random_opening(opening_max, opening_decay, &mut rng).0);
            (start, rng.gen())
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    // STEPS_SHOW=1: print every position (use with --threads 1).
    let show = std::env::var("STEPS_SHOW").is_ok();
    // Per pair: A's points from each seat order, and whether seat 1 won each.
    let results: Mutex<Vec<(usize, [Outcome; 2], [bool; 2])>> = Mutex::new(Vec::new());
    let err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= pairs || err.lock().unwrap().is_some() {
                    return;
                }
                let (start, salt) = &starts[i];
                let game = |a_first: bool| -> anyhow::Result<(Outcome, bool)> {
                    let mut real = match start {
                        Some(s) => s.clone(),
                        None => {
                            let (ba, bb) =
                                (bid(a.eval.as_ref(), *salt)?, bid(b.eval.as_ref(), *salt)?);
                            // The two bids' average (rounded toward zero) is the handicap.
                            crate::game::MinionsState::with_bid((ba + bb) / 2)
                        }
                    };
                    let mut turns = 0;
                    while !crate::game::Minions::outcome(&real).is_terminal() && turns < 200 {
                        let a_moves = (real.side == 0) == a_first;
                        let p = if a_moves { a } else { b };
                        real = p.turn(&real, salt ^ turns as u64)?;
                        turns += 1;
                        if show {
                            eprintln!(
                                "--- pair {i} game {} turn {turns}: {} moved\n{}",
                                if a_first { 1 } else { 2 },
                                if a_moves { "A" } else { "B" },
                                real.render()
                            );
                        }
                    }
                    let res = crate::game::Minions::outcome(&real);
                    Ok((res, a_first))
                };
                match game(true).and_then(|g1| Ok((g1, game(false)?))) {
                    Ok(((r1, _), (r2, _))) => results.lock().unwrap().push((
                        i,
                        [r1, r2],
                        [r1 == Outcome::FirstWins, r2 == Outcome::FirstWins],
                    )),
                    Err(e) => *err.lock().unwrap() = Some(e),
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    let mut wdl = [0u32; 3];
    let mut score = 0f64;
    let (mut first_wins, mut split_pairs) = (0u32, 0u32);
    for (_, res, first) in results.into_inner().unwrap() {
        let mut pair_a = 0;
        for (k, r) in res.iter().enumerate() {
            let a_first = k == 0;
            first_wins += u32::from(first[k]);
            match r {
                Outcome::FirstWins | Outcome::SecondWins => {
                    if (*r == Outcome::FirstWins) == a_first {
                        wdl[0] += 1;
                        pair_a += 1;
                        score += 1.0;
                    } else {
                        wdl[1] += 1;
                    }
                }
                _ => {
                    wdl[2] += 1;
                    score += 0.5;
                }
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

#[cfg(test)]
mod tests {
    use super::outcome_code;

    #[test]
    fn outcome_codes() {
        // First player (mover 0) wins at turn 20; records from turn 12.
        assert_eq!(outcome_code(0, 1, 12, 20), 3 | (8 << 2));
        assert_eq!(outcome_code(1, 1, 12, 20), 1 | (8 << 2));
        assert_eq!(outcome_code(1, 2, 20, 20), 3);
        assert_eq!(outcome_code(0, 3, 5, 9), 2 | (4 << 2));
        // Unfinished game: unknown, whatever the turns.
        assert_eq!(outcome_code(0, 0, 5, 9), 0);
    }
}
