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
    /// Stop once this many records are written. Shared-tree self-play
    /// instead stops once `target_records / record_frac` nodes were eligible
    /// and writes exactly this many of them.
    pub target_records: u64,
    pub chunk_records: usize,
    pub seed: u64,
    /// Games sharing one search tree (with one worker); 1 = a tree per game.
    pub games: usize,
    /// Save each shared tree here ([`crate::tree_save`]).
    pub save_trees: Option<PathBuf>,
    /// Non-zero: the self-play trees only choose positions. Each sampled one
    /// gets its own best-first search of this many expansions at the end of
    /// the generation ([`resample`], the eval procedure), whose root value
    /// is the record's target.
    pub resample_expansions: u64,
    pub resample_heads: usize,
    pub resample_threads: usize,
    /// Search the sampled positions with [`crate::two_tier`] (the budget is
    /// still `resample_expansions` step expansions) instead of best-first.
    pub resample_two_tier: Option<crate::two_tier::TwoTierParams>,
}

/// A sampled position waiting for its own search ([`Config::resample_expansions`]).
struct Pending {
    state: StepState,
    /// A terminal's value needs no search.
    terminal: Option<f32>,
}

/// Shared-tree batches started, numbering the saved trees.
static BATCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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
    /// Keep every record in memory until [`Writer::finish_held`] (shared-tree
    /// self-play, which trims a generation to an exact count).
    hold: bool,
    /// With resampling: the sampled positions, searched and encoded at the end.
    pending: Option<Vec<Pending>>,
}

impl Writer {
    /// Records already encoded (by [`encode_record`]), with their movers.
    fn push_encoded(&mut self, recs: &[u8], movers: &[u8]) {
        self.game.extend_from_slice(recs);
        self.movers.extend_from_slice(movers);
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
            if !self.hold && self.n >= self.per_chunk {
                self.flush()?;
            }
        }
        self.movers.clear();
        Ok(())
    }

    /// Write a uniform sample of `keep` of the held records (all of them if
    /// there are fewer), in the order they were made, as chunks. Returns how
    /// many were written.
    fn finish_held(
        &mut self,
        keep: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> anyhow::Result<usize> {
        let all = std::mem::take(&mut self.buf);
        let n = all.len() / REC_BYTES;
        let mut idx: Vec<usize> = if n > keep {
            rand::seq::index::sample(rng, n, keep).into_vec()
        } else {
            (0..n).collect()
        };
        idx.sort_unstable();
        self.n = 0;
        for &i in &idx {
            self.buf
                .extend_from_slice(&all[i * REC_BYTES..(i + 1) * REC_BYTES]);
            self.n += 1;
            if self.n >= self.per_chunk {
                self.flush()?;
            }
        }
        self.flush()?;
        Ok(idx.len())
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
    /// Nodes eligible for recording (visited, or distinct terminals) so far.
    pub eligible: AtomicU64,
    /// Records written after trimming (shared-tree self-play).
    pub written: AtomicU64,
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
    /// Main-thread wait for the group in flight (see `SearchStats::wait_ns`).
    pub wait_ns: AtomicU64,
    pub take_ns: AtomicU64,
    pub land_ns: AtomicU64,
    /// Shared-tree work between rounds (thread-ns): snapshot, records,
    /// retain, and freeing the old tree (on a background thread).
    pub between_ns: [AtomicU64; 4],
    /// Starting bids, indexed from `BID_MIN`.
    pub bids: [AtomicU64; (crate::game::BID_MAX - crate::game::BID_MIN + 1) as usize],
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
        self.wait_ns.fetch_add(s.wait_ns, Ordering::Relaxed);
        self.take_ns.fetch_add(s.take_ns, Ordering::Relaxed);
        self.land_ns.fetch_add(s.land_ns, Ordering::Relaxed);
    }
}

/// Record the nodes of `tree` outside the subtree of `keep` (all nodes when
/// `keep` is `None`): every node with at least `min_visits` descents, and
/// every terminal node once per distinct position (many step orders reach
/// the same end). Each is kept with probability `frac * w / mean(w)` (capped
/// at 1), `w = max(visits, 1)^alpha`, so the expected count stays `frac` of
/// the eligible nodes while well-visited nodes are favoured (`alpha = 0` is
/// uniform).
/// Which nodes lie in `keep`'s subtrees.
pub(crate) fn inside_mask(tree: &Tree, keep: &[u32]) -> Vec<bool> {
    let mut inside = vec![false; tree.nodes.len()];
    // Children are always created after their parent, so one forward pass
    // marks the kept subtrees.
    for &k in keep {
        inside[k as usize] = true;
    }
    if let Some(&lo) = keep.iter().min() {
        for i in (lo as usize + 1)..tree.nodes.len() {
            let p = tree.nodes[i].parent;
            if p != u32::MAX && inside[p as usize] {
                inside[i] = true;
            }
        }
    }
    inside
}

fn record_nodes(
    tree: &Tree,
    keep: &[u32],
    min_visits: u32,
    frac: f64,
    alpha: f64,
    rng: &mut rand_chacha::ChaCha8Rng,
    w: &mut Writer,
) -> anyhow::Result<(u64, u64)> {
    use rand::Rng;
    let inside = inside_mask(tree, keep);
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
    // Forced steps are sampled like any node but do not count toward the
    // generation's eligible-node target: they are a quarter of the nodes and
    // over half of the visited ones (each inherits its parent's descents), so
    // counting them would end self-play at about half the moves.
    let eligible = cand.iter().filter(|&&id| !tree.is_forced(id)).count() as u64;
    let picked: Vec<u32> = cand
        .into_iter()
        .filter(|&id| frac >= 1.0 || rng.gen::<f64>() < frac * weight(id) / mean)
        .collect();
    if let Some(pending) = w.pending.as_mut() {
        for &id in &picked {
            let nd = &tree.nodes[id as usize];
            pending.push(Pending {
                state: tree.created_state(id),
                terminal: (nd.kind == 2).then_some(nd.value),
            });
        }
        return Ok((picked.len() as u64, eligible));
    }
    // Encoding (a state replay or copy, its view) in parallel over slices,
    // appended in order: the same bytes as one thread pushing each record.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = picked.len().div_ceil(threads).max(1);
    let parts: Vec<(Vec<u8>, Vec<u8>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = picked
            .chunks(per)
            .map(|ids| {
                sc.spawn(move || {
                    let (mut recs, mut movers) = (Vec::new(), Vec::with_capacity(ids.len()));
                    for &id in ids {
                        let nd = &tree.nodes[id as usize];
                        // The view the node's prior was evaluated on, and the
                        // one search evaluates: before its expansion collapsed
                        // forced steps (e.g. the enemy necromancer's "spare"
                        // that starts nearly every turn) into the stored state.
                        let st = tree.created_state(id);
                        encode_record(&st.view(), nd.value, nd.prior, nd.visits, &mut recs);
                        movers.push(u8::from(st.flipped));
                    }
                    (recs, movers)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("record encoder"))
            .collect()
    });
    for (recs, movers) in &parts {
        w.push_encoded(recs, movers);
    }
    Ok((picked.len() as u64, eligible))
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
    let salt: u64 = rng.gen();
    // The standard opening starts at the net's own bid, so the game itself is
    // the only source of randomness.
    let start = if cfg.opening_max == 0 {
        let b = bid(eval, salt)?;
        c.bids[(b - crate::game::BID_MIN) as usize].fetch_add(1, Ordering::Relaxed);
        crate::game::MinionsState::with_bid(b)
    } else {
        crate::selfplay::random_opening(cfg.opening_max, cfg.opening_decay, rng).0
    };
    let mut tree = Tree::new(StepState::begin(&start, salt), cfg.params.clone(), eval)?;
    let mut turns = 0u64;
    while !tree.root.is_terminal() && !tree.is_solved() {
        if stop.load(Ordering::Relaxed) {
            return w.finish_game(0, 0);
        }
        let t = Instant::now();
        let before = tree.stats.expansions;
        let mut starved = 0;
        if cfg.descent && cfg.params.threads > 1 {
            // One tree with the whole machine: GPU and CPU overlapped.
            let (budget, secs) = (cfg.expansions, cfg.move_secs);
            tree.descent_run(eval, cfg.params.heads, &mut |tr| {
                tr.stats.expansions - before >= budget
                    || secs > 0.0 && t.elapsed().as_secs_f64() > secs
            })?;
        }
        while !(cfg.descent && cfg.params.threads > 1)
            && tree.stats.expansions - before < cfg.expansions
        {
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
        crate::steps::prof::flush();
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
        let tp = Instant::now();
        let next = walk_turn(&mut tree, 0, cfg, eval, rng, c)?;
        c.walk_ns
            .fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let Some(next) = next else { break };
        let (rec, _) = record_nodes(
            &tree,
            &[next],
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
    let (rec, _) = record_nodes(
        &tree,
        &[],
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

/// Play one whole turn from `from`: steps until the move passes or the game
/// ends; the node reached, or `None` if the walk got stuck. A pick that
/// lands on a dead end is backed out of: expanding it marks it dead, and the
/// walk resumes from its parent.
fn walk_turn(
    tree: &mut Tree,
    from: u32,
    cfg: &Config,
    eval: &dyn Evaluator,
    rng: &mut rand_chacha::ChaCha8Rng,
    c: &Counters,
) -> anyhow::Result<Option<u32>> {
    let tp = Instant::now();
    let (mut walk_exp, mut walk_steps) = (0u32, 0u32);
    let mut n = from;
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
            if n == from {
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
    if tp.elapsed().as_secs_f64() > 5.0 {
        eprintln!(
            "slow walk {:.1}s: {walk_steps} steps, {walk_exp} single expansions, turn {}",
            tp.elapsed().as_secs_f64(),
            tree.state_of(from).base.turn
        );
    }
    Ok(next)
}

/// `cfg.games` games from the same start, searched in one shared tree: each
/// round the descents are spread over the games' positions (games at the
/// same position share them), then every game plays one turn from its own
/// position. Afterwards only the subtrees under the games' new positions are
/// kept; everything else is recorded with its final values, as a rerooting
/// does for one game. Outcomes are left unknown in the records (they are
/// shared between games).
fn play_shared(
    cfg: &Config,
    eval: &dyn Evaluator,
    rng: &mut rand_chacha::ChaCha8Rng,
    w: &mut Writer,
    c: &Counters,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    use rand::Rng;
    let salt: u64 = rng.gen();
    let start = if cfg.opening_max == 0 {
        let b = bid(eval, salt)?;
        c.bids[(b - crate::game::BID_MIN) as usize].fetch_add(1, Ordering::Relaxed);
        crate::game::MinionsState::with_bid(b)
    } else {
        crate::selfplay::random_opening(cfg.opening_max, cfg.opening_decay, rng).0
    };
    let mut tree = Tree::new(StepState::begin(&start, salt), cfg.params.clone(), eval)?;
    let mut saver = match &cfg.save_trees {
        Some(dir) => Some(crate::tree_save::TreeSaver::new(
            dir,
            BATCH.fetch_add(1, Ordering::Relaxed),
            cfg.games.max(1),
            &start,
            salt,
            cfg.params.power,
        )?),
        None => None,
    };
    let mut mv = 0usize;
    // Each game's position, or None once it is over. Random openings
    // (`opening_max` > 0): every game starts from its own, as one more root
    // of the forest; otherwise all start at the standard opening (node 0).
    let mut games: Vec<Option<u32>> = vec![Some(0); cfg.games.max(1)];
    if cfg.opening_max != 0 {
        for (g, pos) in games.iter_mut().enumerate().skip(1) {
            let s = crate::selfplay::random_opening(cfg.opening_max, cfg.opening_decay, rng).0;
            *pos = Some(tree.add_root(StepState::begin(&s, salt ^ g as u64), eval)?);
        }
    }
    let mut turns = vec![0u64; games.len()];
    let over = |tree: &Tree, n: u32| {
        let nd = &tree.nodes[n as usize];
        nd.kind == 2 || nd.kind == 3 || nd.resolved
    };
    loop {
        for g in games.iter_mut() {
            if g.is_some_and(|n| over(&tree, n)) {
                *g = None;
            }
        }
        let mut at: Vec<u32> = games.iter().flatten().copied().collect();
        at.sort_unstable();
        at.dedup();
        // The generation ends once enough nodes were eligible, mid-game: the
        // subtrees under the games still going are recorded as they stand.
        // (Visit weighting caps some keep probabilities at 1, so the sample
        // can fall short of the target even with the oversampling.)
        let enough = c.eligible.load(Ordering::Relaxed) >= eligible_target(cfg)
            && c.records.load(Ordering::Relaxed) >= cfg.target_records;
        if at.is_empty() || enough || stop.load(Ordering::Relaxed) {
            if !at.is_empty() {
                let (rec, elig) = record_nodes(
                    &tree,
                    &[],
                    cfg.min_visits,
                    oversampled(cfg.record_frac),
                    cfg.record_alpha,
                    rng,
                    w,
                )?;
                w.finish_game(0, 0)?;
                c.records.fetch_add(rec, Ordering::Relaxed);
                c.eligible.fetch_add(elig, Ordering::Relaxed);
                c.add_stats(&tree.stats);
            }
            if enough {
                stop.store(true, Ordering::Relaxed);
            }
            if let Some(s) = saver.take() {
                let why = if at.is_empty() {
                    "games over"
                } else if enough {
                    "enough records"
                } else {
                    "stopped"
                };
                s.finish(&tree, why)?;
            }
            break;
        }
        let t = Instant::now();
        let before = tree.stats.expansions;
        tree.set_starts(&at);
        let (budget, secs) = (cfg.expansions, cfg.move_secs);
        tree.descent_run(eval, cfg.params.heads, &mut |tr| {
            tr.stats.expansions - before >= budget || secs > 0.0 && t.elapsed().as_secs_f64() > secs
        })?;
        tree.clear_descents();
        crate::steps::prof::flush();
        c.search_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        c.nodes_at_move
            .fetch_add(tree.nodes.len() as u64, Ordering::Relaxed);
        let tp = Instant::now();
        let froms = games.clone();
        for (g, pos) in games.iter_mut().enumerate() {
            let Some(n) = *pos else { continue };
            *pos = walk_turn(&mut tree, n, cfg, eval, rng, c)?;
            c.moves.fetch_add(1, Ordering::Relaxed);
            turns[g] += 1;
            if pos.is_none() {
                finished(c, &turns[g], None);
            }
        }
        c.walk_ns
            .fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let tb = Instant::now();
        if let Some(s) = saver.as_mut() {
            let moved: Vec<_> = froms
                .iter()
                .enumerate()
                .filter_map(|(g, from)| from.map(|f| (g, f, games[g])))
                .collect();
            s.snapshot(&tree, &format!("r{mv:03}.bin"), &moved)?;
        }
        c.between_ns[0].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
        mv += 1;
        // Games that just ended are counted; their positions are not kept.
        let mut keep = Vec::new();
        for (g, pos) in games.iter_mut().enumerate() {
            if let Some(n) = *pos {
                if over(&tree, n) {
                    finished(c, &turns[g], Some(&tree.state_of(n)));
                    *pos = None;
                } else {
                    keep.push(n);
                }
            }
        }
        keep.sort_unstable();
        keep.dedup();
        let tb = Instant::now();
        let (rec, elig) = record_nodes(
            &tree,
            &keep,
            cfg.min_visits,
            oversampled(cfg.record_frac),
            cfg.record_alpha,
            rng,
            w,
        )?;
        w.finish_game(0, 0)?;
        c.records.fetch_add(rec, Ordering::Relaxed);
        c.eligible.fetch_add(elig, Ordering::Relaxed);
        c.add_stats(&tree.stats);
        c.between_ns[1].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let tb = Instant::now();
        let (t2, map) = tree.retain(&keep);
        if let Some(s) = saver.as_mut() {
            s.retained(&tree, &keep, &map);
        }
        c.between_ns[2].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // Freeing millions of node states takes a while; the next round
        // does not wait for it.
        let old = std::mem::replace(&mut tree, t2);
        std::thread::spawn(move || {
            let tb = Instant::now();
            drop(old);
            BETWEEN_DROP_NS.fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
        });
        for pos in games.iter_mut() {
            if let Some(n) = pos {
                *n = map[*n as usize];
            }
        }
    }
    Ok(())
}

/// Time spent freeing old trees in the background (see [`play_shared`]).
static BETWEEN_DROP_NS: AtomicU64 = AtomicU64::new(0);

/// Shared-tree self-play samples this much more than `record_frac` and then
/// trims the generation to exactly `target_records` ([`Writer::finish_held`]).
const OVERSAMPLE: f64 = 1.25;

fn oversampled(frac: f64) -> f64 {
    if frac < 1.0 {
        (frac * OVERSAMPLE).min(1.0)
    } else {
        frac
    }
}

/// Eligible nodes after which shared-tree self-play stops: `record_frac` of
/// them is the target.
fn eligible_target(cfg: &Config) -> u64 {
    (cfg.target_records as f64 / cfg.record_frac.min(1.0)).ceil() as u64
}

/// Count one finished game of [`play_shared`] (`end`: its final position,
/// `None` if its walk got stuck).
fn finished(c: &Counters, turns: &u64, end: Option<&StepState>) {
    c.games.fetch_add(1, Ordering::Relaxed);
    c.turns.fetch_add(*turns, Ordering::Relaxed);
    if let Some(e) = end {
        if e.is_terminal() && e.base.winner != 3 {
            c.decisive.fetch_add(1, Ordering::Relaxed);
        }
    }
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
                hold: cfg.games > 1,
                pending: (cfg.resample_expansions > 0).then(Vec::new),
            };
            while !stop.load(Ordering::Relaxed) {
                if cfg.games > 1 {
                    play_shared(&cfg, eval.as_ref(), &mut rng, &mut w, &c, &stop)?;
                } else {
                    play_game(&cfg, eval.as_ref(), &mut rng, &mut w, &c, &stop)?;
                }
                if cfg.games <= 1 && c.records.load(Ordering::Relaxed) >= cfg.target_records {
                    stop.store(true, Ordering::Relaxed);
                }
            }
            if let Some(mut pending) = w.pending.take() {
                // Trim to the generation's count first: only kept positions
                // are searched.
                if w.hold && pending.len() > cfg.target_records as usize {
                    let mut idx = rand::seq::index::sample(
                        &mut rng,
                        pending.len(),
                        cfg.target_records as usize,
                    )
                    .into_vec();
                    idx.sort_unstable();
                    let mut keep = vec![false; pending.len()];
                    idx.iter().for_each(|&i| keep[i] = true);
                    let mut k = keep.into_iter();
                    pending.retain(|_| k.next().unwrap());
                }
                let (recs, movers) = resample(&cfg, eval.as_ref(), &pending)?;
                w.push_encoded(&recs, &movers);
                w.finish_game(0, 0)?;
            }
            if w.hold {
                let n = w.finish_held(cfg.target_records as usize, &mut rng)?;
                c.written.fetch_add(n as u64, Ordering::Relaxed);
                return Ok(());
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
                "  exact search: found {} impossible {} out-of-budget {} (deadline {}) | {:.0} thread-s",
                g(&x[0]),
                g(&x[1]),
                g(&x[2]),
                g(&crate::steps::DEADLINE_HITS),
                g(&x[3]) as f64 * 1e-9
            );
            let z = &crate::steps::EXACT_OUTCOME;
            println!(
                "  exact by outcome: found {:.0}s {:.0} nodes/call, impossible {:.0}s {:.0} nodes/call, out-of-budget {:.0}s | repeated queries {}",
                g(&z[0][0]) as f64 * 1e-9, g(&z[0][1]) as f64 / g(&x[0]).max(1) as f64,
                g(&z[1][0]) as f64 * 1e-9, g(&z[1][1]) as f64 / g(&x[1]).max(1) as f64,
                g(&z[2][0]) as f64 * 1e-9, g(&z[3][0])
            );
            let pr: Vec<String> = crate::steps::prof::NAMES
                .iter()
                .zip(crate::steps::prof::TOTAL.iter())
                .map(|(n, t)| format!("{n} {:.0}s", g(t) as f64 * 1e-9))
                .collect();
            println!("  profile (thread-s, nested): {}", pr.join(", "));
            let lz = &crate::steps::LAZY_STATS;
            if g(&lz[0]) > 0 {
                println!(
                    "  lazy witnesses: offered {} resolved {} dead {}",
                    g(&lz[0]),
                    g(&lz[1]),
                    g(&lz[2])
                );
            }
            let b = &c.between_ns;
            println!(
                "  between rounds (s): snapshot {:.1}, records {:.1}, retain {:.1}, free old tree {:.1} (background)",
                g(&b[0]) as f64 * 1e-9,
                g(&b[1]) as f64 * 1e-9,
                g(&b[2]) as f64 * 1e-9,
                g(&BETWEEN_DROP_NS) as f64 * 1e-9
            );
            let ep = &crate::bfs::EVAL_PROF;
            println!(
                "  evaluator (s): featurize {:.1}, wait for values {:.1}, in ONNX calls {:.1} (runners summed); main thread waited {:.1} for groups",
                g(&ep[0]) as f64 * 1e-9,
                g(&ep[1]) as f64 * 1e-9,
                g(&ep[2]) as f64 * 1e-9,
                g(&c.wait_ns) as f64 * 1e-9
            );
            println!(
                "  main thread (s): select {:.1}, take states {:.1}, wait for groups {:.1}, land {:.1}, backup {:.1}",
                g(&c.select_ns) as f64 * 1e-9,
                g(&c.take_ns) as f64 * 1e-9,
                g(&c.wait_ns) as f64 * 1e-9,
                g(&c.land_ns) as f64 * 1e-9,
                g(&c.backup_ns) as f64 * 1e-9
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
    if cfg.games > 1 {
        println!(
            "done: {} records ({} sampled from {} eligible nodes), {} games finished",
            c.written.load(Ordering::Relaxed),
            c.records.load(Ordering::Relaxed),
            c.eligible.load(Ordering::Relaxed),
            c.games.load(Ordering::Relaxed)
        );
    } else {
        println!(
            "done: {} records, {} games",
            c.records.load(Ordering::Relaxed),
            c.games.load(Ordering::Relaxed)
        );
    }
    let bids: Vec<String> = (c.bids.iter().zip(crate::game::BID_MIN..))
        .filter_map(|(n, b)| {
            let n = n.load(Ordering::Relaxed);
            (n > 0).then(|| format!("{b}:{n}"))
        })
        .collect();
    if !bids.is_empty() {
        println!("bids: {}", bids.join(" "));
    }
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
        // Best first; under lazy witnesses an option may defer its check
        // to the child, which can turn out a dead end: take the best child
        // whose check succeeds.
        let mut order: Vec<usize> = (0..kids.len()).collect();
        order.sort_by(|&a, &b| q[b].total_cmp(&q[a]));
        let mut kids: Vec<Option<StepState>> = kids.into_iter().map(Some).collect();
        let mut next = None;
        for i in order {
            let mut k = kids[i].take().expect("kid");
            if k.resolve_pending() {
                next = Some(k);
                break;
            }
        }
        s = next.ok_or_else(|| anyhow::anyhow!("every option of a greedy step is a dead end"))?;
    }
    Ok(s.real_base())
}

/// `n` uniformly random turns from the standard start with `handicap`
/// dollars to blue, resampled while they end the game: a match's short,
/// distinct opening.
pub fn short_opening(n: u32, handicap: i16, rng: &mut rand_chacha::ChaCha8Rng) -> crate::game::MinionsState {
    use playground_core::Game;
    loop {
        let mut s = crate::game::MinionsState::with_bid(handicap);
        for _ in 0..n {
            if crate::game::Minions::outcome(&s).is_terminal() {
                break;
            }
            let spec = crate::turn::TurnSpec::random_playout(&s, rng);
            s = crate::turn::realize(&s, &spec).next;
        }
        if !crate::game::Minions::outcome(&s).is_terminal() {
            return s;
        }
    }
}

/// `n` turns from the standard start at `handicap`, each one tier-2 descent
/// of `eval`'s lattice with a random variable order and each choice drawn
/// by the ordinal rule over the net's values (self-play's opening sampler):
/// plausible turns, distinct per pair. Resampled while the game ends.
pub fn sampled_opening(
    n: u32,
    handicap: i16,
    eval: &dyn crate::lattice::LatticeEval,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> anyhow::Result<crate::game::MinionsState> {
    use crate::lattice::{ExpandReq, Kind, Lattice, Need};
    use rand::Rng;
    'again: loop {
        let mut s = StepState::begin(&crate::game::MinionsState::with_bid(handicap), rng.gen());
        for _ in 0..n {
            let mut lat = Lattice::new(s.clone())?;
            if lat.terminal.is_some() {
                continue 'again;
            }
            let mut at = 0u32;
            loop {
                match lat.nodes[at as usize].kind {
                    Kind::Sink => break,
                    Kind::Dead => at = 0,
                    Kind::Inner => {
                        if let Some(Need::Expand(e)) = lat.ensure_expanded(at)? {
                            let (slots, node) = lat.request(e);
                            let req = ExpandReq { root: &lat.start, node, slots: &slots, trunk: lat.trunk.clone() };
                            let out = eval.expand(&[req])?.pop().expect("one output");
                            lat.feed_expand(e, &out);
                        }
                        let nd = &lat.nodes[at as usize];
                        if nd.kind != Kind::Inner {
                            continue;
                        }
                        let mut vars: Vec<u8> = nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
                        vars.dedup();
                        let var = vars[rng.gen_range(0..vars.len())];
                        let idx: Vec<usize> = (0..nd.kids.len()).filter(|&i| nd.kids[i].var == var && !nd.kids[i].dead).collect();
                        let vals: Vec<f32> = idx.iter().map(|&i| nd.kids[i].v).collect();
                        let k = idx[crate::latgen::pick_ordinal(&vals, rng)];
                        at = lat.child(at, k)?;
                    }
                }
            }
            s = lat.nodes[at as usize].state.clone();
            if s.is_terminal() {
                continue 'again;
            }
        }
        return Ok(s.real_base());
    }
}

/// `s` with `x` dollars added: to blue (the second player) if positive, to
/// yellow if negative. The adjudication window restarts from here.
pub fn with_money_bid(s: &crate::game::MinionsState, x: i16) -> crate::game::MinionsState {
    let mut t = s.clone();
    let side = if x >= 0 { 1 } else { 0 };
    t.money[side] = t.money[side].saturating_add(x.unsigned_abs());
    t.prime_nw_hist();
    t
}

/// Bids range over this many dollars either way.
pub const EVEN_BID_MAX: i16 = 30;

/// [`even_bid`] by the lattice's own tier-1 value (what search reads). The
/// steps-format export of the same net (`value_sparse.onnx`) is nearly blind
/// to money (2026-10-11: slope ~0 vs 0.09 per dollar here), so bids and
/// money probes read this.
pub fn lattice_even_bid(lat: &dyn crate::lattice::LatticeEval, s: &crate::game::MinionsState, salt: u64) -> anyhow::Result<i16> {
    let yellow = |x: i16| -> anyhow::Result<f32> {
        let t = with_money_bid(s, x);
        let v = lat.value(&[&StepState::begin(&t, salt)])?[0];
        Ok(if t.side == 0 { v } else { 1.0 - v })
    };
    let (mut lo, mut hi) = (-EVEN_BID_MAX, EVEN_BID_MAX);
    if yellow(lo)? < 0.5 {
        return Ok(lo);
    }
    if yellow(hi)? >= 0.5 {
        return Ok(hi);
    }
    while hi - lo > 1 {
        let mid = (lo + hi).div_euclid(2);
        if yellow(mid)? >= 0.5 {
            lo = mid
        } else {
            hi = mid
        }
    }
    Ok(if (yellow(lo)? - 0.5).abs() <= (0.5 - yellow(hi)?).abs() { lo } else { hi })
}

/// A side's bid for position `s`: the dollars `x` (to blue if positive, to
/// yellow if negative, on top of the money `s` has) at which its evaluator
/// rates the position even, by binary search on yellow's (decreasing) value.
pub fn even_bid(eval: &dyn Evaluator, s: &crate::game::MinionsState, salt: u64) -> anyhow::Result<i16> {
    let yellow = |x: i16| -> anyhow::Result<f32> {
        let t = with_money_bid(s, x);
        let v = eval.eval(&[StepState::begin(&t, salt).view()])?[0];
        Ok(if t.side == 0 { v } else { 1.0 - v })
    };
    if std::env::var_os("BID_DEBUG").is_some() {
        let mut c = Vec::new();
        for x in (-EVEN_BID_MAX..=EVEN_BID_MAX).step_by(5) {
            c.push(format!("{x}:{:.2}", yellow(x)?));
        }
        eprintln!("yellow value by $ to blue (turn {}, money {:?}): {}", s.turn, s.money, c.join(" "));
    }
    let (mut lo, mut hi) = (-EVEN_BID_MAX, EVEN_BID_MAX);
    if yellow(lo)? < 0.5 {
        return Ok(lo);
    }
    if yellow(hi)? >= 0.5 {
        return Ok(hi);
    }
    while hi - lo > 1 {
        let mid = (lo + hi).div_euclid(2);
        if yellow(mid)? >= 0.5 {
            lo = mid
        } else {
            hi = mid
        }
    }
    // The closer of the two to even.
    Ok(if (yellow(lo)? - 0.5).abs() <= (0.5 - yellow(hi)?).abs() { lo } else { hi })
}

/// How a net values money: over `positions` openings sampled as matches
/// sample them (`opening_turns` net-sampled turns at `handicap`), yellow's
/// value (by `value`, the tier-1 net) with extra dollars to blue (or yellow,
/// if negative) on top of the position's money; prints the mean curve, the
/// share of positions where it falls monotonically, its slope around 0 and
/// the even bids. A net that prices money: a steep, monotone curve and bids
/// off the +-EVEN_BID_MAX bounds.
pub fn money_probe(
    value: &dyn Evaluator,
    lat: &dyn crate::lattice::LatticeEval,
    positions: u32,
    opening_turns: u32,
    handicap: i16,
    seed: u64,
    turns_left: Option<u16>,
) -> anyhow::Result<()> {
    const GRID: [i16; 9] = [-10, -6, -4, -2, 0, 2, 4, 6, 10];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let (mut sum, mut mono, mut slopes, mut bids) = ([0f64; 9], 0u32, Vec::new(), Vec::new());
    for i in 0..positions {
        let mut s = sampled_opening(opening_turns, handicap, lat, &mut rng)?;
        // --turns-left: the same position, but the cap that many turns away
        // (turn parity kept, so the side to move is unchanged).
        if let Some(k) = turns_left {
            let mut t = crate::game::MAX_TURNS.saturating_sub(k);
            if t % 2 != s.turn % 2 {
                t = t.saturating_sub(1);
            }
            s.turn = t;
        }
        let salt = seed ^ u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let ts: Vec<crate::game::MinionsState> = GRID.iter().map(|&x| with_money_bid(&s, x)).collect();
        let ss: Vec<StepState> = ts.iter().map(|t| StepState::begin(t, salt)).collect();
        let views: Vec<View> = ss.iter().map(|x| x.view()).collect();
        // MONEY_PROBE_STEPS=1: the old path, the tier-1 net exported as a
        // steps value net (value_sparse.onnx). By default the lattice's own
        // tier-1 value, as search reads it.
        let v = if std::env::var("MONEY_PROBE_STEPS").is_ok_and(|x| x == "1") {
            value.eval(&views)?
        } else {
            lat.value(&ss.iter().collect::<Vec<_>>())?
        };
        let y: Vec<f64> = ts
            .iter()
            .zip(&v)
            .map(|(t, &v)| f64::from(if t.side == 0 { v } else { 1.0 - v }))
            .collect();
        for k in 0..9 {
            sum[k] += y[k];
        }
        if y.windows(2).all(|w| w[1] <= w[0] + 0.005) {
            mono += 1;
        }
        slopes.push((y[2] - y[6]) / 8.0);
        bids.push(f64::from(if std::env::var("MONEY_PROBE_STEPS").is_ok_and(|x| x == "1") {
            even_bid(value, &s, salt)?
        } else {
            lattice_even_bid(lat, &s, salt)?
        }));
    }
    let n = f64::from(positions.max(1));
    let curve: Vec<String> = GRID.iter().zip(&sum).map(|(x, s)| format!("{x}:{:.3}", s / n)).collect();
    let ms = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
        (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt())
    };
    let ((sm, ssd), (bm, bsd)) = (ms(&slopes), ms(&bids));
    let bound = bids.iter().filter(|b| b.abs() >= f64::from(EVEN_BID_MAX)).count();
    println!(
        "money probe: {positions} positions after {opening_turns} sampled turns at handicap {handicap}{}",
        turns_left.map_or(String::new(), |k| format!(", {k} turns left"))
    );
    println!("  yellow value by $ to blue: {}", curve.join(" "));
    println!("  monotone (falling) in {:.0}% of positions", 100.0 * f64::from(mono) / n);
    println!("  slope -4..+4: {sm:.4} per dollar (sd {ssd:.4})");
    println!("  even bid: mean {bm:.1}, sd {bsd:.1}, at the +-{EVEN_BID_MAX} bound in {bound} of {positions}");
    Ok(())
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

/// Best-first iterations on `tree` until `expansions` leaf expansions (or
/// `secs` seconds if positive), the tree is solved, or the candidate pool
/// starves (fewer than a quarter of the heads filled 8 times in a row).
fn search_tree(
    tree: &mut Tree,
    eval: &dyn Evaluator,
    expansions: u64,
    secs: f64,
) -> anyhow::Result<()> {
    let t = Instant::now();
    let mut starved = 0;
    while !tree.is_solved() && tree.stats.expansions < expansions {
        if secs > 0.0 && t.elapsed().as_secs_f64() > secs {
            break;
        }
        let got = tree.iterate(eval)?;
        if got == 0 {
            break;
        }
        if got * 4 < tree.params.heads {
            starved += 1;
            if starved >= 8 {
                break;
            }
        } else {
            starved = 0;
        }
    }
    Ok(())
}

/// The end-of-generation searches of [`Config::resample_expansions`]: every
/// pending position gets a fresh tree searched like [`search_turn`] (best-first,
/// `resample_heads` leaves per iteration) for `resample_expansions`
/// expansions; its root value is the record's target and its static value the
/// prior. Positions run on `resample_threads` threads, which share `eval`'s
/// batching. Returns the records in the positions' order, with their movers.
fn resample(
    cfg: &Config,
    eval: &dyn Evaluator,
    pending: &[Pending],
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let t0 = Instant::now();
    let params = Params {
        heads: cfg.resample_heads.max(1),
        power: cfg.params.power,
        threads: 1,
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    // Two-tier totals: tier-1 descents, tier-2 descents, tier-2 repeats,
    // widenings dropped, root turns, tier-1 nodes, evaluations.
    let tt_stats: [AtomicU64; 7] = Default::default();
    let parts: Vec<anyhow::Result<Vec<(usize, Vec<u8>, u64)>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..cfg.resample_threads.max(1))
            .map(|_| {
                sc.spawn(|| -> anyhow::Result<Vec<(usize, Vec<u8>, u64)>> {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(p) = pending.get(i) else { break };
                        let (target, prior, exp) = match p.terminal {
                            Some(v) => (v, v, 0),
                            None if cfg.resample_two_tier.is_some() => {
                                let tt = cfg.resample_two_tier.clone().unwrap();
                                let mut s =
                                    crate::two_tier::TwoTier::new(p.state.clone(), tt, eval)?;
                                s.run(eval, cfg.resample_expansions)?;
                                let st = &s.stats;
                                for (a, v) in tt_stats.iter().zip([
                                    st.descents,
                                    st.t2_descents,
                                    st.t2_repeats,
                                    st.widen_dropped,
                                    s.nodes[0].children.len() as u64,
                                    s.nodes.len() as u64,
                                    st.evals,
                                ]) {
                                    a.fetch_add(v, Ordering::Relaxed);
                                }
                                (s.root_value(), s.nodes[0].prior, st.expansions)
                            }
                            None => {
                                let mut tree = Tree::new(p.state.clone(), params.clone(), eval)?;
                                search_tree(&mut tree, eval, cfg.resample_expansions, 0.0)?;
                                (
                                    tree.nodes[0].value,
                                    tree.nodes[0].prior,
                                    tree.stats.expansions,
                                )
                            }
                        };
                        let mut rec = Vec::with_capacity(REC_BYTES);
                        encode_record(
                            &p.state.view(),
                            target,
                            prior,
                            exp.min(u64::from(u32::MAX)) as u32,
                            &mut rec,
                        );
                        out.push((i, rec, exp));
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("resample thread"))
            .collect()
    });
    let mut all = Vec::with_capacity(pending.len());
    for p in parts {
        all.extend(p?);
    }
    all.sort_unstable_by_key(|x| x.0);
    let exp: u64 = all.iter().map(|x| x.2).sum();
    let secs = t0.elapsed().as_secs_f64();
    let how = match &cfg.resample_two_tier {
        Some(tt) => format!("two-tier, c {} retries {}", tt.c_uct, tt.retries),
        None => format!("best-first, {} heads", params.heads),
    };
    println!(
        "  resample: {} positions x {} expansions ({how}, {} threads): {:.0}s, {:.0} expansions/s",
        pending.len(),
        cfg.resample_expansions,
        cfg.resample_threads,
        secs,
        exp as f64 / secs.max(1e-9)
    );
    if cfg.resample_two_tier.is_some() {
        let g: Vec<f64> = tt_stats
            .iter()
            .map(|a| a.load(Ordering::Relaxed) as f64 / pending.len().max(1) as f64)
            .collect();
        println!(
            "  two-tier per position: tier-1 descents {:.1}, tier-2 descents {:.1} ({:.1} repeats), widenings dropped {:.2}, root turns {:.1}, tier-1 nodes {:.1}, evals {:.0}",
            g[0], g[1], g[2], g[3], g[4], g[5], g[6]
        );
    }
    let mut recs = Vec::with_capacity(all.len() * REC_BYTES);
    let mut movers = Vec::with_capacity(all.len());
    for (i, rec, _) in all {
        recs.extend_from_slice(&rec);
        movers.push(u8::from(pending[i].state.flipped));
    }
    Ok((recs, movers))
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
    let mut tree = Tree::new(StepState::begin(real, salt), params.clone(), eval)?;
    search_tree(&mut tree, eval, expansions, secs)?;
    let mut n = 0u32;
    loop {
        tree.ensure_expanded(n, eval)?;
        if tree.kind(n) == 3 {
            anyhow::ensure!(n != 0, "dead root in a search turn");
            n = tree.nodes[n as usize].parent;
            continue;
        }
        let Some(ch) = tree.best_child(n) else {
            let nd = &tree.nodes[n as usize];
            let kinds: Vec<u8> = tree.children(n).map(|c| tree.kind(c)).collect();
            anyhow::bail!(
                "no live child in a search turn: node {n} kind {} nch {} resolved {} comp {} children kinds {kinds:?}\n{}",
                nd.kind,
                nd.nch,
                nd.resolved,
                nd.comp,
                real.render()
            );
        };
        if tree.nodes[ch as usize].flip || tree.kind(ch) == 2 {
            let fin = tree.state_of(ch);
            return Ok(SearchedTurn {
                next: fin.real_base(),
                done: *fin.done,
                value: tree.nodes[0].value,
                expansions: tree.stats.expansions,
            });
        }
        n = ch;
    }
}

/// One turn by [`crate::two_tier`] search: `expansions` step expansions from
/// the turn's start, then the best turn found played.
pub fn two_tier_turn(
    real: &crate::game::MinionsState,
    eval: &dyn Evaluator,
    params: &crate::two_tier::TwoTierParams,
    expansions: u64,
    salt: u64,
) -> anyhow::Result<SearchedTurn> {
    let mut s = crate::two_tier::TwoTier::new(StepState::begin(real, salt), params.clone(), eval)?;
    s.run(eval, expansions)?;
    let b = s
        .best_child()
        .ok_or_else(|| anyhow::anyhow!("no turn found\n{}", real.render()))?;
    let fin = s.end_state(b);
    Ok(SearchedTurn {
        next: fin.real_base(),
        done: *fin.done,
        value: s.root_value(),
        expansions: s.stats.expansions,
    })
}

/// How one side plays its turns in a match.
#[derive(Clone)]
pub struct Player {
    pub eval: Arc<dyn Evaluator>,
    /// 0 = greedy (static values only); otherwise [`search_turn`] with this
    /// many expansions per turn.
    pub expansions: u64,
    pub params: Params,
    /// Search with [`crate::two_tier`] (budget `expansions`) instead of
    /// best-first.
    pub two_tier: Option<crate::two_tier::TwoTierParams>,
    /// Play by turn-lattice tier-1 search instead (`eval` then only bids).
    pub lattice: Option<LatticePlay>,
    /// Play the heuristic greedy solver ([`crate::solver::greedy_move`], one
    /// atomic move at a time) instead (`eval` then only bids).
    pub solver: bool,
}

/// A turn-lattice player: `n1` tier-1 visits, `inflight` at a time, then
/// the most visited turn.
#[derive(Clone)]
pub struct LatticePlay {
    pub eval: Arc<dyn crate::lattice::LatticeEval>,
    pub n1: u32,
    pub inflight: usize,
    pub params: crate::tier1::T1Params,
    /// > 0: search this many seconds a turn instead of `n1` visits.
    pub secs: f64,
}

/// Match game logs for the data explorer (`steps-match --game-log`): the
/// file, and the current match's two players' names.
static MATCH_LOG: std::sync::OnceLock<std::sync::Mutex<(std::io::BufWriter<std::fs::File>, String, String)>> =
    std::sync::OnceLock::new();

/// Log every game of this process's matches to `path` (JSON lines).
pub fn set_match_log(path: &std::path::Path) -> anyhow::Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let _ = MATCH_LOG.set(std::sync::Mutex::new((f, String::new(), String::new())));
    Ok(())
}

/// The players' names in the match log's next games.
pub fn set_match_names(a: &str, b: &str) {
    if let Some(m) = MATCH_LOG.get() {
        let mut g = m.lock().unwrap();
        g.1 = a.to_string();
        g.2 = b.to_string();
    }
}

fn match_logging() -> bool {
    MATCH_LOG.get().is_some()
}

/// `steps-match --shadow`: at every logged turn the player not moving also
/// searches the position (logged as the turn's "shadow"; never played), so
/// the explorer can set the two nets' beliefs side by side.
static MATCH_SHADOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_match_shadow(on: bool) {
    MATCH_SHADOW.store(on, Ordering::Relaxed);
}

/// `steps-match --openings`: pair i starts from entry i.
static MATCH_STARTS: std::sync::OnceLock<Vec<crate::game::MinionsState>> = std::sync::OnceLock::new();

pub fn set_match_starts(starts: Vec<crate::game::MinionsState>) {
    let _ = MATCH_STARTS.set(starts);
}

/// One turn by tier-1 lattice search.
pub fn lattice_turn(
    real: &crate::game::MinionsState,
    lp: &LatticePlay,
    salt: u64,
) -> anyhow::Result<crate::game::MinionsState> {
    lattice_turn_logged(real, lp, salt).map(|x| x.0)
}

/// [`lattice_turn`], and (when match games are logged) its root search and
/// the turn played, for the explorer.
pub fn lattice_turn_logged(
    real: &crate::game::MinionsState,
    lp: &LatticePlay,
    salt: u64,
) -> anyhow::Result<(crate::game::MinionsState, Option<serde_json::Value>)> {
    let start = StepState::begin(real, salt);
    // The mover kills the enemy necromancer: that ends the game.
    if !start.is_terminal() {
        if let Ok(end) = crate::lattice::necro_kill(&start) {
            let log = match_logging().then(|| serde_json::json!({
                "necro_kill": true,
                "played": crate::lattice_ui::turn_label(&start, &end),
            }));
            return Ok((end.real_base(), log));
        }
    }
    let mut t = crate::tier1::Tier1::new(start, lp.params.clone());
    let t0 = std::time::Instant::now();
    if lp.secs > 0.0 {
        let dl = t0 + std::time::Duration::from_secs_f64(lp.secs);
        t.run_until(lp.eval.as_ref(), u32::MAX, lp.inflight, Some(dl))?;
    } else {
        t.run_inflight(lp.eval.as_ref(), lp.n1, lp.inflight)?;
    }
    // STEPS_SLOW_TURN_SECS: report turns slower than that (match hangs).
    if let Some(lim) = std::env::var("STEPS_SLOW_TURN_SECS").ok().and_then(|v| v.parse::<f64>().ok()) {
        let el = t0.elapsed().as_secs_f64();
        if el > lim {
            let nodes: usize = t.nodes.iter().filter_map(|n| n.lattice.as_ref()).map(|l| l.nodes.len()).sum();
            eprintln!(
                "slow turn: {el:.1}s, turn {}, {} tier-1 nodes, {} root children, {nodes} lattice nodes; lazy {:?}\n{}",
                real.turn,
                t.nodes.len(),
                t.nodes[0].children.len(),
                crate::steps::LAZY_STATS.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect::<Vec<_>>(),
                real.render()
            );
        }
    }
    let b = t
        .best_child()
        .ok_or_else(|| anyhow::anyhow!("no turn found\n{}", real.render()))?;
    let log = match_logging().then(|| {
        use crate::lattice_ui::turn_label;
        let root = &t.nodes[0];
        let pri = t.priors(0);
        let mut kids: Vec<(usize, u32)> = root.children.iter().copied().enumerate().collect();
        kids.sort_by_key(|&(_, c)| std::cmp::Reverse(t.nodes[c as usize].n));
        // Featured games (--shadow) log every root turn, so each net's view of
        // the other's choice is there however widely the visits spread.
        let top = if MATCH_SHADOW.load(Ordering::Relaxed) { usize::MAX } else { 12 };
        let cands: Vec<serde_json::Value> = kids
            .iter()
            .take(top)
            .map(|&(i, c)| {
                let ch = &t.nodes[c as usize];
                serde_json::json!({
                    "label": turn_label(&root.state, &ch.state),
                    "n": ch.n,
                    "q": (ch.n > 0).then(|| 1.0 - ch.q()),
                    "prior": pri.get(i),
                    "picked": c == b,
                })
            })
            .collect();
        serde_json::json!({
            "visits": root.n,
            "kids": root.children.len(),
            "value": t.root_value(),
            "secs": t0.elapsed().as_secs_f64(),
            "cands": cands,
            "played": turn_label(&root.state, &t.nodes[b as usize].state),
        })
    });
    Ok((t.nodes[b as usize].state.real_base(), log))
}

impl Player {
    /// [`Self::turn`], with a lattice player's search for the match log.
    fn turn_logged(
        &self,
        real: &crate::game::MinionsState,
        salt: u64,
    ) -> anyhow::Result<(crate::game::MinionsState, Option<serde_json::Value>)> {
        match &self.lattice {
            Some(lp) if !self.solver => lattice_turn_logged(real, lp, salt),
            _ => Ok((self.turn(real, salt)?, None)),
        }
    }

    fn turn(
        &self,
        real: &crate::game::MinionsState,
        salt: u64,
    ) -> anyhow::Result<crate::game::MinionsState> {
        if self.solver {
            use playground_core::Game;
            let mut s = real.clone();
            while !crate::game::Minions::outcome(&s).is_terminal() && s.side == real.side {
                let mv = crate::solver::greedy_move(&s).best_move;
                s = crate::game::Minions::apply(&s, &mv);
            }
            Ok(s)
        } else if let Some(lp) = &self.lattice {
            lattice_turn(real, lp, salt)
        } else if self.expansions == 0 {
            greedy_turn(real, self.eval.as_ref(), salt)
        } else if let Some(tt) = &self.two_tier {
            two_tier_turn(real, self.eval.as_ref(), tt, self.expansions, salt).map(|t| t.next)
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
    opening_turns: u32,
    handicap: i16,
    even_bid_on: bool,
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
    // `opening_turns` > 0: that many random turns from the standard start
    // at `handicap`, then (`even_bid_on`) each side bids money to even the
    // position (applied per game, the two bids averaged); else the older
    // random openings / standard bid. Nets trained at one fixed handicap
    // cannot price money (gen 660's value moves 0.52 -> 0.47 over +-30 $),
    // so the bid is off by default.
    let starts: Vec<(Option<crate::game::MinionsState>, u64)> = (0..pairs)
        .map(|i| {
            if let Some(fixed) = MATCH_STARTS.get() {
                return (Some(fixed[i % fixed.len()].clone()), rng.gen());
            }
            let start = if opening_turns > 0 {
                Some(match a.lattice.as_ref().or(b.lattice.as_ref()) {
                    // Both sides' nets in turn, so neither picks every opening.
                    Some(_) => {
                        let p = if rng.gen::<bool>() { a } else { b };
                        let l = p.lattice.as_ref().or(a.lattice.as_ref()).or(b.lattice.as_ref()).expect("a lattice player");
                        sampled_opening(opening_turns, handicap, l.eval.as_ref(), &mut rng)
                            .unwrap_or_else(|_| short_opening(opening_turns, handicap, &mut rng))
                    }
                    None => short_opening(opening_turns, handicap, &mut rng),
                })
            } else {
                (opening_max != 0)
                    .then(|| crate::selfplay::random_opening(opening_max, opening_decay, &mut rng).0)
            };
            (start, rng.gen())
        })
        .collect();
    let bids_log: Mutex<Vec<i16>> = Mutex::new(Vec::new());
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
                        Some(s) if opening_turns > 0 && even_bid_on => {
                            let (ba, bb) = (
                                even_bid(a.eval.as_ref(), s, *salt)?,
                                even_bid(b.eval.as_ref(), s, *salt)?,
                            );
                            let x = (ba + bb) / 2;
                            if a_first {
                                bids_log.lock().unwrap().push(x);
                            }
                            with_money_bid(s, x)
                        }
                        Some(s) => s.clone(),
                        None => {
                            let (ba, bb) =
                                (bid(a.eval.as_ref(), *salt)?, bid(b.eval.as_ref(), *salt)?);
                            // The two bids' average (rounded toward zero) is the handicap.
                            crate::game::MinionsState::with_bid((ba + bb) / 2)
                        }
                    };
                    let mut turns = 0;
                    let mut log: Vec<serde_json::Value> = Vec::new();
                    while !crate::game::Minions::outcome(&real).is_terminal() && turns < 200 {
                        let a_moves = (real.side == 0) == a_first;
                        let p = if a_moves { a } else { b };
                        if match_logging() {
                            let shadow = if MATCH_SHADOW.load(Ordering::Relaxed) {
                                let o = if a_moves { b } else { a };
                                o.turn_logged(&real, salt ^ turns as u64)?.1
                            } else {
                                None
                            };
                            let (next, search) = p.turn_logged(&real, salt ^ turns as u64)?;
                            log.push(serde_json::json!({
                                "pos": crate::lattice_ui::board_hex(&real),
                                "turn": real.turn,
                                "side": real.side,
                                "mover": if a_moves { "a" } else { "b" },
                                "search": search,
                                "shadow": shadow,
                            }));
                            real = next;
                        } else {
                            real = p.turn(&real, salt ^ turns as u64)?;
                        }
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
                    if let Some(m) = MATCH_LOG.get() {
                        use std::io::Write;
                        let mut g = m.lock().unwrap();
                        let line = serde_json::json!({
                            "pair": i,
                            "a_first": a_first,
                            "a": g.1,
                            "b": g.2,
                            "winner": real.winner,
                            "final": crate::lattice_ui::board_hex(&real),
                            "final_turn": real.turn,
                            "turns": log,
                        });
                        writeln!(g.0, "{line}")?;
                        g.0.flush()?;
                    }
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
    let mut bl = bids_log.into_inner().unwrap();
    if !bl.is_empty() {
        bl.sort_unstable();
        eprintln!(
            "post-opening bids ($ to blue): min {} median {} max {}",
            bl[0],
            bl[bl.len() / 2],
            bl[bl.len() - 1]
        );
    }
    Ok((score / (2 * pairs) as f64, wdl))
}

/// The step player (seat A, `steps`) against the old sampled-AZ MCTS
/// (`seat` with `params`). Pairs share a bid start (each side's own bid,
/// averaged) with seats swapped. Each round, the old MCTS moves in every live
/// game where it is to move in one lockstep batch (as its own eval did), and
/// the step player moves in the rest on `threads` threads. Returns A's score
/// and (A wins, B wins, draws), plus mean turns of A's wins and losses.
pub fn play_vs_mcts(
    steps: &Player,
    seat: &crate::eval::Seat,
    params: &playground_core::sampled_mcts::SampledParams,
    pairs: usize,
    seed: u64,
    threads: usize,
    equal_time: bool,
) -> anyhow::Result<(f64, [u32; 3], (f64, f64))> {
    use crate::game::{Minions, MinionsState};
    use crate::turn::{realize, TurnSpec};
    use playground_core::{Game, Outcome};
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mcts_bid = crate::eval::bid_of(seat, params)?;
    struct G {
        real: MinionsState,
        steps_first: bool,
        salt: u64,
        turns: u32,
    }
    let mut games = Vec::with_capacity(2 * pairs);
    for _ in 0..pairs {
        let salt: u64 = rng.gen();
        let sb = f64::from(bid(steps.eval.as_ref(), salt)?);
        let b = match mcts_bid {
            Some(m) => ((sb + m) / 2.0).trunc() as i16,
            None => sb as i16,
        };
        for steps_first in [true, false] {
            games.push(G {
                real: MinionsState::with_bid(b),
                steps_first,
                salt,
                turns: 0,
            });
        }
    }
    eprintln!("bids: steps per pair, mcts {mcts_bid:?}");
    let done = |g: &G| Minions::outcome(&g.real).is_terminal() || g.turns >= 200;
    let mut round = 0u32;
    // Wall time and moves per side; each side has the whole machine while it moves.
    let (mut t_mcts, mut n_mcts, mut t_steps, mut n_steps) = (0f64, 0usize, 0f64, 0usize);
    let per = |t: f64, n: usize| t / n.max(1) as f64;
    // Under `equal_time` the step budget is reset every round to what the
    // MCTS spent per move that round, at the step side's measured rate.
    let mut player = steps.clone();
    let mut expanded = 0f64;
    let mut last_rate = 0f64;
    let mut round_mcts = 0f64;
    loop {
        let live: Vec<usize> = (0..games.len()).filter(|&i| !done(&games[i])).collect();
        if live.is_empty() {
            break;
        }
        if round % 10 == 0 {
            eprintln!(
                "round {round}: {} live; s/move mcts {:.3} ({n_mcts}), steps {:.3} ({n_steps}); {} expansions",
                live.len(),
                per(t_mcts, n_mcts),
                per(t_steps, n_steps),
                player.expansions
            );
        }
        let steps_moves = |g: &G| (g.real.side == 0) == g.steps_first;
        // The old MCTS, batched over its games.
        let mi: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&i| !steps_moves(&games[i]))
            .collect();
        if !mi.is_empty() {
            let t0 = std::time::Instant::now();
            let states: Vec<MinionsState> = mi.iter().map(|&i| games[i].real.clone()).collect();
            let plies: Vec<u32> = mi.iter().map(|&i| games[i].real.turn as u32).collect();
            let mut p = *params;
            p.seed = params.seed ^ (u64::from(round) << 32);
            let (specs, _, _) = crate::eval::pick_many(seat, &states, &plies, &p, false)?;
            for (&i, spec) in mi.iter().zip(specs) {
                let prev = games[i].real.clone();
                let next = realize(&prev, &spec).next;
                // A declaration that changes nothing would loop forever.
                games[i].real = if Minions::encode_state(&next) == Minions::encode_state(&prev) {
                    realize(&prev, &TurnSpec::empty()).next
                } else {
                    next
                };
                games[i].turns += 1;
            }
            round_mcts = t0.elapsed().as_secs_f64() / mi.len() as f64;
            t_mcts += t0.elapsed().as_secs_f64();
            n_mcts += mi.len();
        }
        // The step player, in parallel.
        // Games the old MCTS just finished are over.
        let si: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&i| !mi.contains(&i) && !done(&games[i]) && steps_moves(&games[i]))
            .collect();
        if equal_time && last_rate > 0.0 && round_mcts > 0.0 {
            // Last round's rate (cost per expansion grows with the budget),
            // and a correction that pulls the cumulative totals together.
            let gap = (per(t_mcts, n_mcts) / per(t_steps, n_steps)).clamp(0.5, 2.0);
            player.expansions = ((last_rate * round_mcts * gap) as u64).clamp(100, 200_000);
        }
        let t0 = std::time::Instant::now();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let out: std::sync::Mutex<Vec<(usize, anyhow::Result<MinionsState>)>> =
            std::sync::Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for _ in 0..threads.max(1) {
                sc.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = si.get(k) else {
                        return;
                    };
                    let g = &games[i];
                    let r = player.turn(&g.real, g.salt ^ u64::from(g.turns));
                    out.lock().unwrap().push((i, r));
                });
            }
        });
        for (i, r) in out.into_inner().unwrap() {
            games[i].real = r?;
            games[i].turns += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        if !si.is_empty() && dt > 0.0 {
            last_rate = (player.expansions * si.len() as u64) as f64 / dt;
        }
        t_steps += dt;
        n_steps += si.len();
        expanded += (player.expansions * si.len() as u64) as f64;
        round += 1;
    }
    eprintln!(
        "time per move (machine-amortized): mcts {:.3}s over {n_mcts}, steps {:.3}s over {n_steps}; mean {:.0} expansions",
        per(t_mcts, n_mcts),
        per(t_steps, n_steps),
        expanded / n_steps.max(1) as f64
    );
    let (mut score, mut wdl) = (0f64, [0u32; 3]);
    let (mut win_turns, mut loss_turns) = (Vec::new(), Vec::new());
    for g in &games {
        match Minions::outcome(&g.real) {
            r @ (Outcome::FirstWins | Outcome::SecondWins) => {
                if (r == Outcome::FirstWins) == g.steps_first {
                    wdl[0] += 1;
                    score += 1.0;
                    win_turns.push(f64::from(g.real.turn));
                } else {
                    wdl[1] += 1;
                    loss_turns.push(f64::from(g.real.turn));
                }
            }
            _ => {
                wdl[2] += 1;
                score += 0.5;
            }
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    Ok((
        score / games.len() as f64,
        wdl,
        (mean(&win_turns), mean(&loss_turns)),
    ))
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

/// `search-tree`: one tier-1 search of `real` (by `lp`), printed as a tree:
/// the root's `top` turns by visits, under each its `top2` most-visited
/// replies, then the most-visited line to its end with each position's money
/// and net worth. Values are the root mover's (yellow or blue, as `real`).
pub fn print_search_tree(
    real: &crate::game::MinionsState,
    lp: &LatticePlay,
    salt: u64,
    top: usize,
    top2: usize,
) -> anyhow::Result<()> {
    use crate::game::net_worth;
    use crate::lattice_ui::turn_label;
    let start = StepState::begin(real, salt);
    let mut t = crate::tier1::Tier1::new(start, lp.params.clone());
    t.run_inflight(lp.eval.as_ref(), lp.n1, lp.inflight)?;
    let me = real.side;
    let name = |s: u8| if s == 0 { "yellow" } else { "blue" };
    let money = |s: &crate::game::MinionsState| {
        format!(
            "${}/${} nw {}/{} (yellow/blue)",
            s.money[0],
            s.money[1],
            net_worth(s, 0),
            net_worth(s, 1)
        )
    };
    // Node c's value for the root mover: q() is the value for the side that
    // chose the turn into c, i.e. the mover at c's parent.
    let mine = |c: u32, depth: usize| {
        let q = 1.0 - t.nodes[c as usize].q();
        if depth % 2 == 1 { q } else { 1.0 - q }
    };
    let sorted = |p: u32| {
        let mut k: Vec<u32> = t.nodes[p as usize].children.clone();
        k.sort_by_key(|&c| std::cmp::Reverse(t.nodes[c as usize].n));
        k
    };
    println!(
        "turn {} {} to move, {}; root value for {} {:.3}, {} visits, {} root turns",
        real.turn,
        name(me),
        money(real),
        name(me),
        t.root_value(),
        t.nodes[0].n,
        t.nodes[0].children.len()
    );
    for &c in sorted(0).iter().take(top) {
        let ch = &t.nodes[c as usize];
        let cs = ch.state.real_base();
        println!(
            "  n {:4}  v {:.3}  {}  -> {}",
            ch.n,
            mine(c, 1),
            turn_label(&t.nodes[0].state, &ch.state),
            money(&cs)
        );
        for &g in sorted(c).iter().take(top2) {
            let gn = &t.nodes[g as usize];
            if gn.n == 0 {
                continue;
            }
            println!(
                "      n {:4}  v {:.3}  {}  -> {}",
                gn.n,
                mine(g, 2),
                turn_label(&ch.state, &gn.state),
                money(&gn.state.real_base())
            );
        }
    }
    println!("  most-visited line:");
    let (mut p, mut d) = (0u32, 0usize);
    loop {
        let Some(&c) = sorted(p).first() else { break };
        let cn = &t.nodes[c as usize];
        if cn.n == 0 {
            break;
        }
        d += 1;
        let cs = cn.state.real_base();
        println!(
            "    {d}. turn {} {} n {:4} v {:.3}  {} -> {}{}",
            t.nodes[p as usize].state.real_base().turn,
            name(t.nodes[p as usize].state.real_base().side),
            cn.n,
            mine(c, d),
            turn_label(&t.nodes[p as usize].state, &cn.state),
            money(&cs),
            if cs.winner != 0 { format!("  [ends: winner {}]", cs.winner) } else { String::new() }
        );
        p = c;
    }
    Ok(())
}
