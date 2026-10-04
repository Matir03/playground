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
                done: fin.done,
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
        done: fin.done,
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
}

/// One turn by tier-1 lattice search.
pub fn lattice_turn(
    real: &crate::game::MinionsState,
    lp: &LatticePlay,
    salt: u64,
) -> anyhow::Result<crate::game::MinionsState> {
    let start = StepState::begin(real, salt);
    // The mover kills the enemy necromancer: that ends the game.
    if !start.is_terminal() {
        if let Ok(end) = crate::lattice::necro_kill(&start) {
            return Ok(end.real_base());
        }
    }
    let mut t = crate::tier1::Tier1::new(start, lp.params.clone());
    t.run_inflight(lp.eval.as_ref(), lp.n1, lp.inflight)?;
    let b = t
        .best_child()
        .ok_or_else(|| anyhow::anyhow!("no turn found\n{}", real.render()))?;
    Ok(t.nodes[b as usize].state.real_base())
}

impl Player {
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
