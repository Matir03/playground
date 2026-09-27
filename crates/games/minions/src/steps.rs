//! A turn as a sequence of small, exhaustively enumerable decisions — the
//! action space of the best-first "steps" search (see `docs/minions-steps.md`).
//!
//! The whole-turn [`TurnSpec`] is built one variable at a time, in three stages:
//!
//! 1. **Fates** — every enemy piece, in a shuffled order fixed for the turn, is
//!    declared stay / kill / unsummon.
//! 2. **Moves** — every friendly unit, in a shuffled order fixed for the turn,
//!    picks one of its 37 destination offsets.
//! 3. **Spawns** — every open spawn cell, in a shuffled order, gets nothing or
//!    one affordable unit.
//!
//! Between stages 2 and 3 the attack part is realized, and a declaration the
//! realizer cannot carry out in full is dropped as a dead end, so every complete
//! path through the step tree is exactly one legal turn. Within stages 1 and 2 a
//! child is offered only if a relaxed attack-assignment bound (movement ignores
//! blocking) still admits every declared kill; the realizer is the exact check.
//!
//! Steps with a single legal option are applied automatically, so a node only
//! ever represents a real choice.
//!
//! Everything runs in the mover's canonical frame (the mover is side 0).

use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::assign;
use crate::bitboard::{self, bit, has, Bb, FULL};
use crate::game::{Minions, MinionsMove, MinionsState, PHASE_SPAWN};
use crate::map::{IS_GRAVEYARD, NO_CELL, NUM_CELLS};
use crate::turn::{
    kill_damage_against, realize, TurnSpec, MOVE_DEST, NO_MOVE, N_OFFSETS, TARGET_KILL,
    TARGET_STAY, TARGET_UNSUMMON,
};
use crate::units::{Attack, Unit, BUYABLE};
use playground_core::Game;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Fates,
    Moves,
    Spawns,
    /// The game is over.
    Terminal,
}

/// One turn under construction.
#[derive(Clone, Debug)]
pub struct StepState {
    /// Start-of-turn position, canonical (mover = side 0).
    pub base: MinionsState,
    /// Whether `base` is the real position rotated (the real mover is side 1).
    pub flipped: bool,
    pub spec: TurnSpec,
    pub enemies: Vec<u8>,
    pub friends: Vec<u8>,
    pub spawn_cells: Vec<u8>,
    pub stage: Stage,
    pub idx: usize,
    /// Destinations claimed by decided friendly moves.
    pub claimed: Bb,
    /// During `Spawns`: the spawn-phase position with the spawns so far applied.
    pub post: Option<MinionsState>,
    /// Turn-order salt; the same salt and position give the same orders.
    pub salt: u64,
}

/// Features of a node, for the value nets. Sparse indices are `plane * 49 + cell`.
pub const SPARSE_PLANES: usize = 17;
pub const SPARSE_DIM: usize = SPARSE_PLANES * NUM_CELLS;
pub const N_SCALARS: usize = 16;
pub const MAX_ACTIVE: usize = 96;

/// With kills declared, a move option is checked for a realizable completion
/// when at most this many units remain undecided after it.
const LOOKAHEAD_UNITS: usize = 2;
/// Option lists the completion check may expand before giving up (and
/// allowing the option, leaving any dead end to the search).
const LOOKAHEAD_BUDGET: u32 = 24;

/// Plane indices.
pub const P_UNITS: usize = 0; // 12 planes: mover units 0..6, opponent 6..12
pub const P_GRAVE: usize = 12;
pub const P_DEC_FRIEND: usize = 13;
pub const P_DEC_ENEMY: usize = 14;
pub const P_DEC_SPAWN: usize = 15;
pub const P_NEXT: usize = 16;

/// What the value net sees: the turn so far *projected* onto the board (fated
/// enemies removed, decided units at their destinations, spawns placed) plus
/// which variables are decided and which one is next.
#[derive(Clone, Debug)]
pub struct View {
    pub board: MinionsState,
    pub dec_friend: Bb,
    pub dec_enemy: Bb,
    pub dec_spawn: Bb,
    pub next: Bb,
    pub stage: Stage,
}

impl View {
    /// Sparse active indices and dense scalars.
    pub fn features(&self, active: &mut Vec<u16>, scalars: &mut [f32; N_SCALARS]) {
        active.clear();
        let b = &self.board;
        for c in 0..NUM_CELLS as u8 {
            if let Some((side, u)) = b.piece_at(c) {
                let p = P_UNITS + side as usize * 6 + u as usize;
                active.push((p * NUM_CELLS + c as usize) as u16);
            }
            if IS_GRAVEYARD[c as usize] {
                active.push((P_GRAVE * NUM_CELLS + c as usize) as u16);
            }
        }
        for (p, bb) in [
            (P_DEC_FRIEND, self.dec_friend),
            (P_DEC_ENEMY, self.dec_enemy),
            (P_DEC_SPAWN, self.dec_spawn),
            (P_NEXT, self.next),
        ] {
            for c in bitboard::cells(bb) {
                active.push((p * NUM_CELLS + c as usize) as u16);
            }
        }
        debug_assert!(active.len() <= MAX_ACTIVE);
        scalars[0] = b.money[0] as f32 / 10.0;
        scalars[1] = b.money[1] as f32 / 10.0;
        for side in 0..2 {
            for slot in 0..BUYABLE.len() {
                scalars[2 + side * 5 + slot] = b.reinforcements[side][slot] as f32;
            }
        }
        scalars[12] = crate::game::MAX_TURNS.saturating_sub(b.turn) as f32 / 30.0;
        scalars[13] = (self.stage == Stage::Fates) as u8 as f32;
        scalars[14] = (self.stage == Stage::Moves) as u8 as f32;
        scalars[15] = (self.stage == Stage::Spawns) as u8 as f32;
    }
}

fn side_bb(s: &MinionsState, side: u8) -> Bb {
    let mut b = 0;
    for c in 0..NUM_CELLS as u8 {
        if s.piece_at(c).is_some_and(|(sd, _)| sd == side) {
            b |= bit(c);
        }
    }
    b
}

impl StepState {
    /// The step tree of the turn starting at `real` (any seat to move).
    pub fn begin(real: &MinionsState, salt: u64) -> Self {
        let flipped = real.side == 1;
        let base = real.canonical();
        let mut s = Self {
            base,
            flipped,
            spec: TurnSpec::empty(),
            enemies: Vec::new(),
            friends: Vec::new(),
            spawn_cells: Vec::new(),
            stage: Stage::Fates,
            idx: 0,
            claimed: 0,
            post: None,
            salt,
        };
        if Minions::outcome(&s.base).is_terminal() {
            s.stage = Stage::Terminal;
            return s;
        }
        let mut rng = s.rng(1);
        s.enemies = bitboard::cells(side_bb(&s.base, 1)).collect();
        s.enemies.shuffle(&mut rng);
        s.friends = bitboard::cells(side_bb(&s.base, 0)).collect();
        s.friends.shuffle(&mut rng);
        s.settle();
        s
    }

    fn rng(&self, tag: u64) -> rand_chacha::ChaCha8Rng {
        let k = self.base.key();
        let seed = (k as u64) ^ ((k >> 64) as u64).rotate_left(17) ^ self.salt ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        rand_chacha::ChaCha8Rng::seed_from_u64(seed)
    }

    /// The real position this turn started from.
    pub fn real_base(&self) -> MinionsState {
        if self.flipped { self.base.rotated() } else { self.base.clone() }
    }

    /// Advance past empty stages and auto-apply single-option steps. May move
    /// on to the next turn (a new `StepState` in place), or to `Terminal`.
    fn settle(&mut self) {
        loop {
            match self.stage {
                Stage::Terminal => return,
                Stage::Fates if self.idx >= self.enemies.len() => {
                    self.stage = Stage::Moves;
                    self.idx = 0;
                }
                Stage::Moves if self.idx >= self.friends.len() => {
                    if !self.finish_moves() {
                        // Only reachable if the caller applied an unchecked
                        // step; `children` never offers one.
                        debug_assert!(false, "settled into an unrealizable declaration");
                        self.stage = Stage::Terminal;
                        return;
                    }
                }
                Stage::Spawns if self.idx >= self.spawn_cells.len() => {
                    let post = self.post.take().expect("spawn phase state");
                    let next = Minions::apply(&post, &MinionsMove::EndPhase);
                    self.next_turn(next);
                }
                _ => {
                    let ch = self.children();
                    if ch.len() == 1 {
                        self.apply_raw(ch[0]);
                    } else {
                        return;
                    }
                }
            }
        }
    }

    /// Start the next turn from `next` (in this turn's canonical frame).
    fn next_turn(&mut self, next: MinionsState) {
        let real = if self.flipped { next.rotated() } else { next };
        *self = StepState::begin(&real, self.salt);
    }

    /// Realize the fates and moves. On success moves to `Spawns` (or on to the
    /// next turn / `Terminal`); on failure returns false and leaves `self` as is.
    fn finish_moves(&mut self) -> bool {
        let Some(post) = attack_part(&self.base, &self.spec) else { return false };
        if post.winner != 0 || post.phase != PHASE_SPAWN {
            // Game over, or turn 0 (no spawn phase): `post` is already the next turn.
            if post.winner != 0 {
                let real = if self.flipped { post.rotated() } else { post };
                *self = StepState::begin(&real, self.salt);
            } else {
                self.next_turn(post);
            }
            return true;
        }
        let mut cells: Vec<u8> = bitboard::cells(post.spawn_bb(0)).collect();
        cells.shuffle(&mut self.rng(2));
        self.spawn_cells = cells;
        self.post = Some(post);
        self.stage = Stage::Spawns;
        self.idx = 0;
        true
    }

    pub fn is_terminal(&self) -> bool {
        self.stage == Stage::Terminal
    }

    /// The terminal outcome for the side to move at `base` (1 win, 0 loss, 0.5 draw).
    pub fn terminal_value(&self) -> f32 {
        match self.base.winner {
            1 => 1.0, // canonical side 0 = the mover of `base`
            2 => 0.0,
            _ => 0.5,
        }
    }

    /// Legal options for the current step. Never empty outside `Terminal`.
    pub fn children(&self) -> Vec<u8> {
        match self.stage {
            Stage::Terminal => Vec::new(),
            Stage::Fates => self.fate_options(),
            Stage::Moves => self.move_options(true),
            Stage::Spawns => self.spawn_options(),
        }
    }

    /// Apply one option from [`Self::children`], then settle.
    pub fn apply(&mut self, v: u8) {
        self.apply_raw(v);
        self.settle();
    }

    fn apply_raw(&mut self, v: u8) {
        match self.stage {
            Stage::Fates => {
                let e = self.enemies[self.idx];
                self.spec.targets[e as usize] = v;
                self.idx += 1;
            }
            Stage::Moves => {
                let c = self.friends[self.idx];
                self.spec.moves[c as usize] = v;
                self.claimed |= bit(MOVE_DEST[c as usize][v as usize]);
                self.idx += 1;
            }
            Stage::Spawns => {
                let at = self.spawn_cells[self.idx];
                if v != 0 {
                    let post = self.post.as_mut().expect("spawn phase state");
                    let unit = Unit::from_index(v).expect("spawn code");
                    let slot = unit.reinforcement_slot().expect("buyable");
                    if post.reinforcements[0][slot] == 0 {
                        *post = Minions::apply(post, &MinionsMove::Buy(unit));
                    }
                    *post = Minions::apply(post, &MinionsMove::Spawn { unit, at });
                    self.spec.spawns[at as usize] = v;
                }
                self.idx += 1;
            }
            Stage::Terminal => {}
        }
    }

    /// Whether the side to move at this node differs from `other`'s.
    pub fn mover_flipped_vs(&self, other: &StepState) -> bool {
        self.flipped != other.flipped
    }

    // ── Fates ────────────────────────────────────────────────────────────────

    fn fate_options(&self) -> Vec<u8> {
        let e = self.enemies[self.idx];
        let (_, t) = self.base.piece_at(e).expect("enemy");
        let mut out = vec![TARGET_STAY];
        for v in [TARGET_KILL, TARGET_UNSUMMON] {
            if v == TARGET_UNSUMMON && t.stats().persistent {
                continue;
            }
            let mut spec = self.spec;
            spec.targets[e as usize] = v;
            if relaxed_feasible(&self.base, &spec, &[]) {
                out.push(v);
            }
        }
        out
    }

    // ── Moves ────────────────────────────────────────────────────────────────

    fn move_options(&self, lookahead: bool) -> Vec<u8> {
        let c = self.friends[self.idx];
        let (_, u) = self.base.piece_at(c).expect("friend");
        let st = u.stats();
        let enemies = side_bb(&self.base, 1);
        let mut staying_enemies: Bb = 0;
        for e in bitboard::cells(enemies) {
            if self.spec.targets[e as usize] == TARGET_STAY {
                staying_enemies |= bit(e);
            }
        }
        // Friendly origins already committed to staying.
        let mut stayers: Bb = 0;
        for &f in &self.friends[..self.idx] {
            if self.spec.moves[f as usize] == 0 {
                stayers |= bit(f);
            }
        }
        // Relaxed passability: enemies declared dead are assumed gone.
        let pass = if st.flying { FULL } else { FULL & !staying_enemies };
        let reach = if self.base.can_move_piece(c) {
            bitboard::reachable(bit(c), st.speed, pass)
        } else {
            bit(c)
        };
        let blocked = self.claimed | staying_enemies | stayers;
        let last = self.idx + 1 == self.friends.len();
        let kills = self.spec.targets.iter().any(|&t| t != TARGET_STAY);
        let decided: Vec<u8> = self.friends[..self.idx].to_vec();
        let mut out = Vec::new();
        for k in 0..N_OFFSETS as u8 {
            let d = MOVE_DEST[c as usize][k as usize];
            if d == NO_CELL || !has(reach, d) {
                continue;
            }
            if k == 0 {
                if has(self.claimed, c) {
                    continue; // someone is moving onto this unit's origin
                }
            } else if has(blocked, d) || self.closes_cycle(c, d) {
                continue;
            }
            let mut spec = self.spec;
            spec.moves[c as usize] = k;
            if last {
                if attack_part(&self.base, &spec).is_none() {
                    continue;
                }
            } else {
                if !self.movers_completable(&spec, self.claimed | bit(d), c, k) {
                    continue;
                }
                let mut dec = decided.clone();
                dec.push(c);
                if !relaxed_feasible(&self.base, &spec, &dec) {
                    continue;
                }
                let remaining = self.friends.len() - self.idx - 1;
                if lookahead && kills && remaining <= LOOKAHEAD_UNITS {
                    let mut t = self.clone();
                    t.apply_raw(k);
                    let mut budget = LOOKAHEAD_BUDGET;
                    if t.completable(&mut budget) == Some(false) {
                        continue;
                    }
                }
            }
            out.push(k);
        }
        out
    }

    /// After tentatively deciding `c` (offset `k`), can every undecided unit
    /// whose origin is now claimed still get out of the way? Each such unit needs
    /// its own path to a hex that ends the turn empty, possibly via the origin
    /// of another undecided unit (which then has to move on in turn). Exact for
    /// the movement constraints as a vertex-disjoint-paths max flow; kills are
    /// not modelled here.
    fn movers_completable(&self, spec: &TurnSpec, claimed: Bb, c: u8, k: u8) -> bool {
        let undecided = &self.friends[self.idx + 1..];
        if undecided.iter().all(|&u| !has(claimed, u)) {
            return true;
        }
        let mut staying_enemies: Bb = 0;
        let mut enemies: Bb = 0;
        for e in 0..NUM_CELLS as u8 {
            if self.base.piece_at(e).is_some_and(|(sd, _)| sd == 1) {
                enemies |= bit(e);
                if spec.targets[e as usize] == TARGET_STAY {
                    staying_enemies |= bit(e);
                }
            }
        }
        // Decided units' final cells and their vacated origins.
        let mut stayers: Bb = 0;
        let mut friend_now: Bb = 0;
        for &f in self.friends.iter() {
            friend_now |= bit(f);
        }
        for &f in self.friends[..self.idx].iter().chain(std::iter::once(&c)) {
            let kk = if f == c { k } else { spec.moves[f as usize] };
            if kk == 0 {
                stayers |= bit(f);
            }
        }
        let und_bb: Bb = undecided.iter().fold(0, |b, &u| b | bit(u));
        // A cell is a sink if nothing will stand on it at the end: empty now,
        // a removed enemy, or a decided mover's vacated origin -- and unclaimed.
        let blocked = claimed | staying_enemies | stayers;
        let sink = |x: u8| -> bool { !has(blocked, x) && !has(und_bb, x) };
        // Graph: node per undecided unit (split in/out) and per cell (split).
        // ids: S=0, T=1, unit u (index i): in 2+2i, out 3+2i; cell x: in 2+2n+2x, out 3+2n+2x.
        let n = undecided.len();
        let unit_idx = |x: u8| undecided.iter().position(|&u| u == x);
        let nn = 2 + 2 * n + 2 * NUM_CELLS;
        let mut g = Flow::new(nn);
        let (uin, uout) = (|i: usize| 2 + 2 * i, |i: usize| 3 + 2 * i);
        let (cin, cout) = (|x: u8| 2 + 2 * n + 2 * x as usize, |x: u8| 3 + 2 * n + 2 * x as usize);
        let mut must = 0;
        for (i, &u) in undecided.iter().enumerate() {
            g.add(uin(i), uout(i), 1);
            if has(claimed, u) {
                g.add(0, uin(i), 1);
                must += 1;
            }
            let (_, unit) = self.base.piece_at(u).expect("friend");
            let st = unit.stats();
            if !self.base.can_move_piece(u) {
                continue;
            }
            let pass = if st.flying { FULL } else { FULL & !staying_enemies };
            let reach = bitboard::reachable(bit(u), st.speed, pass) & !bit(u);
            for x in bitboard::cells(reach) {
                if has(blocked, x) {
                    continue;
                }
                g.add(uout(i), cin(x), 1);
            }
        }
        for x in 0..NUM_CELLS as u8 {
            g.add(cin(x), cout(x), 1);
            if sink(x) {
                g.add(cout(x), 1, 1);
            } else if let Some(j) = unit_idx(x) {
                if !has(claimed, x) {
                    g.add(cout(x), uin(j), 1); // push that unit on in turn
                }
            }
        }
        let _ = (enemies, friend_now);
        g.max_flow(0, 1, must) >= must
    }

    /// Whether the remaining moves can be completed into a realizable
    /// declaration: depth-first over [`Self::move_options`], stopping at the
    /// first success. `None` when `budget` (option lists expanded) runs out.
    fn completable(&self, budget: &mut u32) -> Option<bool> {
        if self.stage != Stage::Moves || self.idx >= self.friends.len() {
            return Some(true);
        }
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let opts = self.move_options(false);
        if self.idx + 1 == self.friends.len() {
            return Some(!opts.is_empty()); // the last unit's options are realize-checked
        }
        let mut unknown = false;
        for k in opts {
            let mut t = self.clone();
            t.apply_raw(k);
            match t.completable(budget) {
                Some(true) => return Some(true),
                None => unknown = true,
                Some(false) => {}
            }
        }
        if unknown { None } else { Some(false) }
    }

    /// Whether moving `c` to `d` closes a cycle of decided moves (a0->b0 with
    /// b0->a0 already declared): units only move onto empty hexes, so a cycle
    /// can never be carried out.
    fn closes_cycle(&self, c: u8, mut d: u8) -> bool {
        for _ in 0..self.friends.len() {
            if d == c {
                return true;
            }
            let Some(&f) = self.friends[..self.idx].iter().find(|&&f| f == d) else {
                return false;
            };
            let k = self.spec.moves[f as usize];
            if k == 0 {
                return false;
            }
            d = MOVE_DEST[f as usize][k as usize];
        }
        false
    }

    // ── Spawns ───────────────────────────────────────────────────────────────

    fn spawn_options(&self) -> Vec<u8> {
        let post = self.post.as_ref().expect("spawn phase state");
        let mut out = vec![0u8];
        for &u in BUYABLE.iter() {
            let slot = u.reinforcement_slot().expect("buyable");
            if post.reinforcements[0][slot] > 0 || post.money[0] >= u.stats().cost {
                out.push(u as u8);
            }
        }
        out
    }

    // ── Features ─────────────────────────────────────────────────────────────

    /// The projected position and decision markers.
    pub fn view(&self) -> View {
        let mut v = View {
            board: self.base.clone(),
            dec_friend: 0,
            dec_enemy: 0,
            dec_spawn: 0,
            next: 0,
            stage: self.stage,
        };
        match self.stage {
            Stage::Terminal => {}
            Stage::Fates | Stage::Moves => {
                let b = &mut v.board;
                let n_fates = if self.stage == Stage::Fates { self.idx } else { self.enemies.len() };
                for &e in &self.enemies[..n_fates] {
                    v.dec_enemy |= bit(e);
                    let t = self.spec.targets[e as usize];
                    if t == TARGET_STAY {
                        continue;
                    }
                    let (side, unit) = b.piece_at(e).expect("enemy");
                    b.cells[e as usize] = 0;
                    b.pstate[e as usize] = 0;
                    if t == TARGET_UNSUMMON {
                        if let Some(slot) = unit.reinforcement_slot() {
                            b.reinforcements[side as usize][slot] =
                                b.reinforcements[side as usize][slot].saturating_add(1);
                        }
                    } else {
                        b.money[side as usize] += unit.stats().rebate;
                    }
                }
                if self.stage == Stage::Moves {
                    // Lift decided movers first, then place them, so moves onto
                    // another mover's origin project correctly.
                    let movers: Vec<(u8, u8, u8)> = self.friends[..self.idx]
                        .iter()
                        .map(|&f| {
                            let d = MOVE_DEST[f as usize][self.spec.moves[f as usize] as usize];
                            (f, d, b.cells[f as usize])
                        })
                        .collect();
                    for &(f, _, _) in &movers {
                        b.cells[f as usize] = 0;
                    }
                    for &(_, d, code) in &movers {
                        b.cells[d as usize] = code;
                        v.dec_friend |= bit(d);
                    }
                    v.next = bit(self.friends[self.idx]);
                } else {
                    v.next = bit(self.enemies[self.idx]);
                }
            }
            Stage::Spawns => {
                let post = self.post.as_ref().expect("spawn phase state");
                v.board = post.clone();
                v.dec_friend = side_bb(post, 0);
                v.dec_enemy = side_bb(post, 1);
                for &at in &self.spawn_cells[..self.idx] {
                    v.dec_spawn |= bit(at);
                }
                v.next = bit(self.spawn_cells[self.idx]);
            }
        }
        v
    }
}

/// A tiny unit-capacity max flow (augmenting DFS), for [`StepState::movers_completable`].
struct Flow {
    head: Vec<u32>,
    to: Vec<u32>,
    cap: Vec<i8>,
    next: Vec<u32>,
}

impl Flow {
    fn new(n: usize) -> Self {
        Self { head: vec![u32::MAX; n], to: Vec::new(), cap: Vec::new(), next: Vec::new() }
    }

    fn add(&mut self, a: usize, b: usize, c: i8) {
        for (x, y, cc) in [(a, b, c), (b, a, 0)] {
            self.to.push(y as u32);
            self.cap.push(cc);
            self.next.push(self.head[x]);
            self.head[x] = (self.to.len() - 1) as u32;
        }
    }

    fn max_flow(&mut self, s: usize, t: usize, want: usize) -> usize {
        let mut flow = 0;
        let mut seen = vec![false; self.head.len()];
        while flow < want {
            seen.iter_mut().for_each(|v| *v = false);
            if !self.augment(s, t, &mut seen) {
                break;
            }
            flow += 1;
        }
        flow
    }

    fn augment(&mut self, v: usize, t: usize, seen: &mut [bool]) -> bool {
        if v == t {
            return true;
        }
        seen[v] = true;
        let mut e = self.head[v];
        while e != u32::MAX {
            let ei = e as usize;
            let w = self.to[ei] as usize;
            if self.cap[ei] > 0 && !seen[w] && self.augment(w, t, seen) {
                self.cap[ei] -= 1;
                self.cap[ei ^ 1] += 1;
                return true;
            }
            e = self.next[ei];
        }
        false
    }
}

/// Realize the fates and moves of `spec` from `base` (spawns ignored). Returns
/// the position after the attack phase closes — the spawn-phase position, or
/// the next turn on turn 0, or a finished game — or `None` if any declared
/// move or kill did not happen.
pub fn attack_part(base: &MinionsState, spec: &TurnSpec) -> Option<MinionsState> {
    let mut sp = *spec;
    sp.spawns = [0; NUM_CELLS];
    let r = realize(base, &sp);
    let mut s = base.clone();
    for a in &r.actions {
        s = Minions::apply(&s, a);
        if *a == MinionsMove::EndPhase || s.winner != 0 {
            break;
        }
    }
    if s.winner != 0 {
        return Some(s); // a won game needs nothing else declared to happen
    }
    let f = r.fidelity;
    (f.done_moves == f.declared_moves && f.done_kills == f.declared_kills).then_some(s)
}

/// A necessary condition for the declared kills to be achievable: every
/// friendly unit is placed at its decided destination (`decided`) or, if
/// undecided, may attack anything within its speed plus range of its origin
/// (ignoring blocking). Checked with the assignment deficiency bound plus the
/// greedy/complete solver on this relaxed reach.
fn relaxed_feasible(base: &MinionsState, spec: &TurnSpec, decided: &[u8]) -> bool {
    let me = 0u8;
    // Where things stand: staying enemies block ground movement and hexes;
    // decided destinations and decided stayers take their hexes.
    let mut staying_enemies: Bb = 0;
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != me) && spec.targets[e as usize] == TARGET_STAY {
            staying_enemies |= bit(e);
        }
    }
    let mut taken: Bb = 0; // hexes decided units end on
    for &f in decided {
        let k = spec.moves[f as usize];
        taken |= bit(if k == NO_MOVE { f } else { MOVE_DEST[f as usize][k as usize] });
    }
    let mut atk_cells: Vec<(u8, Unit, Bb)> = Vec::new(); // (origin, unit, possible end hexes)
    for c in 0..NUM_CELLS as u8 {
        let Some((sd, u)) = base.piece_at(c) else { continue };
        if sd != me || !base.can_attack_piece(c, u) {
            continue;
        }
        let st = u.stats();
        let ends: Bb = if decided.contains(&c) {
            let k = spec.moves[c as usize];
            let d = if k == NO_MOVE { c } else { MOVE_DEST[c as usize][k as usize] };
            if st.lumbering && d != c { 0 } else { bit(d) }
        } else {
            let stay = if has(taken, c) { 0 } else { bit(c) };
            if st.lumbering || !base.can_move_piece(c) {
                stay
            } else {
                let pass = if st.flying { FULL } else { FULL & !staying_enemies };
                let moves = bitboard::reachable(bit(c), st.speed, pass) & !bit(c) & !taken & !staying_enemies;
                stay | moves
            }
        };
        if ends != 0 {
            atk_cells.push((c, u, ends));
        }
    }
    let can_hit = |_o: u8, u: Unit, ends: Bb, e: u8| -> bool {
        let ring = bitboard::reachable(bit(e), u.stats().range, FULL);
        ends & ring != 0
    };
    let mut targets = Vec::new();
    for e in 0..NUM_CELLS as u8 {
        let t = spec.targets[e as usize];
        if t == TARGET_STAY {
            continue;
        }
        let Some((sd, tu)) = base.piece_at(e) else { continue };
        if sd == me {
            continue;
        }
        let mut reach = Vec::new();
        for (i, &(o, u, ends)) in atk_cells.iter().enumerate() {
            if !can_hit(o, u, ends, e) {
                continue;
            }
            let dmg = if t == TARGET_UNSUMMON {
                u32::from(matches!(u.stats().attack, Attack::Unsummon))
            } else {
                kill_damage_against(u, tu) as u32
            };
            if dmg > 0 {
                reach.push((i, dmg));
            }
        }
        let need = if t == TARGET_UNSUMMON {
            1
        } else {
            tu.stats().defense.saturating_sub(base.damage_on(e)) as u32
        };
        if reach.is_empty() {
            return false;
        }
        targets.push(assign::Target { cell: e, need, reach });
    }
    if targets.is_empty() {
        return true;
    }
    let attackers: Vec<assign::Attacker> =
        atk_cells.iter().map(|&(c, _, _)| assign::Attacker { cell: c, attacks: 1 }).collect();
    if assign::hall_violation(&targets, &attackers).is_some() {
        return false;
    }
    if assign::greedy(&targets, &attackers).iter().all(|p| p.is_some()) {
        return true;
    }
    !matches!(
        assign::solve(&targets, &attackers, crate::turn::SOLVER_BUDGET),
        assign::Assignment::Infeasible
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Walk random step paths through whole games. Every completed path must
    /// be a legal turn that hands over the move, and dead ends (a step with no
    /// option) must be rare; the search prunes them. Restarts the turn on one.
    #[test]
    fn random_step_paths_are_legal_turns() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut turns, mut options, mut steps, mut dead, mut stuck) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut dead_at = [0usize; 3];
        let mut cls = [0usize; 4];
        for g in 0..40u64 {
            let bid = rng.gen_range(crate::game::BID_MIN..=crate::game::BID_MAX);
            let mut real = MinionsState::with_bid(bid);
            for _ in 0..40 {
                if Minions::outcome(&real).is_terminal() {
                    break;
                }
                let start = StepState::begin(&real, g);
                let mut done = None;
                for _try in 0..50 {
                    let mut s = start.clone();
                    let ok = loop {
                        if s.is_terminal() || s.flipped != start.flipped || s.base.turn != start.base.turn {
                            break true;
                        }
                        let ch = s.children();
                        if ch.is_empty() {
                            if s.stage == Stage::Moves {
                                let last = s.idx + 1 == s.friends.len();
                                let kills = s.spec.targets.iter().any(|&t| t != TARGET_STAY);
                                cls[usize::from(last) * 2 + usize::from(kills)] += 1;
                            }
                            dead += 1;
                            dead_at[s.stage as usize] += 1;
                            break false;
                        }
                        options += ch.len();
                        steps += 1;
                        s.apply(ch[rng.gen_range(0..ch.len())]);
                    };
                    if ok {
                        done = Some(s);
                        break;
                    }
                }
                let Some(s) = done else {
                    stuck += 1;
                    break;
                };
                let next_real = s.real_base();
                assert!(
                    Minions::outcome(&next_real).is_terminal() || next_real.side != real.side,
                    "a turn must hand over the move"
                );
                real = next_real;
                turns += 1;
            }
        }
        eprintln!(
            "{turns} turns, {steps} choice steps, {:.1} options/step, {dead} dead ends (fates/moves/spawns {dead_at:?}), {stuck} turns never finished; [notlast/nokill, notlast/kill, last/nokill, last/kill] {cls:?}",
            options as f64 / steps as f64
        );
        assert!(turns > 200, "played {turns} turns");
        // Random walks declare a kill on ~2/3 of enemies, far more than search
        // does; this bounds the relaxation's slack, the search prunes the rest.
        assert!(dead < 2 * turns, "dead ends too common: {dead} in {turns} turns");
    }

    /// The view's sparse features stay in range and within MAX_ACTIVE.
    #[test]
    fn features_are_in_range() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let mut real = MinionsState::with_bid(0);
        let mut act = Vec::new();
        let mut sc = [0f32; N_SCALARS];
        for _ in 0..30 {
            if Minions::outcome(&real).is_terminal() {
                break;
            }
            let mut s = StepState::begin(&real, 1);
            let f0 = s.flipped;
            while !s.is_terminal() && s.flipped == f0 {
                s.view().features(&mut act, &mut sc);
                assert!(act.len() <= MAX_ACTIVE);
                assert!(act.iter().all(|&i| (i as usize) < SPARSE_DIM));
                let ch = s.children();
                if ch.is_empty() {
                    break;
                }
                s.apply(ch[rng.gen_range(0..ch.len())]);
            }
            if s.flipped == f0 && !s.is_terminal() {
                break; // dead end; enough coverage
            }
            real = s.real_base();
        }
    }
}
