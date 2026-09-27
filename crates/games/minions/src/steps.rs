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
//! Steps with a single legal option are collapsed at expansion time
//! ([`StepState::collapse_forced`]), never across a turn boundary.
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
    /// The complete declaration of the turn that led to this one, in that
    /// turn's canonical frame (empty at a tree's start): what
    /// [`turn_actions`] turns back into atomic moves.
    pub done: TurnSpec,
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
    /// The current step's options and, parallel, a witness for each: a
    /// complete fates+moves declaration that extends the decisions with that
    /// option and realizes in full. Computed once, cached.
    opts: Option<(Vec<u8>, Vec<TurnSpec>)>,
    /// A realizable completion of the decisions so far (see `opts`). Every
    /// option offered has one, so the step tree has no dead ends.
    witness: TurnSpec,
}

/// Features of a node, for the value nets. Sparse indices are `plane * 49 + cell`.
pub const SPARSE_PLANES: usize = 25;
pub const SPARSE_DIM: usize = SPARSE_PLANES * NUM_CELLS;
pub const N_SCALARS: usize = 16;
pub const MAX_ACTIVE: usize = 96;

/// Plane indices.
pub const P_UNITS: usize = 0; // 12 planes: mover units 0..6, opponent 6..12
pub const P_GRAVE: usize = 12;
pub const P_DEC_FRIEND: usize = 13;
pub const P_DEC_ENEMY: usize = 14;
pub const P_DEC_SPAWN: usize = 15;
pub const P_NEXT: usize = 16;
/// Hexes whose enemy is fated this turn: killed, unsummoned, and a one-hot of
/// the unit that stood there (6 planes, by `Unit` index). The projected board
/// has removed it, but which unit it was still constrains the rest of the
/// turn (who attacked it, which hexes it blocked) and cannot be read back.
pub const P_KILLED: usize = 17;
pub const P_UNSUMMONED: usize = 18;
pub const P_GONE_UNIT: usize = 19;

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
    /// Fated enemies (see [`P_KILLED`]): hexes killed, hexes unsummoned, and
    /// per unit type the hexes such a unit was removed from.
    pub killed: Bb,
    pub unsummoned: Bb,
    pub gone_units: [Bb; 6],
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
            (P_KILLED, self.killed),
            (P_UNSUMMONED, self.unsummoned),
        ] {
            for c in bitboard::cells(bb) {
                active.push((p * NUM_CELLS + c as usize) as u16);
            }
        }
        for (u, &bb) in self.gone_units.iter().enumerate() {
            for c in bitboard::cells(bb) {
                active.push(((P_GONE_UNIT + u) * NUM_CELLS + c as usize) as u16);
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

/// Whether the friendly unit on `c` could attack some enemy in `enemies` this
/// turn, optimistically: from any hex within its speed ignoring blockers
/// (from `c` only if lumbering or unable to move).
fn could_attack(base: &MinionsState, c: u8, enemies: Bb) -> bool {
    let Some((_, u)) = base.piece_at(c) else {
        return false;
    };
    if !base.can_attack_piece(c, u) {
        return false;
    }
    let st = u.stats();
    let from = if st.lumbering || !base.can_move_piece(c) {
        bit(c)
    } else {
        bitboard::reachable(bit(c), st.speed, FULL)
    };
    bitboard::cells(enemies)
        .any(|e| from & bitboard::reachable(bit(e), st.range, FULL) & !bit(e) != 0)
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
            done: TurnSpec::empty(),
            enemies: Vec::new(),
            friends: Vec::new(),
            spawn_cells: Vec::new(),
            stage: Stage::Fates,
            idx: 0,
            claimed: 0,
            post: None,
            salt,
            opts: None,
            witness: TurnSpec::empty(),
        };
        if Minions::outcome(&s.base).is_terminal() {
            s.stage = Stage::Terminal;
            return s;
        }
        let mut rng = s.rng(1);
        s.enemies = bitboard::cells(side_bb(&s.base, 1)).collect();
        s.enemies.shuffle(&mut rng);
        // The necromancer's fate first: "kill it" ends the game, and "spare
        // it" keeps every later witness from winning its way past the rest.
        if let Some(i) = s.enemies.iter().position(|&e| {
            s.base
                .piece_at(e)
                .is_some_and(|(_, u)| u.stats().necromancer)
        }) {
            let n = s.enemies.remove(i);
            s.enemies.insert(0, n);
        }
        // Units that could in theory attack some enemy this turn decide first
        // (their moves constrain the rest most); shuffled within each group.
        let enemies = side_bb(&s.base, 1);
        let (mut threats, mut others): (Vec<u8>, Vec<u8>) =
            bitboard::cells(side_bb(&s.base, 0)).partition(|&c| could_attack(&s.base, c, enemies));
        threats.shuffle(&mut rng);
        others.shuffle(&mut rng);
        threats.extend(others);
        s.friends = threats;
        s.settle();
        s
    }

    fn rng(&self, tag: u64) -> rand_chacha::ChaCha8Rng {
        let k = self.base.key();
        let seed = (k as u64)
            ^ ((k >> 64) as u64).rotate_left(17)
            ^ self.salt
            ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        rand_chacha::ChaCha8Rng::seed_from_u64(seed)
    }

    /// The real position this turn started from.
    pub fn real_base(&self) -> MinionsState {
        if self.flipped {
            self.base.rotated()
        } else {
            self.base.clone()
        }
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
                _ => return,
            }
        }
    }

    /// Start the next turn from `next` (in this turn's canonical frame).
    fn next_turn(&mut self, next: MinionsState) {
        let real = if self.flipped { next.rotated() } else { next };
        let done = self.spec;
        *self = StepState::begin(&real, self.salt);
        self.done = done;
    }

    /// Realize the fates and moves. On success moves to `Spawns` (or on to the
    /// next turn / `Terminal`); on failure returns false and leaves `self` as is.
    fn finish_moves(&mut self) -> bool {
        let Some(post) = attack_part(&self.base, &self.spec) else {
            return false;
        };
        if post.winner != 0 || post.phase != PHASE_SPAWN {
            // Game over, or turn 0 (no spawn phase): `post` is already the next turn.
            if post.winner != 0 {
                let real = if self.flipped { post.rotated() } else { post };
                let done = self.spec;
                *self = StepState::begin(&real, self.salt);
                self.done = done;
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

    /// Forget the cached options and their witnesses (a node whose children
    /// exist no longer needs them).
    pub fn drop_options(&mut self) {
        self.opts = None;
    }

    /// Options for the current step, computed once and cached.
    pub fn options(&mut self) -> Vec<u8> {
        if self.opts.is_none() {
            self.opts = Some(self.compute_options());
        }
        self.opts.as_ref().map(|o| o.0.clone()).unwrap_or_default()
    }

    /// Apply single-option steps while they stay within this mover's turn.
    /// Returns the options of the first real choice (empty = dead end).
    pub fn collapse_forced(&mut self) -> Vec<u8> {
        loop {
            let o = self.options();
            if o.len() != 1 {
                return o;
            }
            let t = self.child(o[0]);
            if t.flipped != self.flipped || t.is_terminal() {
                return o; // the forced step ends the turn: keep it as a child
            }
            *self = t;
        }
    }

    /// What option `v` of the current step decides, in the real board's cell
    /// names, for analysis views.
    pub fn step_label(&self, v: u8) -> String {
        let cell = |c: u8| {
            let r = if self.flipped {
                crate::map::rot(c as usize) as u8
            } else {
                c
            };
            crate::map::cell_name(r)
        };
        let unit = |c: u8| {
            self.base
                .piece_at(c)
                .map_or("?".to_string(), |(_, u)| u.stats().name.to_string())
        };
        match self.stage {
            Stage::Fates => {
                let e = self.enemies[self.idx];
                let what = match v {
                    TARGET_STAY => "spare",
                    TARGET_KILL => "kill",
                    _ => "unsummon",
                };
                format!("{what} {} {}", unit(e), cell(e))
            }
            Stage::Moves => {
                let c = self.friends[self.idx];
                let d = MOVE_DEST[c as usize][v as usize];
                if d == c {
                    format!("hold {} {}", unit(c), cell(c))
                } else {
                    format!("move {} {}{}", unit(c), cell(c), cell(d))
                }
            }
            Stage::Spawns => {
                let at = self.spawn_cells[self.idx];
                match Unit::from_index(v) {
                    Some(u) if v != 0 => format!("spawn {} {}", u.stats().name, cell(at)),
                    _ => format!("no spawn {}", cell(at)),
                }
            }
            Stage::Terminal => String::new(),
        }
    }

    /// Legal options for the current step (empty at a dead end or `Terminal`).
    pub fn children(&self) -> Vec<u8> {
        match &self.opts {
            Some(o) => o.0.clone(),
            None => self.compute_options().0,
        }
    }

    fn compute_options(&self) -> (Vec<u8>, Vec<TurnSpec>) {
        match self.stage {
            Stage::Terminal => (Vec::new(), Vec::new()),
            Stage::Fates => self.fate_options(),
            Stage::Moves => self.move_options(),
            Stage::Spawns => {
                let o = self.spawn_options();
                let w = vec![self.witness; o.len()];
                (o, w)
            }
        }
    }

    /// Apply one option from [`Self::children`], then settle.
    pub fn apply(&mut self, v: u8) {
        self.apply_raw(v);
        self.settle();
    }

    fn apply_raw(&mut self, v: u8) {
        // The option's witness becomes the new one; compute the options if a
        // caller applied without asking for them.
        let (o, w) = match self.opts.take() {
            Some(x) => x,
            None => self.compute_options(),
        };
        if let Some(i) = o.iter().position(|&x| x == v) {
            self.witness = w[i];
        }
        self.apply_step(v);
    }

    /// The state after option `v`, settled. Unlike `clone` + `apply`, this does
    /// not copy this state's option cache (up to 37 witnesses per clone).
    pub fn child(&self, v: u8) -> StepState {
        let witness = match &self.opts {
            Some((o, w)) => o.iter().position(|&x| x == v).map(|i| w[i]),
            None => {
                let (o, w) = self.compute_options();
                o.iter().position(|&x| x == v).map(|i| w[i])
            }
        };
        let mut c = StepState {
            base: self.base.clone(),
            flipped: self.flipped,
            spec: self.spec,
            done: self.done,
            enemies: self.enemies.clone(),
            friends: self.friends.clone(),
            spawn_cells: self.spawn_cells.clone(),
            stage: self.stage,
            idx: self.idx,
            claimed: self.claimed,
            post: self.post.clone(),
            salt: self.salt,
            opts: None,
            witness: witness.unwrap_or(self.witness),
        };
        c.apply_step(v);
        c.settle();
        c
    }

    /// Record decision `v` for the current variable (no witness bookkeeping).
    fn apply_step(&mut self, v: u8) {
        match self.stage {
            Stage::Fates => {
                let e = self.enemies[self.idx];
                self.spec.targets[e as usize] = v;
                self.idx += 1;
                // Killing the necromancer wins on the spot, whatever else is
                // declared: the option's witness realizes it, and the node is
                // terminal.
                if v != TARGET_STAY
                    && self
                        .base
                        .piece_at(e)
                        .is_some_and(|(_, u)| u.stats().necromancer)
                {
                    if let Some(post) = attack_part(&self.base, &self.witness) {
                        if post.winner != 0 {
                            let real = if self.flipped { post.rotated() } else { post };
                            let done = self.witness;
                            *self = StepState::begin(&real, self.salt);
                            self.done = done;
                        }
                    }
                }
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

    /// Debug: describe why this state has no options.
    pub fn debug_dead(&self) -> String {
        let w_ok = self.stage == Stage::Moves && attack_part(&self.base, &self.witness).is_some();
        let mism = self.friends[..self.idx.min(self.friends.len())]
            .iter()
            .filter(|&&f| self.spec.moves[f as usize] != self.witness.moves[f as usize])
            .count();
        let tmism = (0..NUM_CELLS)
            .filter(|&c| self.spec.targets[c] != self.witness.targets[c])
            .count();
        let c = self.friends.get(self.idx).copied().unwrap_or(0);
        let mut extra = String::new();
        if self.stage == Stage::Moves {
            for &f in &self.friends[..self.idx] {
                let k = self.spec.moves[f as usize];
                if k != NO_MOVE && MOVE_DEST[f as usize][k as usize] == c {
                    extra += &format!(
                        " claimant {f} {:?} k{k} w_k{}",
                        self.base.piece_at(f),
                        self.witness.moves[f as usize]
                    );
                }
            }
            let r = realize(&self.base, &self.witness);
            extra += &format!(
                " | w acts {:?} fid {:?}",
                r.actions
                    .iter()
                    .filter(|a| matches!(a, MinionsMove::Move{to, ..} if *to == c)
                        || matches!(a, MinionsMove::Move{from, ..} if *from == c))
                    .collect::<Vec<_>>(),
                (
                    r.fidelity.declared_moves,
                    r.fidelity.done_moves,
                    r.fidelity.declared_kills,
                    r.fidelity.done_kills
                )
            );
        }
        format!(
            "stage {:?} idx {} friends {} witness_realizes {} next_w {:?} move_mismatch {} target_mismatch {} next claimed {} can_move {} unit {:?}",
            self.stage, self.idx, self.friends.len(), w_ok,
            (self.stage == Stage::Moves).then(|| self.witness.moves[c as usize]),
            mism, tmism, has(self.claimed, c), self.base.can_move_piece(c), self.base.piece_at(c),
        ) + &extra
    }

    /// Whether the side to move at this node differs from `other`'s.
    pub fn mover_flipped_vs(&self, other: &StepState) -> bool {
        self.flipped != other.flipped
    }

    // ── Fates ────────────────────────────────────────────────────────────────

    fn fate_options(&self) -> (Vec<u8>, Vec<TurnSpec>) {
        let e = self.enemies[self.idx];
        let (_, t) = self.base.piece_at(e).expect("enemy");
        // The current witness's own fate for `e` is always completable (the
        // witness itself). It is usually "stay", but an exact-search witness
        // may kill an undecided enemy to open a path.
        let own = self.witness.targets[e as usize];
        let (mut out, mut ws) = (vec![own], vec![self.witness]);
        for v in [TARGET_STAY, TARGET_KILL, TARGET_UNSUMMON] {
            if v == own || v == TARGET_UNSUMMON && t.stats().persistent {
                continue;
            }
            let mut spec = self.spec;
            spec.targets[e as usize] = v;
            let mut cand = self.witness;
            cand.targets[e as usize] = v;
            if known_realizable(&self.base, &cand) {
                out.push(v);
                ws.push(cand);
                continue;
            }
            // With the unfated enemies treated as removable the relaxation is
            // a necessary condition; treating them as staying (as the
            // constructor does) is not -- killing one can open a path -- so
            // that only gates the cheap constructor, and the exact search
            // decides.
            let open = self.enemies[self.idx + 1..]
                .iter()
                .fold(0, |b, &x| b | bit(x));
            if !relaxed_feasible(&self.base, &spec, &[], open) {
                continue; // necessary even with the unfated enemies removable
            }
            if relaxed_feasible(&self.base, &spec, &[], 0) {
                if let Some(w) = build_witness(&self.base, &spec, &[]) {
                    out.push(v);
                    ws.push(w);
                    continue;
                }
            }
            let fated = self.enemies[..=self.idx].iter().fold(0, |b, &x| b | bit(x));
            let t0 = std::time::Instant::now();
            let r = exact_realize(&self.base, &spec, 0, fated, EXACT_BUDGET);
            exact_site(0, t0);
            if let Exact::Found(w, _) = r {
                out.push(v);
                ws.push(w);
            }
        }
        (out, ws)
    }

    // ── Moves ────────────────────────────────────────────────────────────────

    fn move_options(&self) -> (Vec<u8>, Vec<TurnSpec>) {
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
        let pass = if st.flying {
            FULL
        } else {
            FULL & !staying_enemies
        };
        let reach = if self.base.can_move_piece(c) {
            bitboard::reachable(bit(c), st.speed, pass)
        } else {
            bit(c)
        };
        let blocked = self.claimed | staying_enemies | stayers;
        let last = self.idx + 1 == self.friends.len();
        let decided: Vec<u8> = self.friends[..self.idx].to_vec();
        let (mut out, mut ws) = (Vec::new(), Vec::new());
        let wk = match self.witness.moves[c as usize] {
            NO_MOVE => 0,
            k => k,
        };
        for k in 0..N_OFFSETS as u8 {
            let d = MOVE_DEST[c as usize][k as usize];
            if d == NO_CELL {
                continue;
            }
            // The witness's own choice is certified, whatever the filters below
            // think (a witness may win the game before a conflict matters).
            if k == wk {
                out.push(k);
                ws.push(self.witness);
                continue;
            }
            if !has(reach, d) {
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
                if attack_part(&self.base, &spec).is_some() {
                    out.push(k);
                    ws.push(spec);
                }
                continue;
            }
            // The current witness, with this unit's move swapped in.
            let mut cand = self.witness;
            cand.moves[c as usize] = k;
            if known_realizable(&self.base, &cand) {
                out.push(k);
                ws.push(cand);
                continue;
            }
            let mut dec = decided.clone();
            dec.push(c);
            // A necessary condition here: every enemy is fated by now, so
            // the relaxation's "undecided enemies stay" is vacuous.
            // (`witness_drop_rate` checks it.)
            if !relaxed_feasible(&self.base, &spec, &dec, 0) {
                continue;
            }
            // Not necessary (it routes evacuating units on vertex-disjoint
            // paths, but friends pass through friends), so it only gates the
            // cheap constructor; the exact search decides.
            if self.movers_completable(&spec, self.claimed | bit(d), c, k) {
                if let Some(w) = build_witness(&self.base, &spec, &dec) {
                    out.push(k);
                    ws.push(w);
                    continue;
                }
            }
            let decided_bb = dec.iter().fold(0, |b, &x| b | bit(x));
            let fated = side_bb(&self.base, 1);
            let t0 = std::time::Instant::now();
            let r = exact_realize(&self.base, &spec, decided_bb, fated, EXACT_BUDGET);
            exact_site(1, t0);
            if let Exact::Found(w, _) = r {
                out.push(k);
                ws.push(w);
            }
        }
        (out, ws)
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
        // Decided moves as origin -> destination, including the tentative one.
        let mut dest_of = [NO_CELL; NUM_CELLS];
        for &f in self.friends[..self.idx].iter().chain(std::iter::once(&c)) {
            let kk = if f == c { k } else { spec.moves[f as usize] };
            if kk != 0 {
                dest_of[f as usize] = MOVE_DEST[f as usize][kk as usize];
            }
        }
        // A vacated origin whose chain of decided moves ends on an undecided
        // unit's origin is only free if that unit moves first -- which, when it
        // is the unit being routed, is a cycle. Conservatively not a sink.
        let chain_to_undecided = |x: u8| -> bool {
            let mut y = x;
            for _ in 0..NUM_CELLS {
                let d = dest_of[y as usize];
                if d == NO_CELL {
                    return false;
                }
                if has(und_bb, d) {
                    return true;
                }
                y = d;
            }
            true
        };
        // A cell is a sink if nothing will stand on it at the end: empty now,
        // a removed enemy, or a decided mover's vacated origin -- and unclaimed.
        let blocked = claimed | staying_enemies | stayers;
        let sink = |x: u8| -> bool {
            !has(blocked, x)
                && !has(und_bb, x)
                && !(dest_of[x as usize] != NO_CELL && chain_to_undecided(x))
        };
        // Graph: node per undecided unit (split in/out) and per cell (split).
        // ids: S=0, T=1, unit u (index i): in 2+2i, out 3+2i; cell x: in 2+2n+2x, out 3+2n+2x.
        let n = undecided.len();
        let unit_idx = |x: u8| undecided.iter().position(|&u| u == x);
        let nn = 2 + 2 * n + 2 * NUM_CELLS;
        let mut g = Flow::new(nn);
        let (uin, uout) = (|i: usize| 2 + 2 * i, |i: usize| 3 + 2 * i);
        let (cin, cout) = (
            |x: u8| 2 + 2 * n + 2 * x as usize,
            |x: u8| 3 + 2 * n + 2 * x as usize,
        );
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
            let pass = if st.flying {
                FULL
            } else {
                FULL & !staying_enemies
            };
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

    #[cfg(test)]
    fn reject_reasons(&self) -> [usize; 6] {
        // [unreachable/blocked/cycle, flow, relaxed, lookahead, last-realize, ok]
        let c = self.friends[self.idx];
        let (_, u) = self.base.piece_at(c).expect("friend");
        let st = u.stats();
        let mut staying_enemies: Bb = 0;
        for e in bitboard::cells(side_bb(&self.base, 1)) {
            if self.spec.targets[e as usize] == TARGET_STAY {
                staying_enemies |= bit(e);
            }
        }
        let mut stayers: Bb = 0;
        for &f in &self.friends[..self.idx] {
            if self.spec.moves[f as usize] == 0 {
                stayers |= bit(f);
            }
        }
        let pass = if st.flying {
            FULL
        } else {
            FULL & !staying_enemies
        };
        let reach = bitboard::reachable(bit(c), st.speed, pass);
        let blocked = self.claimed | staying_enemies | stayers;
        let last = self.idx + 1 == self.friends.len();
        let mut r = [0usize; 6];
        for k in 0..N_OFFSETS as u8 {
            let d = MOVE_DEST[c as usize][k as usize];
            if d == NO_CELL || !has(reach, d) {
                continue;
            }
            if (k == 0 && has(self.claimed, c))
                || (k != 0 && (has(blocked, d) || self.closes_cycle(c, d)))
            {
                r[0] += 1;
                continue;
            }
            let mut spec = self.spec;
            spec.moves[c as usize] = k;
            if last {
                if attack_part(&self.base, &spec).is_none() {
                    r[4] += 1;
                } else {
                    r[5] += 1;
                }
                continue;
            }
            if !self.movers_completable(&spec, self.claimed | bit(d), c, k) {
                r[1] += 1;
                continue;
            }
            let mut dec: Vec<u8> = self.friends[..self.idx].to_vec();
            dec.push(c);
            if !relaxed_feasible(&self.base, &spec, &dec, 0) {
                r[2] += 1;
                continue;
            }
            r[5] += 1;
        }
        r
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
            killed: 0,
            unsummoned: 0,
            gone_units: [0; 6],
        };
        // Fated enemies, from the start-of-turn position (the spawn stage's
        // board has already removed them).
        let n_fated = match self.stage {
            Stage::Terminal => 0,
            Stage::Fates => self.idx,
            Stage::Moves | Stage::Spawns => self.enemies.len(),
        };
        for &e in &self.enemies[..n_fated] {
            let t = self.spec.targets[e as usize];
            if t == TARGET_STAY {
                continue;
            }
            if let Some((_, unit)) = self.base.piece_at(e) {
                v.gone_units[unit as usize] |= bit(e);
            }
            if t == TARGET_UNSUMMON {
                v.unsummoned |= bit(e);
            } else {
                v.killed |= bit(e);
            }
        }
        match self.stage {
            Stage::Terminal => {}
            Stage::Fates | Stage::Moves => {
                let b = &mut v.board;
                let n_fates = if self.stage == Stage::Fates {
                    self.idx
                } else {
                    self.enemies.len()
                };
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
        Self {
            head: vec![u32::MAX; n],
            to: Vec::new(),
            cap: Vec::new(),
            next: Vec::new(),
        }
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
/// [`attack_part_fast`], then the exact search when the fast realizers miss:
/// complete up to [`EXACT_BUDGET`]. Every realization the step tree commits
/// to goes through here.
pub fn attack_part(base: &MinionsState, spec: &TurnSpec) -> Option<MinionsState> {
    if let Some(s) = attack_part_fast(base, spec) {
        return Some(s);
    }
    let path = exact_complete(base, spec)?;
    let mut s = base.clone();
    for mv in &path {
        s = Minions::apply(&s, mv);
    }
    Some(s)
}

/// Key of a complete declaration at `base`: the position, every friend's
/// destination and every enemy's fate (spawns ignored).
fn exact_key(base: &MinionsState, spec: &TurnSpec) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    Minions::encode_state(base).hash(&mut h);
    for c in 0..NUM_CELLS as u8 {
        if base.piece_at(c).is_some_and(|(sd, _)| sd == 0) {
            let k = spec.moves[c as usize];
            let d = if k == NO_MOVE || k == 0 {
                c
            } else {
                MOVE_DEST[c as usize][k as usize]
            };
            d.hash(&mut h);
        }
    }
    spec.targets.hash(&mut h);
    h.finish()
}

/// Attack-phase paths of complete declarations the exact search found,
/// so committing a declaration never re-runs a search that could run out of
/// budget where the one that offered it did not.
static EXACT_CACHE: std::sync::Mutex<Option<std::collections::HashMap<u64, Vec<MinionsMove>>>> =
    std::sync::Mutex::new(None);

fn exact_cache_get(base: &MinionsState, spec: &TurnSpec) -> Option<Vec<MinionsMove>> {
    let k = exact_key(base, spec);
    EXACT_CACHE.lock().unwrap().as_ref()?.get(&k).cloned()
}

fn exact_cache_put(base: &MinionsState, spec: &TurnSpec, path: &[MinionsMove]) {
    let k = exact_key(base, spec);
    let mut g = EXACT_CACHE.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    if m.len() >= 200_000 {
        m.clear();
    }
    m.insert(k, path.to_vec());
}

/// The attack-phase actions realizing a complete declaration by the exact
/// search (cached).
fn exact_complete(base: &MinionsState, spec: &TurnSpec) -> Option<Vec<MinionsMove>> {
    if let Some(p) = exact_cache_get(base, spec) {
        return Some(p);
    }
    let mut sp = *spec;
    sp.spawns = [0; NUM_CELLS];
    let t0 = std::time::Instant::now();
    let r = exact_realize(base, &sp, side_bb(base, 0), side_bb(base, 1), EXACT_BUDGET);
    exact_site(2, t0);
    match r {
        Exact::Found(_, path) => Some(path),
        _ => None,
    }
}

/// The fast realizers, then the exact-path cache: never searches.
fn known_realizable(base: &MinionsState, spec: &TurnSpec) -> bool {
    attack_part_fast(base, spec).is_some() || exact_cache_get(base, spec).is_some()
}

/// The two fast realizers: all kills planned at once, then one kill at a
/// time (six rotations of which target goes first). Incomplete: a miss is not
/// a proof. Cheap enough for the witness swap checks.
pub fn attack_part_fast(base: &MinionsState, spec: &TurnSpec) -> Option<MinionsState> {
    let mut sp = *spec;
    sp.spawns = [0; NUM_CELLS];
    let r = crate::turn::realize_attack(base, &sp);
    let s = r.next;
    if s.winner != 0 {
        return Some(s); // a won game needs nothing else declared to happen
    }
    let f = r.fidelity;
    if f.done_moves == f.declared_moves
        && f.done_kills == f.declared_kills
        && r.spec.targets == sp.targets
    {
        return Some(s);
    }
    // The realizer plans every kill at once; some turns need one kill to open
    // a hex that a unit moves into before it attacks. Try one kill at a time,
    // each declared target first in turn.
    let n_targets = (0..NUM_CELLS)
        .filter(|&c| sp.targets[c] != TARGET_STAY)
        .count();
    (0..n_targets.min(6)).find_map(|first| realize_sequential(base, &sp, first, None))
}

/// The atomic moves that play `done` -- a complete declaration in the
/// canonical frame of `real`'s mover, as [`StepState::done`] holds it -- from
/// the real position `real`, the way the step tree realized it.
pub fn turn_actions(real: &MinionsState, done: &TurnSpec) -> anyhow::Result<Vec<MinionsMove>> {
    let base = real.canonical();
    let mut sp = *done;
    sp.spawns = [0; NUM_CELLS];
    let r = crate::turn::realize_attack(&base, &sp);
    let f = r.fidelity;
    let (mut acts, mut s) = if r.next.winner != 0
        || (f.done_moves == f.declared_moves && f.done_kills == f.declared_kills)
    {
        (r.actions, r.next)
    } else {
        let n_targets = (0..NUM_CELLS)
            .filter(|&c| sp.targets[c] != TARGET_STAY)
            .count();
        (0..n_targets.min(6))
            .find_map(|first| {
                let mut log = Vec::new();
                realize_sequential(&base, &sp, first, Some(&mut log)).map(|s| (log, s))
            })
            .or_else(|| {
                let path = exact_complete(&base, &sp)?;
                let mut s = base.clone();
                for mv in &path {
                    s = Minions::apply(&s, mv);
                }
                Some((path, s))
            })
            .ok_or_else(|| anyhow::anyhow!("the declared attack part does not realize"))?
    };
    if s.winner == 0 && s.phase == PHASE_SPAWN {
        for at in 0..NUM_CELLS as u8 {
            let v = done.spawns[at as usize];
            if v == 0 {
                continue;
            }
            let unit = Unit::from_index(v).ok_or_else(|| anyhow::anyhow!("bad spawn code {v}"))?;
            let slot = unit
                .reinforcement_slot()
                .ok_or_else(|| anyhow::anyhow!("{unit:?} is not buyable"))?;
            if s.reinforcements[0][slot] == 0 {
                acts.push(MinionsMove::Buy(unit));
                s = Minions::apply(&s, &MinionsMove::Buy(unit));
            }
            acts.push(MinionsMove::Spawn { unit, at });
            s = Minions::apply(&s, &MinionsMove::Spawn { unit, at });
        }
        acts.push(MinionsMove::EndPhase);
    }
    if real.side == 1 {
        acts.iter_mut().for_each(|m| *m = m.rotated());
    }
    Ok(acts)
}

/// Exact-search calls and nanoseconds by call site: fates options, moves
/// options, complete declarations (last unit / commit).
pub static EXACT_SITES: [[std::sync::atomic::AtomicU64; 2]; 3] = [
    [
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ],
    [
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ],
    [
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ],
];

fn exact_site(site: usize, t0: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    EXACT_SITES[site][0].fetch_add(1, Relaxed);
    EXACT_SITES[site][1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}

/// Search nodes [`exact_realize`] may visit before giving up.
pub const EXACT_BUDGET: u32 = 20_000;

/// Outcome of [`exact_realize`].
pub enum Exact {
    /// A completed declaration and the attack-phase actions that realize it
    /// (ending with the `EndPhase` into the spawn phase unless the game was
    /// won on the way).
    Found(TurnSpec, Vec<MinionsMove>),
    /// No sequence of legal actions realizes the decisions.
    Impossible,
    /// The budget ran out first.
    Unknown,
}

/// Counters for how often the exact search runs and what it concludes.
pub static EXACT_STATS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0), // found
    std::sync::atomic::AtomicU64::new(0), // impossible
    std::sync::atomic::AtomicU64::new(0), // unknown (budget)
    std::sync::atomic::AtomicU64::new(0), // nanoseconds spent
];

/// Exhaustive search over the attack phase's legal atomic actions for a
/// sequence that realizes the decisions in `spec`: every friend in `decided`
/// ends on its declared hex, every enemy in `fated` meets its declared fate
/// (spared ones untouched), and no enemy is left damaged but alive (a
/// declaration has no "damage only"). Undecided friends may do anything that
/// can matter -- move to where they can attack a remaining target, or vacate a
/// hex that a decided unit or an attacker may need -- and undecided enemies
/// may be killed or unsummoned (a kill that opens a path). Game rules come
/// from `Minions::legal_moves`, so this is complete up to the budget: unlike
/// the fast realizers and the witness constructor it never misses an
/// ordering, an attacker choice or a blocker that has to move.
pub fn exact_realize(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: Bb,
    fated: Bb,
    budget: u32,
) -> Exact {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    struct Ctx<'a> {
        spec: &'a TurnSpec,
        decided: Bb,
        fated: Bb,
        budget: u32,
        seen: HashSet<u64>,
        path: Vec<MinionsMove>,
        found: Option<TurnSpec>,
        out_of_budget: bool,
    }
    // Per cell: the origin of the friendly unit standing there (NO_CELL if
    // none), and per enemy origin how it left the board (0 = present).
    #[derive(Clone)]
    struct St {
        s: MinionsState,
        origin_at: [u8; NUM_CELLS],
        gone: [u8; NUM_CELLS],
    }
    fn dest_of(spec: &TurnSpec, o: u8) -> u8 {
        let k = spec.moves[o as usize];
        if k == NO_MOVE || k == 0 {
            o
        } else {
            MOVE_DEST[o as usize][k as usize]
        }
    }
    // Enemies that still have to go: fated kill/unsummon, still present.
    fn remaining(c: &Ctx, st: &St) -> Bb {
        let mut b = 0;
        for e in bitboard::cells(c.fated) {
            if c.spec.targets[e as usize] != TARGET_STAY && st.gone[e as usize] == 0 {
                b |= bit(e);
            }
        }
        b
    }
    fn goal(c: &Ctx, st: &St) -> bool {
        if st.s.winner != 0 {
            return st.s.winner == 1;
        }
        for o in bitboard::cells(c.decided) {
            if st.origin_at[dest_of(c.spec, o) as usize] != o {
                return false;
            }
        }
        for e in 0..NUM_CELLS as u8 {
            let fate = c.spec.targets[e as usize];
            if has(c.fated, e) && st.gone[e as usize] != fate {
                return false;
            }
            // Damaged but alive: a declaration cannot say that.
            if st.gone[e as usize] == 0
                && st.s.piece_at(e).is_some_and(|(sd, _)| sd == 1)
                && st.s.damage_on(e) > 0
            {
                return false;
            }
        }
        true
    }
    // Cheap dead-end test: a remaining target nobody can still reach -- unless
    // the enemy necromancer can still be hit, since winning makes every other
    // declared fate moot (the game ends mid-turn).
    fn hopeless(c: &Ctx, st: &St, rem: Bb) -> bool {
        let spared: Bb = bitboard::cells(c.fated)
            .filter(|&e| c.spec.targets[e as usize] == TARGET_STAY)
            .fold(0, |b, e| b | bit(e));
        let necro_open = (0..NUM_CELLS as u8).any(|e| {
            !has(spared, e)
                && st
                    .s
                    .piece_at(e)
                    .is_some_and(|(sd, u)| sd == 1 && u.stats().necromancer)
                && hittable(c, st, e, spared)
        });
        if necro_open {
            return false;
        }
        bitboard::cells(rem).any(|t| !hittable(c, st, t, spared))
    }
    // Whether some friend can still end its turn in range of `t` and attack.
    fn hittable(c: &Ctx, st: &St, t: u8, spared: Bb) -> bool {
        let s = &st.s;
        {
            let mut ok = false;
            for a in 0..NUM_CELLS as u8 {
                let Some((sd, u)) = s.piece_at(a) else {
                    continue;
                };
                if sd != 0 || !s.can_attack_piece(a, u) {
                    continue;
                }
                let st_ = u.stats();
                let ring = bitboard::reachable(bit(t), st_.range, FULL) & !bit(t);
                let o = st.origin_at[a as usize];
                let ends = if has(c.decided, o) {
                    let d = dest_of(c.spec, o);
                    if d == a || !s.can_move_piece(a) {
                        bit(a)
                    } else if st_.lumbering {
                        0
                    } else {
                        bit(d)
                    }
                } else if s.can_move_piece(a) && !st_.lumbering {
                    let pass = if st_.flying { FULL } else { FULL & !spared };
                    bitboard::reachable(bit(a), st_.speed, pass)
                } else {
                    bit(a)
                };
                if ends & ring != 0 {
                    ok = true;
                    break;
                }
            }
            ok
        }
    }
    fn dfs(c: &mut Ctx, st: &St) -> bool {
        if goal(c, st) {
            // The completed declaration: where every friend ended, what
            // happened to every enemy.
            let mut w = *c.spec;
            w.spawns = [0; NUM_CELLS];
            for cell in 0..NUM_CELLS as u8 {
                let o = st.origin_at[cell as usize];
                if o != NO_CELL {
                    w.moves[o as usize] = if o == cell {
                        0
                    } else {
                        crate::turn::offset_index(o, cell).unwrap_or(0)
                    };
                }
            }
            for e in 0..NUM_CELLS as u8 {
                if !has(c.fated, e) {
                    w.targets[e as usize] = st.gone[e as usize];
                }
            }
            c.found = Some(w);
            return true;
        }
        if st.s.winner != 0 {
            return false;
        }
        if c.budget == 0 {
            c.out_of_budget = true;
            return false;
        }
        c.budget -= 1;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        Minions::encode_state(&st.s).hash(&mut h);
        st.origin_at.hash(&mut h);
        st.gone.hash(&mut h);
        if !c.seen.insert(h.finish()) {
            return false;
        }
        let rem = remaining(c, st);
        if hopeless(c, st, rem) {
            return false;
        }
        // Hexes someone may still need: decided units' destinations not yet
        // reached, and every hex from which a remaining (or unfated) target
        // can be hit.
        let mut want: Bb = 0;
        for o in bitboard::cells(c.decided) {
            let d = dest_of(c.spec, o);
            if st.origin_at[d as usize] != o {
                want |= bit(d);
            }
        }
        let open_targets: Bb = (0..NUM_CELLS as u8)
            .filter(|&e| {
                st.gone[e as usize] == 0
                    && st.s.piece_at(e).is_some_and(|(sd, _)| sd == 1)
                    && !(has(c.fated, e) && c.spec.targets[e as usize] == TARGET_STAY)
            })
            .fold(0, |b, e| b | bit(e));
        let rmax = (0..6u8)
            .filter_map(Unit::from_index)
            .map(|u| u.stats().range)
            .max()
            .unwrap_or(1);
        let mut rings: Bb = 0;
        for t in bitboard::cells(open_targets) {
            rings |= bitboard::reachable(bit(t), rmax, FULL) & !bit(t);
        }
        let mut attacks = Vec::new();
        let mut moves = Vec::new();
        for mv in Minions::legal_moves(&st.s) {
            match mv {
                MinionsMove::Attack { from, to } => {
                    if !has(open_targets, to) {
                        continue;
                    }
                    let (_, au) = st.s.piece_at(from).expect("attacker");
                    let (_, tu) = st.s.piece_at(to).expect("target");
                    let unsummons =
                        matches!(au.stats().attack, Attack::Unsummon) && !tu.stats().persistent;
                    if has(c.fated, to) {
                        let fate = c.spec.targets[to as usize];
                        if (fate == TARGET_UNSUMMON) != unsummons {
                            continue;
                        }
                    }
                    attacks.push(mv);
                }
                MinionsMove::Move { from, to } => {
                    let o = st.origin_at[from as usize];
                    if has(c.decided, o) {
                        if dest_of(c.spec, o) == to && to != o {
                            moves.push((0u8, mv));
                        }
                        continue;
                    }
                    let (_, u) = st.s.piece_at(from).expect("mover");
                    let range = u.stats().range;
                    let can_hit = bitboard::cells(open_targets)
                        .any(|t| crate::map::dist(to, t) <= range as u32 && to != t);
                    if can_hit {
                        moves.push((1, mv));
                    } else if has(want | rings, from) {
                        // Vacate, preferably to a hex nobody wants.
                        moves.push((if has(want | rings, to) { 3 } else { 2 }, mv));
                    }
                }
                _ => {}
            }
        }
        moves.sort_by_key(|&(p, _)| p);
        let actions = attacks.into_iter().chain(moves.into_iter().map(|(_, m)| m));
        for mv in actions.collect::<Vec<_>>() {
            let mut nx = st.clone();
            nx.s = Minions::apply(&st.s, &mv);
            match mv {
                MinionsMove::Move { from, to } => {
                    nx.origin_at[to as usize] = st.origin_at[from as usize];
                    nx.origin_at[from as usize] = NO_CELL;
                }
                MinionsMove::Attack { from, to } => {
                    if nx.s.piece_at(to).is_none() {
                        let (_, au) = st.s.piece_at(from).expect("attacker");
                        let (_, tu) = st.s.piece_at(to).expect("target");
                        let unsummoned =
                            matches!(au.stats().attack, Attack::Unsummon) && !tu.stats().persistent;
                        nx.gone[to as usize] = if unsummoned {
                            TARGET_UNSUMMON
                        } else {
                            TARGET_KILL
                        };
                    }
                }
                _ => {}
            }
            c.path.push(mv);
            if dfs(c, &nx) {
                return true;
            }
            c.path.pop();
            if c.budget == 0 {
                c.out_of_budget = true;
                return false;
            }
        }
        false
    }
    let mut origin_at = [NO_CELL; NUM_CELLS];
    for cell in 0..NUM_CELLS as u8 {
        if base.piece_at(cell).is_some_and(|(sd, _)| sd == 0) {
            origin_at[cell as usize] = cell;
        }
    }
    let root = St {
        s: base.clone(),
        origin_at,
        gone: [0; NUM_CELLS],
    };
    let mut c = Ctx {
        spec,
        decided,
        fated,
        budget,
        seen: HashSet::new(),
        path: Vec::new(),
        found: None,
        out_of_budget: false,
    };
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let ok = dfs(&mut c, &root);
    EXACT_STATS[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    if ok {
        let mut path = c.path;
        let mut s = base.clone();
        for mv in &path {
            s = Minions::apply(&s, mv);
        }
        if s.winner == 0 {
            path.push(MinionsMove::EndPhase);
        }
        EXACT_STATS[0].fetch_add(1, Relaxed);
        let w = c.found.expect("found");
        exact_cache_put(base, &w, &path);
        Exact::Found(w, path)
    } else if c.out_of_budget {
        EXACT_STATS[2].fetch_add(1, Relaxed);
        Exact::Unknown
    } else {
        EXACT_STATS[1].fetch_add(1, Relaxed);
        Exact::Impossible
    }
}

/// Moves whenever possible, then a single declared kill (targets tried from
/// `first` onward), repeat. Returns the position after the attack phase if every
/// declared move and kill happened.
fn realize_sequential(
    base: &MinionsState,
    spec: &TurnSpec,
    first: usize,
    mut log: Option<&mut Vec<MinionsMove>>,
) -> Option<MinionsState> {
    let me = 0u8;
    let mut s = base.clone();
    let mut want = [NO_CELL; NUM_CELLS];
    for c in 0..NUM_CELLS as u8 {
        if s.piece_at(c).is_some_and(|(sd, _)| sd == me) {
            let k = spec.moves[c as usize];
            if k != NO_MOVE && k != 0 {
                want[c as usize] = MOVE_DEST[c as usize][k as usize];
            }
        }
    }
    let mut targets: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&c| spec.targets[c as usize] != TARGET_STAY)
        .collect();
    let nt = targets.len().max(1);
    targets.rotate_left(first % nt);
    let mut alive: Vec<bool> = targets
        .iter()
        .map(|&t| s.piece_at(t).is_some_and(|(sd, _)| sd != me))
        .collect();
    for _ in 0..4 * NUM_CELLS {
        // Moves.
        let mut moved = true;
        while moved {
            moved = false;
            let enemy = side_bb(&s, 1);
            for o in 0..NUM_CELLS as u8 {
                let d = want[o as usize];
                if d == NO_CELL {
                    continue;
                }
                let Some((sd, u)) = s.piece_at(o) else {
                    return None;
                };
                if sd != me || !s.can_move_piece(o) || s.piece_at(d).is_some() {
                    continue;
                }
                let st = u.stats();
                let pass = if st.flying { FULL } else { FULL & !enemy };
                if !has(bitboard::reachable(bit(o), st.speed, pass), d) {
                    continue;
                }
                let mv = MinionsMove::Move { from: o, to: d };
                s = Minions::apply(&s, &mv);
                if let Some(l) = log.as_deref_mut() {
                    l.push(mv);
                }
                want[o as usize] = NO_CELL;
                moved = true;
            }
        }
        if s.winner != 0 {
            return Some(s);
        }
        // One kill.
        let mut fired = false;
        for (ti, &t) in targets.iter().enumerate() {
            if !alive[ti] {
                continue;
            }
            let Some((_, tu)) = s.piece_at(t) else {
                alive[ti] = false;
                continue;
            };
            let unsummon = spec.targets[t as usize] == TARGET_UNSUMMON;
            // Attackers in range, those with no pending move first.
            let mut atk: Vec<(u8, u32)> = Vec::new();
            for a in 0..NUM_CELLS as u8 {
                let Some((sd, u)) = s.piece_at(a) else {
                    continue;
                };
                if sd != me
                    || !s.can_attack_piece(a, u)
                    || crate::map::dist(a, t) > u.stats().range as u32
                {
                    continue;
                }
                let dmg = if unsummon {
                    u32::from(matches!(u.stats().attack, Attack::Unsummon))
                } else {
                    kill_damage_against(u, tu) as u32
                };
                if dmg > 0 {
                    atk.push((a, dmg));
                }
            }
            atk.sort_by_key(|&(a, d)| (want[a as usize] != NO_CELL, std::cmp::Reverse(d)));
            let need = if unsummon {
                1
            } else {
                tu.stats().defense.saturating_sub(s.damage_on(t)) as u32
            };
            let mut got = 0;
            let mut use_: Vec<u8> = Vec::new();
            for &(a, d) in &atk {
                if got >= need {
                    break;
                }
                got += d;
                use_.push(a);
            }
            if got < need {
                continue;
            }
            for a in use_ {
                let mv = MinionsMove::Attack { from: a, to: t };
                s = Minions::apply(&s, &mv);
                if let Some(l) = log.as_deref_mut() {
                    l.push(mv);
                }
                if s.winner != 0 {
                    return Some(s);
                }
            }
            alive[ti] = false;
            fired = true;
            break;
        }
        if !fired {
            break;
        }
    }
    if want.iter().any(|&d| d != NO_CELL) || alive.iter().any(|&a| a) {
        return None;
    }
    let s = Minions::apply(&s, &MinionsMove::EndPhase);
    if let Some(l) = log {
        l.push(MinionsMove::EndPhase);
    }
    Some(s)
}

/// Friendly units that can still attack, with the hexes each could end on:
/// a decided unit its destination (none if lumbering and moved); an undecided
/// one its origin (unless claimed) plus reachable hexes not in `block`.
/// Enemies in `open` (not yet fated) may be removed, so they block nothing.
fn attackers(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: &[u8],
    block: Bb,
    open: Bb,
) -> Vec<(u8, Unit, Bb)> {
    let mut staying_enemies: Bb = 0;
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != 0)
            && spec.targets[e as usize] == TARGET_STAY
            && !has(open, e)
        {
            staying_enemies |= bit(e);
        }
    }
    let mut claimed: Bb = 0;
    for &f in decided {
        let k = spec.moves[f as usize];
        claimed |= bit(if k == NO_MOVE {
            f
        } else {
            MOVE_DEST[f as usize][k as usize]
        });
    }
    let mut out = Vec::new();
    for c in 0..NUM_CELLS as u8 {
        let Some((sd, u)) = base.piece_at(c) else {
            continue;
        };
        if sd != 0 || !base.can_attack_piece(c, u) {
            continue;
        }
        let st = u.stats();
        let ends: Bb = if decided.contains(&c) {
            let k = spec.moves[c as usize];
            let d = if k == NO_MOVE {
                c
            } else {
                MOVE_DEST[c as usize][k as usize]
            };
            if st.lumbering && d != c {
                0
            } else {
                bit(d)
            }
        } else {
            let stay = if has(claimed, c) { 0 } else { bit(c) };
            if st.lumbering || !base.can_move_piece(c) {
                stay
            } else {
                let pass = if st.flying {
                    FULL
                } else {
                    FULL & !staying_enemies
                };
                let moves = bitboard::reachable(bit(c), st.speed, pass)
                    & !bit(c)
                    & !claimed
                    & !staying_enemies
                    & !block;
                stay | moves
            }
        };
        if ends != 0 {
            out.push((c, u, ends));
        }
    }
    out
}

/// The declared kills/unsummons as assignment targets over `atk`, or `None`
/// if one has nobody able to reach it. Attack rings exclude the target's own
/// hex: a unit attacks after it moves, never before.
fn kill_targets(
    base: &MinionsState,
    spec: &TurnSpec,
    atk: &[(u8, Unit, Bb)],
) -> Option<Vec<assign::Target>> {
    let mut targets = Vec::new();
    for e in 0..NUM_CELLS as u8 {
        let t = spec.targets[e as usize];
        if t == TARGET_STAY {
            continue;
        }
        let Some((sd, tu)) = base.piece_at(e) else {
            continue;
        };
        if sd == 0 {
            continue;
        }
        let mut reach = Vec::new();
        for (i, &(_, u, ends)) in atk.iter().enumerate() {
            let ring = bitboard::reachable(bit(e), u.stats().range, FULL) & !bit(e);
            if ends & ring == 0 {
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
        if reach.is_empty() {
            return None;
        }
        let need = if t == TARGET_UNSUMMON {
            1
        } else {
            tu.stats().defense.saturating_sub(base.damage_on(e)) as u32
        };
        targets.push(assign::Target {
            cell: e,
            need,
            reach,
        });
    }
    Some(targets)
}

/// A cheap necessary condition for the declared kills: reach on real end
/// hexes, the deficiency bound, and a positional search (budget-limited).
/// A necessary condition for `spec`'s declared kills and decided moves when
/// the enemies in `open` are still unfated (they may be killed to open paths
/// and hexes, so they are treated as removable).
fn relaxed_feasible(base: &MinionsState, spec: &TurnSpec, decided: &[u8], open: Bb) -> bool {
    let atk = attackers(base, spec, decided, 0, open);
    let Some(targets) = kill_targets(base, spec, &atk) else {
        return false;
    };
    if targets.is_empty() {
        return true;
    }
    let attackers: Vec<assign::Attacker> = atk
        .iter()
        .map(|&(c, _, _)| assign::Attacker {
            cell: c,
            attacks: 1,
        })
        .collect();
    if assign::hall_violation(&targets, &attackers).is_some() {
        return false;
    }
    let mut budget = POSITION_BUDGET;
    positions(base, &targets, &atk, &mut budget, &mut |_| true) != Some(false)
}

/// Build a complete fates+moves declaration that realizes `spec`'s declared
/// kills and decided moves: attackers placed by the positional search, units
/// forced off claimed origins relocated, everyone else staying. `None` if no
/// attempt realizes within the budget.
fn build_witness(base: &MinionsState, spec: &TurnSpec, decided: &[u8]) -> Option<TurnSpec> {
    let friends: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&c| base.piece_at(c).is_some_and(|(sd, _)| sd == 0))
        .collect();
    let undecided: Vec<u8> = friends
        .iter()
        .copied()
        .filter(|f| !decided.contains(f))
        .collect();
    let mut claimed: Bb = 0;
    for &f in decided {
        let k = spec.moves[f as usize];
        claimed |= bit(if k == NO_MOVE {
            f
        } else {
            MOVE_DEST[f as usize][k as usize]
        });
    }
    let und_origins: Bb = undecided.iter().fold(0, |b, &u| b | bit(u));
    let mut staying_enemies: Bb = 0;
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != 0)
            && spec.targets[e as usize] == TARGET_STAY
        {
            staying_enemies |= bit(e);
        }
    }
    // Undecided attackers may not step onto another undecided unit's origin
    // (it stays by default) -- conservative, but keeps witnesses simple.
    let atk = attackers(base, spec, decided, und_origins, 0);
    let targets = kill_targets(base, spec, &atk)?;
    let mut found = None;
    let mut attempts = 0u32;
    let mut try_positions = |chosen: &[(usize, u8)]| -> bool {
        attempts += 1;
        let mut w = *spec;
        let mut taken = claimed;
        let mut moved: Bb = 0;
        for &(ai, h) in chosen {
            let a = atk[ai].0;
            if decided.contains(&a) {
                continue;
            }
            let Some(k) = crate::turn::offset_index(a, h) else {
                return false;
            };
            w.moves[a as usize] = k;
            taken |= bit(h);
            moved |= bit(a);
        }
        // Everyone undecided and unplaced stays -- unless its origin is claimed.
        for &u in &undecided {
            if has(moved, u) {
                continue;
            }
            if !has(claimed, u) {
                w.moves[u as usize] = 0;
                continue;
            }
            let (_, unit) = base.piece_at(u).expect("friend");
            let st = unit.stats();
            if !base.can_move_piece(u) {
                return false;
            }
            let pass = if st.flying {
                FULL
            } else {
                FULL & !staying_enemies
            };
            let free = bitboard::reachable(bit(u), st.speed, pass)
                & !bit(u)
                & !taken
                & !staying_enemies
                & !(und_origins & !moved);
            let Some(h) = bitboard::cells(free).next() else {
                return false;
            };
            let Some(k) = crate::turn::offset_index(u, h) else {
                return false;
            };
            w.moves[u as usize] = k;
            taken |= bit(h);
            moved |= bit(u);
        }
        if attack_part_fast(base, &w).is_some() {
            found = Some(w);
            true
        } else {
            attempts >= WITNESS_ATTEMPTS
        }
    };
    let mut budget = POSITION_BUDGET;
    if targets.is_empty() {
        try_positions(&[]);
    } else {
        positions(base, &targets, &atk, &mut budget, &mut try_positions);
    }
    found
}

/// Budget (search nodes) for [`positions`].
const POSITION_BUDGET: u32 = 4000;
/// Complete positional assignments a witness search realizes before giving up.
const WITNESS_ATTEMPTS: u32 = 24;

/// Enumerate assignments of attackers to distinct hexes covering every target
/// (most constrained target first), calling `found` on each; stops when it
/// returns true. `Some(true)` if stopped by `found`, `Some(false)` if
/// exhausted, `None` if the budget ran out.
fn positions(
    _base: &MinionsState,
    targets: &[assign::Target],
    atk: &[(u8, Unit, Bb)],
    budget: &mut u32,
    found: &mut dyn FnMut(&[(usize, u8)]) -> bool,
) -> Option<bool> {
    let mut order: Vec<usize> = (0..targets.len()).collect();
    order.sort_by_key(|&i| targets[i].reach.len());
    let rings: Vec<Vec<Bb>> = targets
        .iter()
        .map(|t| {
            atk.iter()
                .map(|&(_, u, _)| {
                    bitboard::reachable(bit(t.cell), u.stats().range, FULL) & !bit(t.cell)
                })
                .collect()
        })
        .collect();
    let mut chosen = Vec::new();
    positions_dfs(
        targets,
        &order,
        0,
        0,
        atk,
        &rings,
        0,
        0,
        &mut chosen,
        budget,
        found,
    )
}

#[allow(clippy::too_many_arguments)]
fn positions_dfs(
    targets: &[assign::Target],
    order: &[usize],
    ti: usize,
    owed: u32,
    atk: &[(u8, Unit, Bb)],
    rings: &[Vec<Bb>],
    used: u64,
    hexes: Bb,
    chosen: &mut Vec<(usize, u8)>,
    budget: &mut u32,
    found: &mut dyn FnMut(&[(usize, u8)]) -> bool,
) -> Option<bool> {
    if ti == order.len() {
        return if found(chosen) {
            Some(true)
        } else {
            Some(false)
        };
    }
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    let t = &targets[order[ti]];
    let owed = if owed == 0 { t.need } else { owed };
    let mut unknown = false;
    for &(ai, dmg) in &t.reach {
        if used & (1 << ai) != 0 {
            continue;
        }
        let free = atk[ai].2 & rings[order[ti]][ai] & !hexes;
        for h in bitboard::cells(free) {
            let (nti, nowed) = if dmg >= owed {
                (ti + 1, 0)
            } else {
                (ti, owed - dmg)
            };
            chosen.push((ai, h));
            let r = positions_dfs(
                targets,
                order,
                nti,
                nowed,
                atk,
                rings,
                used | (1 << ai),
                hexes | bit(h),
                chosen,
                budget,
                found,
            );
            chosen.pop();
            match r {
                Some(true) => return Some(true),
                None => unknown = true,
                Some(false) => {}
            }
            if *budget == 0 {
                return None;
            }
        }
    }
    if unknown {
        None
    } else {
        Some(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Walk random step paths through whole games. Every completed path must
    /// be a legal turn that hands over the move, and dead ends (a step with no
    /// option) must be rare; the search prunes them. Restarts the turn on one.
    /// How many options the witness system drops that the exact search can
    /// realize (run with --ignored --nocapture; a measurement, not a check).
    #[test]
    #[ignore]
    fn witness_drop_rate() {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let (mut steps, mut offered, mut rejected, mut dropped, mut unknown, mut false_offer) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut dropped_stage = [0usize; 2];
        let t0 = std::time::Instant::now();
        let mut exact_ns = 0u128;
        let games: u64 = std::env::var("DROP_GAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        for g in 0..games {
            let (real, _) = crate::selfplay::random_opening(16, 1.0884, &mut rng);
            let mut s = StepState::begin(&real, g);
            let mut turns = 0;
            while !s.is_terminal() && turns < 30 {
                let opts = s.children();
                assert!(!opts.is_empty());
                if matches!(s.stage, Stage::Fates | Stage::Moves) {
                    steps += 1;
                    let (cands, fated, decided): (Vec<u8>, Bb, Bb) = match s.stage {
                        Stage::Fates => {
                            let e = s.enemies[s.idx];
                            let fated = s.enemies[..=s.idx].iter().fold(0, |b, &x| b | bit(x));
                            let _ = e;
                            (vec![TARGET_STAY, TARGET_KILL, TARGET_UNSUMMON], fated, 0)
                        }
                        _ => {
                            let c = s.friends[s.idx];
                            let fated = s.enemies.iter().fold(0, |b, &x| b | bit(x));
                            let decided = s.friends[..=s.idx].iter().fold(0, |b, &x| b | bit(x));
                            let ks = (0..N_OFFSETS as u8)
                                .filter(|&k| MOVE_DEST[c as usize][k as usize] != NO_CELL)
                                .collect();
                            (ks, fated, decided)
                        }
                    };
                    for v in cands {
                        let mut spec = s.spec;
                        match s.stage {
                            Stage::Fates => {
                                let e = s.enemies[s.idx];
                                if v == TARGET_UNSUMMON
                                    && s.base
                                        .piece_at(e)
                                        .is_some_and(|(_, u)| u.stats().persistent)
                                {
                                    continue;
                                }
                                spec.targets[e as usize] = v;
                            }
                            _ => spec.moves[s.friends[s.idx] as usize] = v,
                        }
                        let te = std::time::Instant::now();
                        let r = exact_realize(&s.base, &spec, decided, fated, EXACT_BUDGET);
                        exact_ns += te.elapsed().as_nanos();
                        let is_offered = opts.contains(&v);
                        match r {
                            Exact::Found(..) => {
                                if is_offered {
                                    offered += 1;
                                } else {
                                    dropped += 1;
                                    dropped_stage[usize::from(s.stage == Stage::Moves)] += 1;
                                    if s.stage == Stage::Moves && dropped <= 8 {
                                        let c = s.friends[s.idx];
                                        let d = MOVE_DEST[c as usize][v as usize];
                                        let mut stay_e: Bb = 0;
                                        for e in bitboard::cells(side_bb(&s.base, 1)) {
                                            if s.spec.targets[e as usize] == TARGET_STAY {
                                                stay_e |= bit(e);
                                            }
                                        }
                                        let stayers = s.friends[..s.idx]
                                            .iter()
                                            .filter(|&&f| s.spec.moves[f as usize] == 0)
                                            .fold(0, |b, &f| b | bit(f));
                                        let mut sp2 = s.spec;
                                        sp2.moves[c as usize] = v;
                                        let dec: Vec<u8> = s.friends[..=s.idx].to_vec();
                                        eprintln!(
                                            "gates: movers_completable {} relaxed_feasible {}",
                                            s.movers_completable(&sp2, s.claimed | bit(d), c, v),
                                            relaxed_feasible(&s.base, &sp2, &dec, 0)
                                        );
                                        eprintln!(
                                            "DROP move c {c} k {v} d {d}: claimed {} stay_enemy {} stayer {} cycle {} can_move {} last {} witness_k {}",
                                            has(s.claimed, d), has(stay_e, d), has(stayers, d),
                                            v != 0 && s.closes_cycle(c, d), s.base.can_move_piece(c),
                                            s.idx + 1 == s.friends.len(), s.witness.moves[c as usize],
                                        );
                                    }
                                }
                            }
                            Exact::Impossible => {
                                if is_offered {
                                    false_offer += 1;
                                    if false_offer <= 6 {
                                        let (o, ws) = match s.stage {
                                            Stage::Fates => s.fate_options(),
                                            _ => s.move_options(),
                                        };
                                        let w = ws[o.iter().position(|&x| x == v).unwrap()];
                                        let mut sp = w;
                                        sp.spawns = [0; NUM_CELLS];
                                        let r = crate::turn::realize_attack(&s.base, &sp);
                                        let full = exact_realize(
                                            &s.base,
                                            &w,
                                            side_bb(&s.base, 0),
                                            side_bb(&s.base, 1),
                                            1_000_000,
                                        );
                                        let mut diffs = Vec::new();
                                        for c in 0..NUM_CELLS {
                                            if w.targets[c] != r.spec.targets[c] {
                                                diffs.push(format!(
                                                    "target {c}: decl {} got {}",
                                                    w.targets[c], r.spec.targets[c]
                                                ));
                                            }
                                            if s.base
                                                .piece_at(c as u8)
                                                .is_some_and(|(sd, _)| sd == 0)
                                                && w.moves[c] != r.spec.moves[c]
                                                && !(w.moves[c] == NO_MOVE && r.spec.moves[c] == 0)
                                            {
                                                diffs.push(format!(
                                                    "move {c}: decl {} got {}",
                                                    w.moves[c], r.spec.moves[c]
                                                ));
                                            }
                                        }
                                        eprintln!(
                                            "FALSE OFFER stage {:?} v {v}: witness vs fast realize: fid {:?} winner {} diffs {:?}; exact on full witness: {}; fates {:?}",
                                            s.stage, r.fidelity, r.next.winner, diffs,
                                            match full { Exact::Found(..) => "found", Exact::Impossible => "impossible", Exact::Unknown => "unknown" },
                                            s.enemies[..s.idx.min(s.enemies.len())].iter().map(|&e| (e, s.spec.targets[e as usize])).collect::<Vec<_>>(),
                                        );
                                    }
                                } else {
                                    rejected += 1;
                                }
                            }
                            Exact::Unknown => unknown += 1,
                        }
                    }
                }
                let before = (s.flipped, s.base.turn);
                s = s.child(opts[rng.gen_range(0..opts.len())]);
                if (s.flipped, s.base.turn) != before {
                    turns += 1;
                }
            }
            eprintln!(
                "game {g}: {steps} steps, dropped {dropped}, offered {offered}, false offers {false_offer}, unknown {unknown}, exact {:.1}s",
                exact_ns as f64 * 1e-9
            );
        }
        eprintln!(
            "{steps} fate/move steps: offered+feasible {offered}, rejected+infeasible {rejected}, \
             DROPPED (feasible, not offered) {dropped} [fates {}, moves {}], offered but exact says \
             impossible {false_offer}, exact budget-out {unknown}; exact search {:.1}s of {:.1}s",
            dropped_stage[0],
            dropped_stage[1],
            exact_ns as f64 * 1e-9,
            t0.elapsed().as_secs_f64()
        );
    }

    /// `turn_actions` replays each turn the step tree plays: the atomic moves
    /// lead from the turn's real start to the real position the tree reached.
    #[test]
    fn turn_actions_replay_step_turns() {
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let (mut turns, mut games) = (0, 0);
        for g in 0..30u64 {
            let (mut real, _) = crate::selfplay::random_opening(8, 1.0884, &mut rng);
            let mut s = StepState::begin(&real, g);
            while !s.is_terminal() && turns < 2000 {
                let ch = s.children();
                assert!(!ch.is_empty(), "dead end");
                let before = s.flipped;
                let turn_no = s.base.turn;
                s = s.child(ch[rng.gen_range(0..ch.len())]);
                if s.flipped != before || s.is_terminal() || s.base.turn != turn_no {
                    let acts = turn_actions(&real, &s.done).unwrap();
                    let mut r = real.clone();
                    for a in &acts {
                        assert!(
                            Minions::legal_moves(&r).contains(a),
                            "illegal {a:?} in turn {turns}"
                        );
                        r = Minions::apply(&r, a);
                    }
                    assert_eq!(
                        Minions::encode_state(&r),
                        Minions::encode_state(&s.real_base()),
                        "turn {turns}: actions {acts:?}"
                    );
                    real = s.real_base();
                    turns += 1;
                }
            }
            games += 1;
        }
        eprintln!("{turns} turns over {games} games replayed");
        assert!(turns > 200);
    }

    #[test]
    fn random_step_paths_are_legal_turns() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut turns, mut options, mut steps, mut dead, mut stuck) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut dead_at = [0usize; 3];
        let mut cls = [0usize; 4];
        let mut reasons = [0usize; 6];
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
                        if s.is_terminal()
                            || s.flipped != start.flipped
                            || s.base.turn != start.base.turn
                        {
                            break true;
                        }
                        let ch = s.children();
                        if ch.is_empty() {
                            if s.stage == Stage::Moves {
                                let rr = s.reject_reasons();
                                if std::env::var("DUMP_DEAD").is_ok()
                                    && s.friends.len() <= 7
                                    && s.idx + 3 <= s.friends.len()
                                {
                                    eprintln!(
                                        "=== DEAD idx {}/{} next unit at {} {:?}",
                                        s.idx,
                                        s.friends.len(),
                                        s.friends[s.idx],
                                        s.base.piece_at(s.friends[s.idx])
                                    );
                                    for cc in 0..NUM_CELLS as u8 {
                                        if let Some(pc) = s.base.piece_at(cc) {
                                            let k = s.spec.moves[cc as usize];
                                            let d = if k == NO_MOVE {
                                                255
                                            } else {
                                                MOVE_DEST[cc as usize][k as usize]
                                            };
                                            let ord = s.friends.iter().position(|&f| f == cc);
                                            eprintln!("  cell {cc:2} (r{} c{}) {:?} order {:?} move->{d} target {}", cc / 7, cc % 7, pc, ord, s.spec.targets[cc as usize]);
                                        }
                                    }
                                    eprintln!(
                                        "  claimed {:?} reasons {:?}",
                                        bitboard::cells(s.claimed).collect::<Vec<_>>(),
                                        rr
                                    );
                                }
                                for (a, b) in reasons.iter_mut().zip(rr) {
                                    *a += b;
                                }
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
            "{turns} turns, {steps} choice steps, {:.1} options/step, {dead} dead ends (fates/moves/spawns {dead_at:?}), {stuck} turns never finished; [notlast/nokill, notlast/kill, last/nokill, last/kill] {cls:?}; option rejections [blocked, flow, relaxed, lookahead, last-realize, ok] {reasons:?}",
            options as f64 / steps as f64
        );
        assert!(turns > 200, "played {turns} turns");
        // Every option carries a realizable witness, so dead ends should be
        // (nearly) impossible; the search prunes any that slip through.
        assert!(
            dead * 50 < turns,
            "dead ends too common: {dead} in {turns} turns"
        );
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
