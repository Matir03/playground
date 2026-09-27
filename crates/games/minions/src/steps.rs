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
pub const SPARSE_PLANES: usize = 17;
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
        s.friends = bitboard::cells(side_bb(&s.base, 0)).collect();
        s.friends.shuffle(&mut rng);
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
        *self = StepState::begin(&real, self.salt);
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
        // Staying is always completable: the current witness leaves every
        // undecided enemy alone.
        let mut stay = self.witness;
        stay.targets[e as usize] = TARGET_STAY;
        let (mut out, mut ws) = (vec![TARGET_STAY], vec![stay]);
        for v in [TARGET_KILL, TARGET_UNSUMMON] {
            if v == TARGET_UNSUMMON && t.stats().persistent {
                continue;
            }
            let mut spec = self.spec;
            spec.targets[e as usize] = v;
            let mut cand = self.witness;
            cand.targets[e as usize] = v;
            if attack_part(&self.base, &cand).is_some() {
                out.push(v);
                ws.push(cand);
                continue;
            }
            if !relaxed_feasible(&self.base, &spec, &[]) {
                continue;
            }
            if let Some(w) = build_witness(&self.base, &spec, &[]) {
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
            if attack_part(&self.base, &cand).is_some() {
                out.push(k);
                ws.push(cand);
                continue;
            }
            if !self.movers_completable(&spec, self.claimed | bit(d), c, k) {
                continue;
            }
            let mut dec = decided.clone();
            dec.push(c);
            if !relaxed_feasible(&self.base, &spec, &dec) {
                continue;
            }
            if let Some(w) = build_witness(&self.base, &spec, &dec) {
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
            if !relaxed_feasible(&self.base, &spec, &dec) {
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
        };
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
pub fn attack_part(base: &MinionsState, spec: &TurnSpec) -> Option<MinionsState> {
    let mut sp = *spec;
    sp.spawns = [0; NUM_CELLS];
    let r = crate::turn::realize_attack(base, &sp);
    let s = r.next;
    if s.winner != 0 {
        return Some(s); // a won game needs nothing else declared to happen
    }
    let f = r.fidelity;
    if f.done_moves == f.declared_moves && f.done_kills == f.declared_kills {
        return Some(s);
    }
    // The realizer plans every kill at once; some turns need one kill to open
    // a hex that a unit moves into before it attacks. Try one kill at a time,
    // each declared target first in turn.
    let n_targets = (0..NUM_CELLS)
        .filter(|&c| sp.targets[c] != TARGET_STAY)
        .count();
    (0..n_targets.min(6)).find_map(|first| realize_sequential(base, &sp, first))
}

/// Moves whenever possible, then a single declared kill (targets tried from
/// `first` onward), repeat. Returns the position after the attack phase if every
/// declared move and kill happened.
fn realize_sequential(base: &MinionsState, spec: &TurnSpec, first: usize) -> Option<MinionsState> {
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
                s = Minions::apply(&s, &MinionsMove::Move { from: o, to: d });
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
                s = Minions::apply(&s, &MinionsMove::Attack { from: a, to: t });
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
    Some(s)
}

/// Friendly units that can still attack, with the hexes each could end on:
/// a decided unit its destination (none if lumbering and moved); an undecided
/// one its origin (unless claimed) plus reachable hexes not in `block`.
fn attackers(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: &[u8],
    block: Bb,
) -> Vec<(u8, Unit, Bb)> {
    let mut staying_enemies: Bb = 0;
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != 0)
            && spec.targets[e as usize] == TARGET_STAY
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
fn relaxed_feasible(base: &MinionsState, spec: &TurnSpec, decided: &[u8]) -> bool {
    let atk = attackers(base, spec, decided, 0);
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
    let atk = attackers(base, spec, decided, und_origins);
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
        if attack_part(base, &w).is_some() {
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
