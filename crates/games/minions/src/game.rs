//! Rules for the simplified Minions variant.
//!
//! ## What this is
//!
//! Minions (Matir03/minions) is a two-team game played across several 10×10 hex
//! boards: a *General* spends money on spells to walk a shared techline and
//! unlock units, while *Captains* fight for each board, and the first team to
//! `n - floor(n/4)` board points wins. This crate keeps the board fight and
//! throws away everything above it:
//!
//! - **One board.** No blotto across boards, no board resets, no reinforcement
//!   zones carried between boards.
//! - **One board point.** Winning the board wins the game, so board points need
//!   no representation at all.
//! - **No generaling.** No techline, no spells, no `$8n` spell purchases. The
//!   five buyable units ([`crate::units::BUYABLE`]) are unlocked from move one.
//! - **Five units + necromancer.** `zombie` (the always-acquired basic) plus the
//!   first four techline cards `skeleton`, `serpent`, `warg`, `ghost`. Only the
//!   basic necromancer exists, so the "choose an advanced necromancer" setup
//!   phase is gone with the resets.
//! - **All land** — see [`crate::map`].
//!
//! Also dropped, because they only make the action space bigger without
//! changing what the game is about: cyclic (swap) moves, blink, spell casting,
//! and resignation.
//!
//! ## What is kept, faithfully
//!
//! Unit stats and keywords; the attack/spawn phase split; damage that
//! accumulates within a turn and clears at end of turn; `lumbering` barring
//! move-then-attack; `flying` moving through enemies; `persistent` converting
//! `Unsummon` into 1 damage; unsummoned units bouncing to reinforcements;
//! rebates paid to the *owner* of a dying unit; income of
//! `graveyards + 2 + 1 soul`; the first player's opening turn having no spawn
//! phase; and the two ways to take a board — kill the enemy necromancer, or
//! hold [`GRAVEYARDS_TO_WIN`] graveyards at the end of the opponent's turn.
//!
//! ## Turn shape
//!
//! A turn is a *sequence* of [`MinionsMove`]s, the way an Arimaa turn is a
//! sequence of steps, ending with the [`MinionsMove::EndPhase`] that closes the
//! spawn phase. Attack phase → spawn phase → end of turn; each `EndPhase`
//! advances one step. Every position therefore has at least one legal move.
//!
//! Games that reach [`MAX_TURNS`] are decided by [`adjudicate`] rather than
//! played out — the full game has no draws, but two passive players can shuffle
//! forever, and self-play needs a bound.

use playground_core::{Game, Outcome, Player, SearchOutcome, Solver};

use crate::map::{
    self, cell_name, dist, IS_GRAVEYARD, NEIGHBORS, NECRO_START, NO_CELL, NUM_CELLS,
};
use crate::units::{Attack, Unit, BUYABLE, NUM_UNITS};

pub use crate::map::GRAVEYARDS_TO_WIN;

/// Money the second player starts with, compensating for moving second. The
/// full game's `START_MONEY_PER_BOARD`, for our one board.
pub const START_MONEY_SECOND: u16 = 6;
/// Flat part of end-of-turn income: `+2` per board, `+1` soul for the basic
/// necromancer.
pub const BASE_INCOME: u16 = 3;
/// Turns (counting each side's separately, so this is 20 apiece) after which the
/// game is decided by [`adjudicate`] instead of being played out.
pub const MAX_TURNS: u16 = 40;

/// Feature planes produced by [`Game::to_features`].
pub const PLANES: usize = 31;

/// Who wins a game that hits [`MAX_TURNS`]: **more gold wins**, exact ties draw.
///
/// Deliberately the single place this rule lives, because it is load-bearing and
/// suspect. Most games hit the cap (the greedy heuristic finishes around turn 33),
/// so this is the de-facto objective rather than a tiebreak — and gold is *spent*
/// on units, so scoring raw gold rewards hoarding over developing. The degenerate
/// line: with both sides passing and no graveyards contested, income is
/// [`BASE_INCOME`] a turn each, so after 20 turns apiece the first player holds 60
/// and the second 66 from [`START_MONEY_SECOND`] — "pass forever" wins for the
/// second player.
///
/// If the self-play spend-rate watchdog shows that happening, the spend-neutral
/// fix is one line here: add the rebate value of units on the board (liquidation
/// value), so converting gold into units costs nothing.
pub fn adjudicate(state: &MinionsState) -> u8 {
    match state.money[0].cmp(&state.money[1]) {
        std::cmp::Ordering::Greater => 1, // first wins
        std::cmp::Ordering::Less => 2,    // second wins
        std::cmp::Ordering::Equal => 3,   // draw
    }
}

// ── Phases ────────────────────────────────────────────────────────────────────

pub const PHASE_ATTACK: u8 = 0;
pub const PHASE_SPAWN: u8 = 1;

// ── Cell / piece-state packing ────────────────────────────────────────────────
//
// cells[i]: 0 = empty, otherwise 1 + unit_index + 6*side.
// pstate[i]: bit0 moved, bit1 attacked, bit2 exhausted, bits3..5 damage taken.

const MOVED: u8 = 1 << 0;
const ATTACKED: u8 = 1 << 1;
const EXHAUSTED: u8 = 1 << 2;
const DAMAGE_SHIFT: u8 = 3;
const DAMAGE_MASK: u8 = 0b111 << DAMAGE_SHIFT;

#[inline]
fn encode_cell(side: u8, unit: Unit) -> u8 {
    1 + unit as u8 + NUM_UNITS as u8 * side
}

#[inline]
fn cell_side(c: u8) -> u8 {
    (c - 1) / NUM_UNITS as u8
}

#[inline]
fn cell_unit(c: u8) -> Unit {
    Unit::from_index((c - 1) % NUM_UNITS as u8).expect("cell holds a valid unit")
}

#[inline]
fn damage_of(p: u8) -> u8 {
    (p & DAMAGE_MASK) >> DAMAGE_SHIFT
}

#[inline]
fn with_damage(p: u8, d: u8) -> u8 {
    (p & !DAMAGE_MASK) | ((d & 0b111) << DAMAGE_SHIFT)
}

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinionsState {
    /// Occupancy, one byte per cell. See the packing note above.
    pub cells: [u8; NUM_CELLS],
    /// Per-piece turn state, parallel to `cells` (0 where empty).
    pub pstate: [u8; NUM_CELLS],
    /// Money per side, indexed by side (0 = first).
    pub money: [u16; 2],
    /// Reinforcements per side, indexed by [`Unit::reinforcement_slot`].
    pub reinforcements: [[u8; BUYABLE.len()]; 2],
    /// 0 = first player to move, 1 = second.
    pub side: u8,
    /// [`PHASE_ATTACK`] or [`PHASE_SPAWN`].
    pub phase: u8,
    /// Turns completed. Turn 0 is the first player's opening turn, which has no
    /// spawn phase.
    pub turn: u16,
    /// Monotonic count of *actions*, not turns — this is `Game::ply`.
    pub ply: u32,
    /// 0 = in progress, 1 = first wins, 2 = second wins, 3 = draw.
    pub winner: u8,
}

impl Default for MinionsState {
    fn default() -> Self {
        let mut cells = [0u8; NUM_CELLS];
        for side in 0..2usize {
            cells[NECRO_START[side] as usize] = encode_cell(side as u8, Unit::Necromancer);
            for z in map::start_zombies(side) {
                cells[z as usize] = encode_cell(side as u8, Unit::Zombie);
            }
        }
        Self {
            cells,
            pstate: [0; NUM_CELLS],
            money: [0, START_MONEY_SECOND],
            reinforcements: [[0; BUYABLE.len()]; 2],
            side: 0,
            phase: PHASE_ATTACK,
            turn: 0,
            ply: 0,
            winner: 0,
        }
    }
}

impl MinionsState {
    #[inline]
    pub fn piece_at(&self, i: u8) -> Option<(u8, Unit)> {
        let c = self.cells[i as usize];
        if c == 0 {
            None
        } else {
            Some((cell_side(c), cell_unit(c)))
        }
    }

    /// Graveyards `side` currently occupies — its income, and its progress
    /// toward a graveyard-control win.
    pub fn graveyards_held(&self, side: u8) -> u32 {
        (0..NUM_CELLS)
            .filter(|&i| IS_GRAVEYARD[i] && self.cells[i] != 0 && cell_side(self.cells[i]) == side)
            .count() as u32
    }

    /// Empty cells adjacent to one of `side`'s spawners (necromancer, zombie).
    /// A unit spawned this turn is a spawner immediately, as in the full game.
    pub fn spawn_cells(&self, side: u8) -> Vec<u8> {
        let mut seen = [false; NUM_CELLS];
        let mut out = Vec::new();
        for (i, &c) in self.cells.iter().enumerate() {
            if c == 0 || cell_side(c) != side || !cell_unit(c).stats().spawn {
                continue;
            }
            for &n in &NEIGHBORS[i] {
                if n != NO_CELL && self.cells[n as usize] == 0 && !seen[n as usize] {
                    seen[n as usize] = true;
                    out.push(n);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Cells a piece at `from` may move to: empty, and reachable within its
    /// speed through cells that are empty or friendly (or anything, if flying).
    pub fn move_targets(&self, from: u8) -> Vec<u8> {
        let Some((side, unit)) = self.piece_at(from) else {
            return Vec::new();
        };
        let stats = unit.stats();
        let flying = stats.flying;

        let mut steps = [u8::MAX; NUM_CELLS];
        steps[from as usize] = 0;
        let mut frontier = vec![from];
        let mut out = Vec::new();

        for step in 1..=stats.speed {
            let mut next = Vec::new();
            for &cur in &frontier {
                for &n in &NEIGHBORS[cur as usize] {
                    if n == NO_CELL || steps[n as usize] != u8::MAX {
                        continue;
                    }
                    let occ = self.cells[n as usize];
                    // Enemy units block the path unless we fly over them.
                    if occ != 0 && !flying && cell_side(occ) != side {
                        continue;
                    }
                    steps[n as usize] = step;
                    if occ == 0 {
                        out.push(n);
                    }
                    next.push(n);
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        out.sort_unstable();
        out
    }

    /// Damage a piece has taken so far this turn.
    #[inline]
    pub fn damage_on(&self, i: u8) -> u8 {
        damage_of(self.pstate[i as usize])
    }

    /// True if the piece on `i` may still move this turn.
    #[inline]
    pub fn can_move_piece(&self, i: u8) -> bool {
        let p = self.pstate[i as usize];
        p & (MOVED | EXHAUSTED | ATTACKED) == 0
    }

    /// True if the piece on `i` may still attack this turn. `lumbering` units
    /// that have already moved cannot.
    #[inline]
    pub fn can_attack_piece(&self, i: u8, unit: Unit) -> bool {
        let p = self.pstate[i as usize];
        if p & (EXHAUSTED | ATTACKED) != 0 {
            return false;
        }
        // Lumbering units cannot move and attack in the same turn.
        !(unit.stats().lumbering && p & MOVED != 0)
    }

    /// Enemy pieces a piece at `from` may attack right now.
    pub fn attack_targets(&self, from: u8) -> Vec<u8> {
        let Some((side, unit)) = self.piece_at(from) else {
            return Vec::new();
        };
        if !self.can_attack_piece(from, unit) {
            return Vec::new();
        }
        let range = unit.stats().range as u32;
        (0..NUM_CELLS as u8)
            .filter(|&i| {
                let c = self.cells[i as usize];
                c != 0 && cell_side(c) != side && dist(from, i) <= range
            })
            .collect()
    }

    /// Debug render: `y` descending so the second player is at the top, rows
    /// indented to show the rhombus's lean (`+y` runs up and to the right).
    /// Uppercase = first player, `$` = an empty graveyard.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for y in (0..map::N).rev() {
            s.push_str(&format!("{y} "));
            for _ in 0..y {
                s.push(' ');
            }
            for x in 0..map::N {
                let i = map::idx(x as u8, y as u8) as usize;
                let c = self.cells[i];
                let ch = if c == 0 {
                    if IS_GRAVEYARD[i] { '$' } else { '.' }
                } else {
                    let u = cell_unit(c).stats().ch;
                    if cell_side(c) == 0 { u } else { u.to_ascii_lowercase() }
                };
                s.push(ch);
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str(&format!(
            "side={} phase={} turn={} money={:?} gy={}:{}\n",
            if self.side == 0 { "first" } else { "second" },
            if self.phase == PHASE_ATTACK { "attack" } else { "spawn" },
            self.turn,
            self.money,
            self.graveyards_held(0),
            self.graveyards_held(1),
        ));
        s
    }
}

// ── Moves ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinionsMove {
    /// Attack phase: walk a piece to an empty cell within its speed.
    Move { from: u8, to: u8 },
    /// Attack phase: strike an enemy within range.
    Attack { from: u8, to: u8 },
    /// Spawn phase: pay a unit's cost to put it in reinforcements.
    Buy(Unit),
    /// Spawn phase: place a reinforcement next to one of your spawners.
    Spawn { unit: Unit, at: u8 },
    /// Close the current phase: attack → spawn, spawn → end of turn. On the
    /// first player's opening turn there is no spawn phase, so this ends it.
    EndPhase,
}

impl std::fmt::Display for MinionsMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MinionsMove::Move { from, to } => write!(f, "move {}{}", cell_name(*from), cell_name(*to)),
            MinionsMove::Attack { from, to } => write!(f, "attack {}{}", cell_name(*from), cell_name(*to)),
            MinionsMove::Buy(u) => write!(f, "buy {}", u.stats().name),
            MinionsMove::Spawn { unit, at } => write!(f, "spawn {} {}", unit.stats().name, cell_name(*at)),
            MinionsMove::EndPhase => write!(f, "end"),
        }
    }
}

// ── Rules ─────────────────────────────────────────────────────────────────────

pub struct Minions;

impl Minions {
    /// Resolve one attack. Mirrors `Board::try_attack` / `Board::attack_piece`.
    fn resolve_attack(state: &mut MinionsState, from: u8, to: u8) {
        let (att_side, att_unit) = state.piece_at(from).expect("attacker exists");
        let (def_side, def_unit) = state.piece_at(to).expect("target exists");
        let def_stats = def_unit.stats();

        state.pstate[from as usize] |= ATTACKED;

        let (damage, bounce) = match att_unit.stats().attack {
            Attack::Damage(d) => (d, false),
            // Persistent units shrug off unsummon for 1 damage instead.
            Attack::Unsummon => {
                if def_stats.persistent {
                    (1, false)
                } else {
                    (0, true)
                }
            }
        };

        let total = damage_of(state.pstate[to as usize]) + damage;
        if !bounce && total < def_stats.defense {
            state.pstate[to as usize] = with_damage(state.pstate[to as usize], total);
            return;
        }

        // The target leaves the board.
        state.cells[to as usize] = 0;
        state.pstate[to as usize] = 0;

        if bounce {
            let slot = def_unit.reinforcement_slot().expect("necromancers are persistent");
            state.reinforcements[def_side as usize][slot] += 1;
        } else if def_stats.necromancer {
            // Killing the necromancer takes the board, and the board is the game.
            state.winner = att_side + 1;
        } else {
            state.money[def_side as usize] += def_stats.rebate;
        }
    }

    /// End-of-turn sequence: clear per-turn piece state, collect income, check
    /// the graveyard-control loss, then hand over. Mirrors `Board::end_turn`.
    fn end_turn(state: &mut MinionsState) {
        for i in 0..NUM_CELLS {
            state.pstate[i] = 0;
        }

        let side = state.side as usize;
        state.money[side] += state.graveyards_held(state.side) as u16 + BASE_INCOME;

        if state.winner == 0 && state.graveyards_held(1 - state.side) >= GRAVEYARDS_TO_WIN {
            state.winner = (1 - state.side) + 1;
        }

        state.turn += 1;
        if state.winner == 0 && state.turn >= MAX_TURNS {
            state.winner = adjudicate(state);
        }

        state.side = 1 - state.side;
        state.phase = PHASE_ATTACK;
    }
}

impl Game for Minions {
    type State = MinionsState;
    type Move = MinionsMove;

    fn id() -> u32 {
        6
    }

    fn initial_state() -> Self::State {
        MinionsState::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.side == 0 { Player::First } else { Player::Second }
    }

    fn outcome(state: &Self::State) -> Outcome {
        match state.winner {
            1 => Outcome::FirstWins,
            2 => Outcome::SecondWins,
            3 => Outcome::Draw,
            _ => Outcome::InProgress,
        }
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if Self::outcome(state).is_terminal() {
            return Vec::new();
        }
        let side = state.side;
        let mut out = Vec::new();

        if state.phase == PHASE_ATTACK {
            for i in 0..NUM_CELLS as u8 {
                let Some((s, _)) = state.piece_at(i) else { continue };
                if s != side {
                    continue;
                }
                if state.can_move_piece(i) {
                    for to in state.move_targets(i) {
                        out.push(MinionsMove::Move { from: i, to });
                    }
                }
                for to in state.attack_targets(i) {
                    out.push(MinionsMove::Attack { from: i, to });
                }
            }
        } else {
            for unit in BUYABLE {
                if state.money[side as usize] >= unit.stats().cost {
                    out.push(MinionsMove::Buy(unit));
                }
            }
            let cells = state.spawn_cells(side);
            if !cells.is_empty() {
                for unit in BUYABLE {
                    let slot = unit.reinforcement_slot().expect("buyable units have a slot");
                    if state.reinforcements[side as usize][slot] == 0 {
                        continue;
                    }
                    for &at in &cells {
                        out.push(MinionsMove::Spawn { unit, at });
                    }
                }
            }
        }

        // Always available, so no position is ever stuck.
        out.push(MinionsMove::EndPhase);
        out
    }

    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State {
        let mut next = state.clone();
        match *mv {
            MinionsMove::Move { from, to } => {
                next.cells[to as usize] = next.cells[from as usize];
                next.pstate[to as usize] = next.pstate[from as usize] | MOVED;
                next.cells[from as usize] = 0;
                next.pstate[from as usize] = 0;
            }
            MinionsMove::Attack { from, to } => Self::resolve_attack(&mut next, from, to),
            MinionsMove::Buy(unit) => {
                let side = next.side as usize;
                next.money[side] -= unit.stats().cost;
                next.reinforcements[side][unit.reinforcement_slot().unwrap()] += 1;
            }
            MinionsMove::Spawn { unit, at } => {
                let side = next.side as usize;
                next.reinforcements[side][unit.reinforcement_slot().unwrap()] -= 1;
                next.cells[at as usize] = encode_cell(next.side, unit);
                // Spawned units are exhausted for the rest of the turn.
                next.pstate[at as usize] = EXHAUSTED;
            }
            MinionsMove::EndPhase => {
                // Turn 0 — the first player's opening — has no spawn phase.
                if next.phase == PHASE_ATTACK && next.turn > 0 {
                    next.phase = PHASE_SPAWN;
                } else {
                    Self::end_turn(&mut next);
                }
            }
        }
        next.ply += 1;
        next
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // Canonical frame: own units first, so the net sees one point of view.
        let me = state.side;
        let mut out = Vec::with_capacity(PLANES * NUM_CELLS);
        fn plane(f: &mut dyn FnMut(usize) -> u8, out: &mut Vec<u8>) {
            for i in 0..NUM_CELLS {
                out.push(f(i));
            }
        }

        for side in [me, 1 - me] {
            for u in 0..NUM_UNITS as u8 {
                let want = encode_cell(side, Unit::from_index(u).unwrap());
                plane(&mut |i| (state.cells[i] == want) as u8, &mut out);
            }
        }
        plane(&mut |i| damage_of(state.pstate[i]), &mut out);
        plane(&mut |i| (state.pstate[i] & MOVED != 0) as u8, &mut out);
        plane(&mut |i| (state.pstate[i] & ATTACKED != 0) as u8, &mut out);
        plane(&mut |i| (state.pstate[i] & EXHAUSTED != 0) as u8, &mut out);
        plane(&mut |i| IS_GRAVEYARD[i] as u8, &mut out);
        plane(&mut |_| state.side, &mut out);
        plane(&mut |_| (state.phase == PHASE_SPAWN) as u8, &mut out);
        for side in [me, 1 - me] {
            let m = state.money[side as usize].min(255) as u8;
            plane(&mut |_| m, &mut out);
        }
        for side in [me, 1 - me] {
            for slot in 0..BUYABLE.len() {
                let n = state.reinforcements[side as usize][slot];
                plane(&mut |_| n, &mut out);
            }
        }
        debug_assert_eq!(out.len(), PLANES * NUM_CELLS);
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(STATE_BYTES);
        out.extend_from_slice(&state.cells);
        out.extend_from_slice(&state.pstate);
        for m in state.money {
            out.extend_from_slice(&m.to_le_bytes());
        }
        for side in state.reinforcements {
            out.extend_from_slice(&side);
        }
        out.push(state.side);
        out.push(state.phase);
        out.extend_from_slice(&state.turn.to_le_bytes());
        out.extend_from_slice(&state.ply.to_le_bytes());
        out.push(state.winner);
        debug_assert_eq!(out.len(), STATE_BYTES);
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        if bytes.len() != STATE_BYTES {
            anyhow::bail!("minions state: expected {STATE_BYTES} bytes, got {}", bytes.len());
        }
        let mut s = MinionsState {
            cells: [0; NUM_CELLS],
            pstate: [0; NUM_CELLS],
            money: [0; 2],
            reinforcements: [[0; BUYABLE.len()]; 2],
            side: 0,
            phase: PHASE_ATTACK,
            turn: 0,
            ply: 0,
            winner: 0,
        };
        let mut o = 0;
        s.cells.copy_from_slice(&bytes[o..o + NUM_CELLS]);
        o += NUM_CELLS;
        s.pstate.copy_from_slice(&bytes[o..o + NUM_CELLS]);
        o += NUM_CELLS;
        for m in s.money.iter_mut() {
            *m = u16::from_le_bytes([bytes[o], bytes[o + 1]]);
            o += 2;
        }
        for side in s.reinforcements.iter_mut() {
            side.copy_from_slice(&bytes[o..o + BUYABLE.len()]);
            o += BUYABLE.len();
        }
        s.side = bytes[o];
        s.phase = bytes[o + 1];
        s.turn = u16::from_le_bytes([bytes[o + 2], bytes[o + 3]]);
        s.ply = u32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]]);
        s.winner = bytes[o + 8];

        for (i, &c) in s.cells.iter().enumerate() {
            if c as usize > 2 * NUM_UNITS {
                anyhow::bail!("minions state: bad cell byte {c} at {}", cell_name(i as u8));
            }
        }
        if s.side > 1 || s.phase > PHASE_SPAWN || s.winner > 3 {
            anyhow::bail!("minions state: bad side/phase/winner byte");
        }
        Ok(s)
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        match *mv {
            MinionsMove::Move { from, to } => vec![0, from, to],
            MinionsMove::Attack { from, to } => vec![1, from, to],
            MinionsMove::Buy(u) => vec![2, u as u8],
            MinionsMove::Spawn { unit, at } => vec![3, unit as u8, at],
            MinionsMove::EndPhase => vec![4],
        }
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        let unit = |b: u8| Unit::from_index(b).ok_or_else(|| anyhow::anyhow!("bad unit index {b}"));
        Ok(match *bytes {
            [0, from, to] => MinionsMove::Move { from, to },
            [1, from, to] => MinionsMove::Attack { from, to },
            [2, u] => MinionsMove::Buy(unit(u)?),
            [3, u, at] => MinionsMove::Spawn { unit: unit(u)?, at },
            [4] => MinionsMove::EndPhase,
            _ => anyhow::bail!("invalid minions move bytes (len={})", bytes.len()),
        })
    }
}

/// Wire size of an encoded state.
pub const STATE_BYTES: usize = NUM_CELLS   // cells
    + NUM_CELLS                            // pstate
    + 4                                    // money
    + 2 * BUYABLE.len()                    // reinforcements
    + 1 + 1 + 2 + 4 + 1; // side, phase, turn, ply, winner

impl Solver for Minions {
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move> {
        crate::solver::greedy_move(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::idx;

    fn play(state: &MinionsState, mvs: &[MinionsMove]) -> MinionsState {
        let mut s = state.clone();
        for mv in mvs {
            assert!(
                Minions::legal_moves(&s).contains(mv),
                "illegal move {mv} in\n{}",
                s.render()
            );
            s = Minions::apply(&s, mv);
        }
        s
    }

    /// Get both sides past the opening turn, into normal attack→spawn turns.
    fn after_opening() -> MinionsState {
        let s = Minions::initial_state();
        let s = play(&s, &[MinionsMove::EndPhase]); // first player, no spawn phase
        assert_eq!(s.side, 1);
        s
    }

    #[test]
    fn initial_position_matches_the_reference_setup() {
        let s = Minions::initial_state();
        assert_eq!(Minions::outcome(&s), Outcome::InProgress);
        assert_eq!(s.piece_at(NECRO_START[0]), Some((0, Unit::Necromancer)));
        assert_eq!(s.piece_at(NECRO_START[1]), Some((1, Unit::Necromancer)));
        for side in 0..2u8 {
            let zombies = (0..NUM_CELLS as u8)
                .filter(|&i| s.piece_at(i) == Some((side, Unit::Zombie)))
                .count();
            assert_eq!(zombies, 6, "side {side} should open with six zombies");
        }
        assert_eq!(s.money, [0, START_MONEY_SECOND]);
    }

    #[test]
    fn opening_turn_has_no_spawn_phase() {
        let s = Minions::initial_state();
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        assert_eq!(s.side, 1, "opening EndPhase should hand over, not open a spawn phase");
        assert_eq!(s.phase, PHASE_ATTACK);
        // The second player does get one.
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        assert_eq!(s.side, 1);
        assert_eq!(s.phase, PHASE_SPAWN);
    }

    #[test]
    fn a_position_always_has_a_legal_move() {
        let mut s = Minions::initial_state();
        for _ in 0..40 {
            let moves = Minions::legal_moves(&s);
            assert!(!moves.is_empty());
            s = Minions::apply(&s, &MinionsMove::EndPhase);
        }
    }

    #[test]
    fn zombies_are_lumbering() {
        let mut s = Minions::initial_state();
        // Put an enemy zombie next to one of ours, then move ours and try to attack.
        let from = idx(2, 3);
        let target = idx(2, 5);
        s.cells[target as usize] = encode_cell(1, Unit::Zombie);
        let dest = idx(2, 4);
        assert!(s.move_targets(from).contains(&dest));
        let moved = Minions::apply(&s, &MinionsMove::Move { from, to: dest });
        assert!(
            moved.attack_targets(dest).is_empty(),
            "a lumbering zombie must not move and attack in the same turn"
        );
        // Without moving first, the attack is on.
        s.cells[dest as usize] = encode_cell(0, Unit::Zombie);
        assert!(s.attack_targets(dest).contains(&target));
    }

    #[test]
    fn enemies_block_a_path_but_friends_do_not() {
        let mut s = Minions::initial_state();
        // c2..j9 is open ground. From f2 the only two-step route to f4 runs
        // through f3, so whatever sits on f3 decides whether f4 is reachable.
        let from = idx(5, 2);
        let corridor = idx(5, 3);
        let behind = idx(5, 4);
        s.cells[from as usize] = encode_cell(0, Unit::Serpent); // speed 2, grounded
        assert!(s.move_targets(from).contains(&behind));

        let mut blocked = s.clone();
        blocked.cells[corridor as usize] = encode_cell(1, Unit::Zombie);
        assert!(!blocked.move_targets(from).contains(&behind), "an enemy blocks the path");
        assert!(!blocked.move_targets(from).contains(&corridor), "and cannot be landed on");

        let mut friendly = s.clone();
        friendly.cells[corridor as usize] = encode_cell(0, Unit::Zombie);
        assert!(friendly.move_targets(from).contains(&behind), "a friend does not block");
        assert!(!friendly.move_targets(from).contains(&corridor), "but still occupies the cell");
    }

    #[test]
    fn flying_ignores_enemy_units_on_the_way() {
        // Both fliers in this roster have speed 1, so flying never actually
        // changes a legal move here — the rule is implemented against the day a
        // faster flier or water is added, and this pins the mechanic itself.
        let mut s = Minions::initial_state();
        for u in [Unit::Skeleton, Unit::Ghost] {
            assert_eq!(u.stats().speed, 1, "{} is no longer speed 1", u.stats().name);
        }
        let from = idx(5, 2);
        let occupied = idx(5, 3);
        s.cells[from as usize] = encode_cell(0, Unit::Ghost);
        s.cells[occupied as usize] = encode_cell(1, Unit::Zombie);
        let targets = s.move_targets(from);
        assert!(!targets.contains(&occupied), "no unit lands on an occupied cell");
        assert!(targets.iter().all(|&t| s.cells[t as usize] == 0));
    }

    #[test]
    fn damage_accumulates_within_a_turn_and_clears_after_it() {
        let mut s = after_opening();
        // Second player's ghost (defense 4) next to two of its enemies is not
        // our concern; put a first-player ghost next to a second-player warg.
        let ghost = idx(5, 5);
        let warg = idx(5, 6);
        s.side = 1;
        s.cells[ghost as usize] = encode_cell(0, Unit::Ghost);
        s.cells[warg as usize] = encode_cell(1, Unit::Warg); // 1 damage per hit

        let s1 = Minions::apply(&s, &MinionsMove::Attack { from: warg, to: ghost });
        assert_eq!(damage_of(s1.pstate[ghost as usize]), 1);
        assert!(s1.attack_targets(warg).is_empty(), "one attack per unit per turn");

        // End the turn: damage is wiped.
        let s2 = play(&s1, &[MinionsMove::EndPhase, MinionsMove::EndPhase]);
        assert_eq!(damage_of(s2.pstate[ghost as usize]), 0);
        assert_eq!(s2.side, 0);
    }

    #[test]
    fn a_kill_pays_the_rebate_to_the_owner() {
        let mut s = after_opening();
        let skeleton = idx(5, 5); // 5 damage
        let serpent = idx(5, 6); // defense 1, rebate 2
        s.side = 0;
        s.cells[skeleton as usize] = encode_cell(0, Unit::Skeleton);
        s.cells[serpent as usize] = encode_cell(1, Unit::Serpent);
        let before = s.money[1];

        let s = Minions::apply(&s, &MinionsMove::Attack { from: skeleton, to: serpent });
        assert_eq!(s.cells[serpent as usize], 0, "serpent should be dead");
        assert_eq!(s.money[1], before + 2, "rebate goes to the dead unit's owner");
    }

    #[test]
    fn unsummon_bounces_a_normal_unit_and_only_dents_a_persistent_one() {
        let mut s = after_opening();
        s.side = 0;
        let necro = NECRO_START[0];
        let victim = NEIGHBORS[necro as usize][0];

        // Non-persistent target: back to its owner's reinforcements.
        let mut a = s.clone();
        a.cells[victim as usize] = encode_cell(1, Unit::Warg);
        let a = Minions::apply(&a, &MinionsMove::Attack { from: necro, to: victim });
        assert_eq!(a.cells[victim as usize], 0);
        assert_eq!(a.reinforcements[1][Unit::Warg.reinforcement_slot().unwrap()], 1);
        assert_eq!(a.money[1], s.money[1], "an unsummon pays no rebate");

        // Persistent target: 1 damage, stays put.
        let mut b = s.clone();
        b.cells[victim as usize] = encode_cell(1, Unit::Ghost); // persistent, defense 4
        let b = Minions::apply(&b, &MinionsMove::Attack { from: necro, to: victim });
        assert_eq!(b.piece_at(victim), Some((1, Unit::Ghost)));
        assert_eq!(damage_of(b.pstate[victim as usize]), 1);
    }

    #[test]
    fn killing_the_necromancer_wins_immediately() {
        let mut s = after_opening();
        s.side = 0;
        let target = NECRO_START[1];
        // Two skeletons (5 damage each) beat defense 7 in one turn.
        let a = NEIGHBORS[target as usize][0];
        let b = NEIGHBORS[target as usize][3];
        s.cells[a as usize] = encode_cell(0, Unit::Skeleton);
        s.cells[b as usize] = encode_cell(0, Unit::Skeleton);

        let s1 = Minions::apply(&s, &MinionsMove::Attack { from: a, to: target });
        assert_eq!(Minions::outcome(&s1), Outcome::InProgress);
        assert_eq!(damage_of(s1.pstate[target as usize]), 5);
        let s2 = Minions::apply(&s1, &MinionsMove::Attack { from: b, to: target });
        assert_eq!(Minions::outcome(&s2), Outcome::FirstWins);
        assert!(Minions::legal_moves(&s2).is_empty());
    }

    #[test]
    fn income_is_graveyards_plus_three() {
        let mut s = after_opening();
        s.side = 1;
        s.phase = PHASE_SPAWN;
        let before = s.money[1];
        let held = s.graveyards_held(1);
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        assert_eq!(s.money[1], before + held as u16 + BASE_INCOME);
    }

    #[test]
    fn holding_eight_graveyards_takes_the_board() {
        let mut s = after_opening();
        s.side = 0; // first player is about to end its turn
        s.phase = PHASE_SPAWN;
        for (n, &(x, y)) in map::GRAVEYARD_XY.iter().enumerate() {
            if n < GRAVEYARDS_TO_WIN as usize {
                s.cells[idx(x, y) as usize] = encode_cell(1, Unit::Zombie);
            }
        }
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        assert_eq!(Minions::outcome(&s), Outcome::SecondWins);
    }

    #[test]
    fn buying_and_spawning_moves_money_to_the_board() {
        let mut s = after_opening();
        s.money[1] = 10;
        let s = play(&s, &[MinionsMove::EndPhase, MinionsMove::Buy(Unit::Skeleton)]);
        assert_eq!(s.money[1], 6);
        assert_eq!(s.reinforcements[1][Unit::Skeleton.reinforcement_slot().unwrap()], 1);

        let at = *s.spawn_cells(1).first().expect("second player has spawn cells");
        let s = play(&s, &[MinionsMove::Spawn { unit: Unit::Skeleton, at }]);
        assert_eq!(s.piece_at(at), Some((1, Unit::Skeleton)));
        assert_eq!(s.pstate[at as usize] & EXHAUSTED, EXHAUSTED, "spawned units are exhausted");
        assert!(!s.move_targets(at).is_empty());
        assert!(!s.can_move_piece(at), "an exhausted unit cannot move this turn");
    }

    #[test]
    fn spawning_needs_a_spawner_and_an_empty_neighbour() {
        let mut s = after_opening();
        s.phase = PHASE_SPAWN;
        s.reinforcements[1][Unit::Warg.reinforcement_slot().unwrap()] = 1;
        let cells = s.spawn_cells(1);
        assert!(!cells.is_empty());
        for &c in &cells {
            assert_eq!(s.cells[c as usize], 0);
            let touches_spawner = NEIGHBORS[c as usize].iter().any(|&n| {
                n != NO_CELL
                    && s.piece_at(n).is_some_and(|(sd, u)| sd == 1 && u.stats().spawn)
            });
            assert!(touches_spawner);
        }
        // Strip the second player's spawners and the spawn zone disappears.
        for i in 0..NUM_CELLS {
            if s.cells[i] != 0 && cell_side(s.cells[i]) == 1 {
                s.cells[i] = 0;
            }
        }
        assert!(s.spawn_cells(1).is_empty());
        assert_eq!(
            Minions::legal_moves(&s).iter().filter(|m| matches!(m, MinionsMove::Spawn { .. })).count(),
            0
        );
    }

    #[test]
    fn stalling_to_the_cap_is_adjudicated_on_gold() {
        let mut s = Minions::initial_state();
        while Minions::outcome(&s) == Outcome::InProgress {
            s = Minions::apply(&s, &MinionsMove::EndPhase);
        }
        assert_eq!(s.turn, MAX_TURNS);
        // Both sides pass, so neither takes a graveyard and both bank BASE_INCOME
        // a turn — the second player wins on its START_MONEY_SECOND head start.
        // This is the hoarding equilibrium `adjudicate` documents.
        assert!(s.money[1] > s.money[0]);
        assert_eq!(Minions::outcome(&s), Outcome::SecondWins);
    }

    #[test]
    fn adjudication_follows_gold_and_draws_exact_ties() {
        let mut s = after_opening();
        s.money = [11, 4];
        assert_eq!(adjudicate(&s), 1);
        s.money = [4, 11];
        assert_eq!(adjudicate(&s), 2);
        s.money = [7, 7];
        assert_eq!(adjudicate(&s), 3);
    }

    #[test]
    fn a_win_on_the_board_outranks_adjudication() {
        // The cap only decides games nobody has already won.
        let mut s = after_opening();
        s.turn = MAX_TURNS - 1;
        s.side = 0;
        s.phase = PHASE_SPAWN;
        s.money = [0, 99]; // second player would win on gold
        s.winner = 1; // but first already killed the necromancer
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        assert_eq!(Minions::outcome(&s), Outcome::FirstWins);
    }

    #[test]
    fn state_round_trips() {
        let mut s = after_opening();
        s.money = [13, 41];
        s.reinforcements[0][2] = 3;
        s.pstate[NECRO_START[0] as usize] = MOVED | with_damage(0, 4);
        let bytes = Minions::encode_state(&s);
        assert_eq!(bytes.len(), STATE_BYTES);
        assert_eq!(Minions::decode_state(&bytes).unwrap(), s);
    }

    #[test]
    fn moves_round_trip() {
        let mvs = [
            MinionsMove::Move { from: 4, to: 14 },
            MinionsMove::Attack { from: 22, to: 23 },
            MinionsMove::Buy(Unit::Ghost),
            MinionsMove::Spawn { unit: Unit::Zombie, at: 77 },
            MinionsMove::EndPhase,
        ];
        for mv in mvs {
            assert_eq!(Minions::decode_move(&Minions::encode_move(&mv)).unwrap(), mv);
        }
    }

    #[test]
    fn features_have_the_declared_shape() {
        let s = Minions::initial_state();
        let f = Minions::to_features(&s);
        assert_eq!(f.len(), PLANES * NUM_CELLS);
        // Plane 0 is the side-to-move's necromancer: exactly one cell set.
        assert_eq!(f[..NUM_CELLS].iter().filter(|&&v| v == 1).count(), 1);
        assert_eq!(f[NECRO_START[0] as usize], 1);
        // After the handover the canonical frame flips.
        let s = Minions::apply(&s, &MinionsMove::EndPhase);
        let f = Minions::to_features(&s);
        assert_eq!(f[NECRO_START[1] as usize], 1);
    }

    #[test]
    fn a_greedy_self_play_game_terminates() {
        let mut s = Minions::initial_state();
        let mut plies = 0;
        while Minions::outcome(&s) == Outcome::InProgress {
            let mv = Minions::choose_move(&s).best_move;
            assert!(Minions::legal_moves(&s).contains(&mv), "solver played {mv} illegally");
            s = Minions::apply(&s, &mv);
            plies += 1;
            assert!(plies < 200_000, "greedy self-play failed to terminate");
        }
        assert!(Minions::outcome(&s).is_terminal());
    }
}
