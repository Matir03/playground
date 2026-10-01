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
//! Unit stats and keywords (except the warg's price: cost 3, rebate 1, where
//! the full game has 4 and 2); the attack/spawn phase split; damage that
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

/// Money the second player starts with, compensating for moving second.
///
/// The full game's `START_MONEY_PER_BOARD` is 6, which this matched until
/// 2026-09-02; it is now 3. This is the *base* compensation only — self-play and
/// eval price the handicap themselves, self-play by drawing from
/// `BID_MIN..=BID_MAX` and eval by bidding, so this constant sets the opening
/// balance for a plain [`MinionsState::default`]: the interactive seat the UI
/// serves, and the fallback when neither side can bid.
pub const START_MONEY_SECOND: u16 = 3;

/// Range of starting-money handicaps drawn during self-play, as *blue's*
/// (second player's) advantage in gold. Negative gives the money to yellow.
///
/// A fixed handicap makes the seat, not the play, decide a game at the turn cap:
/// with `START_MONEY_SECOND` alone the second player finished ahead by +6..+12
/// net worth in every measured game, so a match that alternates seats scores
/// exactly 0.5 whatever the players' strengths and nothing is ever promoted.
///
/// Randomising it does two jobs. It breaks that determinism in the training
/// distribution, and it forces the value head to *price* the handicap — which is
/// what the bidding phase then binary-searches over to find the compensation
/// that makes a game even.
pub const BID_MIN: i16 = -5;
pub const BID_MAX: i16 = 10;
/// Flat part of end-of-turn income: `+2` per board, `+1` soul for the basic
/// necromancer.
pub const BASE_INCOME: u16 = 3;
/// Turns (counting each side's separately, so this is 20 apiece) after which the
/// game is decided by [`adjudicate`] instead of being played out.
/// 20 side-turns — ten apiece — on the 7x7 board.
pub const MAX_TURNS: u16 = 30;

/// Turns averaged when adjudicating at the cap.
///
/// Adjudicating on the single final position hands the result to whoever the
/// clock happens to stop on: a side that has just banked income and not yet been
/// answered looks ahead by a full turn's worth of gold. Averaging the net-worth
/// difference over the last six turns — three apiece — cancels that, and also
/// smooths a one-off trade that happens to land on the boundary.
pub const ADJ_WINDOW: usize = 6;

/// Feature planes produced by [`Game::to_features`].
pub const PLANES: usize = 26;

/// A side's **net worth**: gold, plus the cost of every unit it owns, whether on
/// the board or waiting in reinforcements.
///
/// Units are valued at [`UnitStats::cost`], not `rebate`, and that choice is the
/// whole point: paying `cost` gold for a unit worth `cost` leaves net worth
/// exactly unchanged, so buying and spawning is a **no-op** under this measure.
/// Valuing at `rebate` (which is strictly lower — a skeleton costs 4 and rebates
/// 2) would still tax every purchase and leave a weaker version of the same
/// hoarding incentive.
///
/// Reinforcements count at full value because an unsummoned unit is not lost:
/// it returns to its owner's pool and can be respawned for free. Unsummoning
/// therefore costs its victim tempo but not material, while a **kill** removes
/// the unit outright. That asymmetry is real and this scoring preserves it.
pub fn net_worth(state: &MinionsState, side: u8) -> u32 {
    let mut total = state.money[side as usize] as u32;
    for &c in state.cells.iter() {
        if c != 0 && cell_side(c) == side {
            total += cell_unit(c).stats().cost as u32;
        }
    }
    for (slot, &n) in state.reinforcements[side as usize].iter().enumerate() {
        total += n as u32 * BUYABLE[slot].stats().cost as u32;
    }
    total
}

/// Who wins a game that hits [`MAX_TURNS`]: **higher net worth wins**, exact ties
/// draw.
///
/// Deliberately the single place this rule lives, because it is load-bearing.
/// Most games hit the cap (the greedy heuristic finishes around turn 33), so this
/// is the de-facto objective rather than a tiebreak.
///
/// It scored **raw gold** until 2026-08-26, and self-play found the hole exactly
/// as predicted: gold is *spent* on units, so hoarding beat developing. Three
/// generations of the watchdog, monotonic throughout — mean gold 7.7 -> 18.5 ->
/// 25.1, units on board 16.9 -> 14.1 -> 12.4, and corr(gold, units) +0.47 ->
/// -0.11 -> -0.23. The degenerate line is concrete: with both sides passing,
/// income is [`BASE_INCOME`] a turn each, so after 20 turns apiece the first
/// player holds 60 and the second 63 from [`START_MONEY_SECOND`], and "pass
/// forever" simply wins for the second player.
///
/// Scoring [`net_worth`] removes the incentive at its root rather than penalising
/// the symptom: converting gold into units is value-neutral, so the only way to
/// gain is to take material the opponent does not take back.
/// Terminal "how well was this won" score, from `side`'s seat.
///
/// Two terms, both in the same units so they add without a weight:
///
/// * **Speed.** `10 * (MAX_TURNS - final_turn)`, signed by who won: 0 for a game
///   that goes the distance, 200 for a win on turn 10, 290 for a win on turn 1.
///   Games cannot exceed [`MAX_TURNS`], so this never needs clamping.
///
///   Anchored to the cap rather than a fixed number, so it keeps meaning
///   "points per turn saved against the worst case" if the horizon moves
///   again. It did: the cap was 20 through generation 279, and scores recorded
///   before that are on the old scale and not comparable.
/// * **Material.** The net-worth differential, the same quantity the eval
///   harness already reports as `A_margin`.
///
/// Signed and antisymmetric: `score(s, a) == -score(s, 1 - a)`. A draw scores 0
/// on both terms, since the cap only draws on an exact net-worth tie.
///
/// Note the two terms can disagree, deliberately. A necromancer kill on turn 12
/// while behind on material still scores well, because finishing early is what
/// the speed term is there to reward.
pub fn game_score(state: &MinionsState, side: u8) -> f32 {
    // `winner` is 1/2 for a decided game (side 0 / side 1), 3 a draw, 0 still
    // running. An unfinished position scores its material only, which is what
    // the aux head sees for every non-terminal example.
    let sign = match state.winner {
        1 | 2 => {
            if state.winner - 1 == side { 1.0f32 } else { -1.0f32 }
        }
        _ => 0.0f32,
    };
    let speed = 10.0 * (MAX_TURNS.saturating_sub(state.turn)) as f32;
    let material =
        net_worth(state, side) as f32 - net_worth(state, 1 - side) as f32;
    sign * speed + material
}

pub fn adjudicate(state: &MinionsState) -> u8 {
    // Mean over the window rather than the final position: see [`ADJ_WINDOW`].
    // Summed as i32 and compared against zero, so no division and no rounding
    // decides a close game.
    let total: i32 = state.nw_hist.iter().map(|&v| v as i32).sum();
    match total.cmp(&0) {
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

/// Bit `i` set where `lo <= cells[i] <= hi`. Three SSE2 compares on x86_64
/// (the exact search's move generation calls this per node), a scalar loop
/// elsewhere.
#[inline]
fn byte_mask(cells: &[u8; NUM_CELLS], lo: u8, hi: u8) -> u64 {
    const _: () = assert!(NUM_CELLS <= 64);
    let mut m = 0u64;
    #[cfg(target_arch = "x86_64")]
    let full = {
        use std::arch::x86_64::*;
        let full = NUM_CELLS / 16 * 16;
        // SAFETY: SSE2 is part of the x86_64 baseline, and every load reads
        // 16 bytes inside `cells` (`k + 16 <= full <= NUM_CELLS`).
        unsafe {
            let (vlo, span) = (_mm_set1_epi8(lo as i8), _mm_set1_epi8((hi - lo) as i8));
            let mut k = 0;
            while k < full {
                let v = _mm_sub_epi8(_mm_loadu_si128(cells.as_ptr().add(k) as *const __m128i), vlo);
                // Unsigned v <= span: min(v, span) == v.
                let inside = _mm_cmpeq_epi8(_mm_min_epu8(v, span), v);
                m |= u64::from(_mm_movemask_epi8(inside) as u16) << k;
                k += 16;
            }
        }
        full
    };
    #[cfg(not(target_arch = "x86_64"))]
    let full = 0;
    for (i, &c) in cells.iter().enumerate().skip(full) {
        m |= u64::from(c.wrapping_sub(lo) <= hi - lo) << i;
    }
    m
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
    /// Net-worth difference (first minus second) at the end of each of the last
    /// [`ADJ_WINDOW`] turns, oldest first. Kept in the state because
    /// [`adjudicate`] needs history and `Game::outcome` is stateless.
    pub nw_hist: [i16; ADJ_WINDOW],
    /// Monotonic count of *actions*, not turns — this is `Game::ply`.
    pub ply: u32,
    /// 0 = in progress, 1 = first wins, 2 = second wins, 3 = draw.
    pub winner: u8,
}

impl MinionsState {
    /// The position seen from the mover's side: rotated 180° and colour-swapped,
    /// so the side to move is always side 0 starting from the bottom-left.
    ///
    /// The two seats play the *same game* under this involution (see
    /// [`crate::map::rot`]), but with absolute cell indices the net had to learn
    /// both orientations separately — halving the value of every example and
    /// handing the value head a seat signal to shortcut on. Canonicalising here
    /// means one point of view for both seats.
    ///
    /// Cheap: the rotation is index reversal, so this is a reverse plus a colour
    /// flip. Self-inverse, which `canonical_round_trips` checks.
    /// [`Game::encode_state`]'s bytes, into a buffer padded to whole words
    /// (the padding stays zero).
    pub fn encode_into(&self, out: &mut [u8; STATE_WORDS * 8]) {
        let mut at = 0;
        let mut put = |b: &[u8]| {
            out[at..at + b.len()].copy_from_slice(b);
            at += b.len();
        };
        put(&self.cells);
        put(&self.pstate);
        for m in self.money {
            put(&m.to_le_bytes());
        }
        for side in self.reinforcements {
            put(&side);
        }
        // Adjudication history travels with the state: priming it from the
        // position instead would fabricate a past the game did not have, and a
        // state near the cap would decode to the wrong result.
        for v in self.nw_hist {
            put(&v.to_le_bytes());
        }
        put(&[self.side, self.phase]);
        put(&self.turn.to_le_bytes());
        put(&self.ply.to_le_bytes());
        put(&[self.winner]);
        debug_assert_eq!(at, STATE_BYTES);
    }

    /// A 128-bit hash of the encoded state: two independent multiply-mix lanes
    /// over its words, each finished with murmur3's `fmix64`. Distinct turns
    /// routinely converge on one position, and the search merges them by this
    /// key, so it must be collision-free in practice; 128 bits keeps
    /// accidental merges far below the noise floor. No allocation, and 17
    /// word steps where a byte-wise FNV took 133 dependent 128-bit multiplies.
    pub fn key(&self) -> u128 {
        let mut buf = [0u8; STATE_WORDS * 8];
        self.encode_into(&mut buf);
        let fmix = |mut k: u64| {
            k ^= k >> 33;
            k = k.wrapping_mul(0xff51afd7ed558ccd);
            k ^= k >> 33;
            k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
            k ^ (k >> 33)
        };
        let (mut a, mut b) = (0x9e3779b97f4a7c15u64, 0x6a09e667f3bcc909u64);
        for w in buf.chunks_exact(8) {
            let w = u64::from_le_bytes(w.try_into().unwrap());
            a = (a ^ w).wrapping_mul(0x87c37b91114253d5).rotate_left(31);
            b = (b ^ w.rotate_left(29)).wrapping_mul(0x4cf5ad432745937f).rotate_left(27);
        }
        let (a, b) = (fmix(a ^ b.rotate_left(17)), fmix(b ^ STATE_BYTES as u64));
        ((a as u128) << 64) | b as u128
    }

    pub fn canonical(&self) -> Self {
        if self.side == 0 { self.clone() } else { self.rotated() }
    }

    /// The position rotated 180° with the two seats swapped.
    ///
    /// Unconditional, and a true involution — `rotated().rotated()` is the
    /// identity. [`Self::canonical`] applies it only when it is side 1 to move,
    /// which makes *that* idempotent rather than self-inverse.
    pub fn rotated(&self) -> Self {
        let mut out = self.clone();
        for i in 0..NUM_CELLS {
            let j = crate::map::rot(i);
            let c = self.cells[i];
            // Flip the owner so the mover's pieces are always side 0.
            out.cells[j] = if c == 0 { 0 } else { encode_cell(1 - cell_side(c), cell_unit(c)) };
            out.pstate[j] = self.pstate[i];
        }
        out.money.swap(0, 1);
        out.reinforcements.swap(0, 1);
        out.side = 1 - self.side;
        // `nw_hist` is first-minus-second, so swapping the seats negates it.
        for v in out.nw_hist.iter_mut() {
            *v = -*v;
        }
        out.winner = match self.winner {
            1 => 2,
            2 => 1,
            w => w,
        };
        out
    }

    /// Opening position with a starting-money handicap, as *blue's* (second
    /// player's) advantage in gold. Negative gives the money to yellow instead.
    ///
    /// This is the quantity the bidding phase settles: both sides name the
    /// compensation they think makes the game even, and the average becomes the
    /// real handicap. Self-play draws it from `BID_MIN..=BID_MAX` so the value
    /// head learns to price it rather than seeing one fixed number.
    pub fn with_bid(bid: i16) -> Self {
        let mut s = Self::default();
        s.money = if bid >= 0 {
            [0, bid as u16]
        } else {
            [(-bid) as u16, 0]
        };
        // The window is primed from the position, so it has to be re-primed
        // after the balance changes or adjudication would use the default one.
        s.prime_nw_hist();
        s
    }
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
        let mut me = Self {
            cells,
            pstate: [0; NUM_CELLS],
            money: [0, START_MONEY_SECOND],
            reinforcements: [[0; BUYABLE.len()]; 2],
            nw_hist: [0; ADJ_WINDOW],
            side: 0,
            phase: PHASE_ATTACK,
            turn: 0,
            ply: 0,
            winner: 0,
        };
        me.prime_nw_hist();
        me
    }
}

impl MinionsState {
    /// Fill the adjudication window with the current balance.
    ///
    /// Used for the opening position, where there is no history yet. Without it
    /// a fresh state would adjudicate as a draw regardless of the balance. States
    /// rebuilt from the wire format carry their real history instead.
    pub fn prime_nw_hist(&mut self) {
        let diff = net_worth(self, 0) as i32 - net_worth(self, 1) as i32;
        let v = diff.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        self.nw_hist = [v; ADJ_WINDOW];
    }

    /// The cells holding a `side` unit: [`Self::piece_at`]'s side test on the
    /// cell bytes, without decoding the unit.
    #[inline]
    pub fn side_bb(&self, side: u8) -> crate::bitboard::Bb {
        let lo = 1 + NUM_UNITS as u8 * side;
        crate::bitboard::Bb::from(byte_mask(&self.cells, lo, lo + NUM_UNITS as u8 - 1))
    }

    /// The empty cells.
    #[inline]
    pub fn empty_bb(&self) -> crate::bitboard::Bb {
        crate::bitboard::Bb::from(byte_mask(&self.cells, 0, 0))
    }

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
    ///
    /// A unit spawned **this turn** does not count: spawn chaining is explicitly
    /// illegal, so a freshly placed zombie cannot open further cells until next
    /// turn. Spawned units carry `EXHAUSTED`, which is exactly that predicate.
    pub fn spawn_cells(&self, side: u8) -> Vec<u8> {
        crate::bitboard::cells(self.spawn_bb(side)).collect()
    }

    /// [`Self::spawn_cells`] as a bitboard: the empty neighbours of `side`'s
    /// spawners that have not themselves spawned this turn.
    pub fn spawn_bb(&self, side: u8) -> crate::bitboard::Bb {
        use crate::bitboard::{bit, dilate};
        let (mut from, mut empty) = (0, 0);
        for (i, &c) in self.cells.iter().enumerate() {
            if c == 0 {
                empty |= bit(i as u8);
            } else if cell_side(c) == side
                && cell_unit(c).stats().spawn
                && self.pstate[i] & EXHAUSTED == 0
            {
                from |= bit(i as u8);
            }
        }
        dilate(from) & empty
    }

    /// Cells a piece at `from` may move to: empty, and reachable within its
    /// speed through cells that are empty or friendly (or anything, if flying).
    pub fn move_targets(&self, from: u8) -> Vec<u8> {
        crate::bitboard::cells(self.move_target_bb(from)).collect()
    }

    /// [`Self::move_targets`] as a bitboard (no allocation; read it lowest
    /// cell first for the same order).
    pub fn move_target_bb(&self, from: u8) -> crate::bitboard::Bb {
        let Some((side, unit)) = self.piece_at(from) else {
            return 0;
        };
        let stats = unit.stats();
        let flying = stats.flying;

        // Stack buffers, not `Vec`s. This is the hot path of the forced-
        // exploration playout: `random_playout` calls `legal_moves` once per
        // action and 85% of that is here, so the old version's five allocations
        // per call (`frontier`, `out`, and a fresh `next` on every BFS step)
        // came to roughly 600 allocations per playout. A frontier can never
        // exceed the board.
        let mut steps = [u8::MAX; NUM_CELLS];
        steps[from as usize] = 0;
        let mut frontier = [0u8; NUM_CELLS];
        let mut next = [0u8; NUM_CELLS];
        frontier[0] = from;
        let mut n_front = 1usize;
        let mut out: crate::bitboard::Bb = 0;

        for step in 1..=stats.speed {
            let mut n_next = 0usize;
            for &cur in &frontier[..n_front] {
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
                        out |= crate::bitboard::bit(n);
                    }
                    next[n_next] = n;
                    n_next += 1;
                }
            }
            frontier[..n_next].copy_from_slice(&next[..n_next]);
            n_front = n_next;
            if n_front == 0 {
                break;
            }
        }
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
        if range == 1 {
            // Every unit today: the enemies among the six neighbours, lowest
            // cell first like the scan below (the exact search's move
            // generation spent ~20% of its time in that scan).
            let near = self.side_bb(1 - side) & crate::bitboard::dilate(crate::bitboard::bit(from));
            let mut out = Vec::with_capacity(near.count_ones() as usize);
            out.extend(crate::bitboard::cells(near));
            return out;
        }
        // `collect` from a filter has no useful size hint, so it reallocated as
        // it grew; a unit rarely has more than a handful of targets in range.
        let mut out = Vec::with_capacity(8);
        for i in 0..NUM_CELLS as u8 {
            let c = self.cells[i as usize];
            if c != 0 && cell_side(c) != side && dist(from, i) <= range {
                out.push(i);
            }
        }
        out
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

impl MinionsMove {
    /// The same action on the board rotated 180° ([`MinionsState::rotated`]).
    pub fn rotated(&self) -> Self {
        let r = |c: u8| crate::map::rot(c as usize) as u8;
        match *self {
            MinionsMove::Move { from, to } => MinionsMove::Move { from: r(from), to: r(to) },
            MinionsMove::Attack { from, to } => MinionsMove::Attack { from: r(from), to: r(to) },
            MinionsMove::Spawn { unit, at } => MinionsMove::Spawn { unit, at: r(at) },
            MinionsMove::Buy(u) => MinionsMove::Buy(u),
            MinionsMove::EndPhase => MinionsMove::EndPhase,
        }
    }
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

        // Record this turn's balance before the cap is tested, so the window
        // includes the turn that just finished.
        let diff = net_worth(state, 0) as i32 - net_worth(state, 1) as i32;
        state.nw_hist.rotate_left(1);
        state.nw_hist[ADJ_WINDOW - 1] = diff.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

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
        // Sized to a typical mid-game branching factor (~30-50 moves) so the
        // per-action enumeration in `random_playout` stops reallocating.
        let mut out = Vec::with_capacity(64);

        if state.phase == PHASE_ATTACK {
            // Bitboards read lowest cell first: the same order as scanning the
            // cells, without two vectors per unit (this runs at every node of
            // the exact witness search).
            let enemies = state.side_bb(1 - side);
            for i in crate::bitboard::cells(state.side_bb(side)) {
                if state.can_move_piece(i) {
                    for to in crate::bitboard::cells(state.move_target_bb(i)) {
                        out.push(MinionsMove::Move { from: i, to });
                    }
                }
                let (_, unit) = state.piece_at(i).expect("own unit");
                if unit.stats().range == 1 {
                    if state.can_attack_piece(i, unit) {
                        let near = enemies & crate::bitboard::dilate(crate::bitboard::bit(i));
                        for to in crate::bitboard::cells(near) {
                            out.push(MinionsMove::Attack { from: i, to });
                        }
                    }
                } else {
                    for to in state.attack_targets(i) {
                        out.push(MinionsMove::Attack { from: i, to });
                    }
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
        // No per-piece turn-state planes (damage / moved / attacked / exhausted)
        // and no phase plane. The action space is a whole turn, so the only
        // positions the net ever sees are turn boundaries — and `end_turn`
        // clears every `pstate`, leaving all five constant. Asserted over real
        // games by `turn_boundary_states_have_no_piece_state`.
        //
        // No `side` plane either: the sampler canonicalises to the mover's frame
        // (`MinionsState::canonical`), so it would be a constant 0 — and while
        // the frame was absolute it was a direct seat signal for the value head
        // to shortcut on instead of reading the position.
        //
        // The graveyard mask stays. It is constant across *examples* but varies
        // across *cells*, and the trunk is fully convolutional — this plane is
        // the only thing telling it which six cells decide the game.
        plane(&mut |i| IS_GRAVEYARD[i] as u8, &mut out);
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
        // Turns remaining before the cap forces adjudication. Raw count, like
        // money and reinforcements — the net sees 0..=MAX_TURNS, not a fraction.
        //
        // Appended **last** on purpose: every earlier plane index keeps its
        // meaning, so the watchdog's hardcoded offsets and any older reasoning
        // about the layout stay correct.
        let left = MAX_TURNS.saturating_sub(state.turn).min(255) as u8;
        plane(&mut |_| left, &mut out);
        debug_assert_eq!(out.len(), PLANES * NUM_CELLS);
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut buf = [0u8; STATE_WORDS * 8];
        state.encode_into(&mut buf);
        buf[..STATE_BYTES].to_vec()
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
            nw_hist: [0; ADJ_WINDOW],
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
        for v in s.nw_hist.iter_mut() {
            *v = i16::from_le_bytes([bytes[o], bytes[o + 1]]);
            o += 2;
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
    + 2 * ADJ_WINDOW                       // nw_hist
    + 1 + 1 + 2 + 4 + 1;

/// [`STATE_BYTES`] in whole 8-byte words, for [`MinionsState::key`].
pub const STATE_WORDS: usize = STATE_BYTES.div_ceil(8); // side, phase, turn, ply, winner

impl Solver for Minions {
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move> {
        crate::solver::greedy_move(state)
    }

    /// The trained net, playing a whole turn per search — see
    /// [`crate::net_seat`]. Returns `None` unless `serve --net-dir` configured
    /// one, so asking to play the net without weights is a clear failure rather
    /// than a silent fallback to the greedy heuristic.
    fn choose_move_external(
        state: &Self::State,
        budget: playground_core::SolverBudget,
    ) -> Option<anyhow::Result<SearchOutcome<Self::Move>>> {
        crate::net_seat::choose(state, budget)
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
        // Up to the cap: past it the game is over and having no moves is correct.
        while !Minions::outcome(&s).is_terminal() {
            let moves = Minions::legal_moves(&s);
            assert!(!moves.is_empty(), "no legal move at turn {}", s.turn);
            s = Minions::apply(&s, &MinionsMove::EndPhase);
        }
        assert_eq!(s.turn, MAX_TURNS, "passing should run to the cap");
    }

    #[test]
    fn zombies_are_lumbering() {
        let mut s = Minions::initial_state();
        // Put an enemy zombie next to one of ours, then move ours and try to attack.
        let from = idx(0, 4);
        let target = idx(0, 6);
        s.cells[from as usize] = encode_cell(0, Unit::Zombie);
        s.cells[target as usize] = encode_cell(1, Unit::Zombie);
        let dest = idx(0, 5);
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
        // From d0 the only two-step route to d2 runs through d1, so whatever
        // sits on d1 decides whether d2 is reachable.
        let from = idx(3, 0);
        let corridor = idx(3, 1);
        let behind = idx(3, 2);
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
    /// Buying and spawning must not move net worth. This is the property the
    /// whole rule rests on: gold converts into units at par, so the only way to
    /// gain is to take material the opponent does not take back. Valuing units
    /// at `rebate` instead of `cost` would break it (a skeleton costs 4, rebates
    /// 2) and leave a weaker form of the hoarding incentive.
    #[test]
    fn buying_and_spawning_leave_net_worth_unchanged() {
        // Accounting test: it drives `apply` directly rather than going through
        // legal-move generation, because the point is the bookkeeping, not when
        // the moves happen to be legal.
        let mut s = Minions::initial_state();
        s.money[0] = 20; // the opening bank is under a skeleton's price
        let before = net_worth(&s, 0);

        let bought = Minions::apply(&s, &MinionsMove::Buy(Unit::Skeleton));
        assert_eq!(bought.money[0], s.money[0] - Unit::Skeleton.stats().cost,
                   "buying must actually cost gold");
        assert_eq!(net_worth(&bought, 0), before, "gold -> reinforcements is not par");

        // ...and the reinforcement converting to a unit on the board is also par.
        let at = (0..NUM_CELLS).find(|&i| bought.cells[i] == 0).expect("an empty cell") as u8;
        let spawned = Minions::apply(&bought, &MinionsMove::Spawn { unit: Unit::Skeleton, at });
        assert_eq!(net_worth(&spawned, 0), before, "reinforcements -> board is not par");
    }

    /// The behaviour that changed: material beats a bank balance.
    #[test]
    fn adjudication_counts_units_not_just_gold() {
        let mut s = after_opening();
        // Second player is richer in raw gold but has spent nothing; first player
        // converted gold into a skeleton on the board. Under the old gold rule the
        // hoarder won; under net worth the position is what counts.
        // Both sides open symmetrically, so their starting material cancels and
        // only the gold and the added unit move the comparison.
        s.money = [0, 4];
        assert_eq!(net_worth(&s, 0) + 4, net_worth(&s, 1), "setup: second is 4 ahead");
        let cell = (0..NUM_CELLS).find(|&i| s.cells[i] == 0).expect("an empty cell");
        s.cells[cell] = encode_cell(0, Unit::Skeleton);
        s.prime_nw_hist();
        assert!(s.money[1] > s.money[0], "second player must be ahead on raw gold");
        assert_eq!(adjudicate(&s), 3, "a skeleton exactly offsets 4 gold");

        // One more zombie of material and the first player is simply ahead.
        let cell2 = (0..NUM_CELLS).find(|&i| s.cells[i] == 0).expect("another empty cell");
        s.cells[cell2] = encode_cell(0, Unit::Zombie);
        s.prime_nw_hist();
        assert_eq!(adjudicate(&s), 1);
    }

    /// Unsummoning costs its victim tempo, not material: the unit returns to the
    /// owner's pool and can be respawned for free, so net worth is unchanged. A
    /// kill removes it outright. The scoring must keep those distinct.
    #[test]
    fn unsummon_preserves_net_worth_but_a_kill_does_not() {
        let mut s = after_opening();
        let cell = (0..NUM_CELLS).find(|&i| s.cells[i] == 0).expect("an empty cell");
        s.cells[cell] = encode_cell(0, Unit::Skeleton);
        let with_unit = net_worth(&s, 0);

        // Unsummon: board -> reinforcements.
        let mut unsummoned = s.clone();
        unsummoned.cells[cell] = 0;
        let slot = BUYABLE.iter().position(|u| *u == Unit::Skeleton).unwrap();
        unsummoned.reinforcements[0][slot] += 1;
        assert_eq!(net_worth(&unsummoned, 0), with_unit, "unsummon must not cost material");

        // Killed: gone entirely.
        let mut killed = s.clone();
        killed.cells[cell] = 0;
        assert_eq!(
            net_worth(&killed, 0),
            with_unit - Unit::Skeleton.stats().cost as u32,
            "a kill must cost the unit's full value"
        );
    }

    #[test]
    fn stalling_to_the_cap_is_adjudicated_on_net_worth() {
        let mut s = Minions::initial_state();
        while Minions::outcome(&s) == Outcome::InProgress {
            s = Minions::apply(&s, &MinionsMove::EndPhase);
        }
        assert_eq!(s.turn, MAX_TURNS);
        // Both sides pass, so neither takes a graveyard, neither spends, and both
        // bank BASE_INCOME a turn — the second player wins on its
        // START_MONEY_SECOND head start. Net worth reduces to gold here precisely
        // because nothing was bought, which is what makes "pass forever" still a
        // *win* for the second player rather than a draw: net worth removes the
        // incentive to hoard over developing, it does not erase the head start.
        assert!(s.money[1] > s.money[0]);
        assert_eq!(Minions::outcome(&s), Outcome::SecondWins);
    }

    #[test]
    fn adjudication_follows_gold_and_draws_exact_ties() {
        let mut s = after_opening();
        // Adjudication averages the window, so each position has to be given a
        // history — `prime_nw_hist` states "this balance has held".
        s.money = [11, 4];
        s.prime_nw_hist();
        assert_eq!(adjudicate(&s), 1);
        s.money = [4, 11];
        s.prime_nw_hist();
        assert_eq!(adjudicate(&s), 2);
        s.money = [7, 7];
        s.prime_nw_hist();
        assert_eq!(adjudicate(&s), 3);
    }

    /// The window is the point: a lead that only exists on the final turn — the
    /// side to move having just banked income the opponent has not answered —
    /// must not decide the game.
    #[test]
    /// A bid is a signed handicap: positive gives blue (second) the money,
    /// negative gives it to yellow (first), and it must reach adjudication.
    #[test]
    fn a_bid_hands_the_starting_money_to_the_named_side() {
        let even = MinionsState::with_bid(0);
        assert_eq!(even.money, [0, 0]);
        assert_eq!(adjudicate(&even), 3, "no handicap, identical armies: a draw");

        let blue = MinionsState::with_bid(7);
        assert_eq!(blue.money, [0, 7]);
        assert_eq!(adjudicate(&blue), 2, "blue's handicap must win the tiebreak");

        let yellow = MinionsState::with_bid(-4);
        assert_eq!(yellow.money, [4, 0]);
        assert_eq!(adjudicate(&yellow), 1, "a negative bid pays yellow instead");

        // The adjudication window is primed from the position, so a handicap
        // applied after construction has to be reflected in it — otherwise the
        // average is taken over a balance the game never had.
        assert!(blue.nw_hist.iter().all(|&v| v < 0), "window must see blue ahead");
        assert!(yellow.nw_hist.iter().all(|&v| v > 0), "window must see yellow ahead");
    }

    fn adjudication_averages_away_a_last_turn_swing() {
        let mut s = after_opening();
        s.money = [0, 0];
        s.prime_nw_hist();
        // Level throughout, then first player banks a turn's income at the buzzer.
        s.nw_hist[ADJ_WINDOW - 1] = 4;
        assert_eq!(adjudicate(&s), 1, "a +4 blip against five level turns still wins?");

        // Against a sustained deficit, the same blip must not flip the result.
        for v in s.nw_hist.iter_mut().take(ADJ_WINDOW - 1) {
            *v = -2;
        }
        assert_eq!(adjudicate(&s), 2, "sustained -2 x5 must outweigh a +4 final turn");
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

#[cfg(test)]
mod move_targets_bench {
    use super::*;
    use rand::SeedableRng;

    /// `move_targets` exactly as it was before the stack-buffer rewrite, so the
    /// speedup is a paired A/B in one binary rather than a before/after across
    /// two builds. Delete once the change is settled.
    fn legacy_move_targets(st: &MinionsState, from: u8) -> Vec<u8> {
        let Some((side, unit)) = st.piece_at(from) else {
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
                    let occ = st.cells[n as usize];
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

    /// `key` follows the encoding: equal states agree, a change to any one
    /// byte of the encoding changes it, and random positions never collide.
    #[test]
    fn key_separates_every_byte_and_does_not_collide() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let mut keys: std::collections::HashMap<u128, Vec<u8>> = Default::default();
        let mut st = Minions::initial_state();
        for step in 0..3000 {
            if Minions::outcome(&st).is_terminal() {
                st = Minions::initial_state();
            }
            let enc = Minions::encode_state(&st);
            assert_eq!(st.key(), st.clone().key());
            if let Some(prev) = keys.insert(st.key(), enc.clone()) {
                assert_eq!(prev, enc, "two encodings share a key");
            }
            if step % 50 == 0 {
                for i in 0..enc.len() {
                    let mut e = enc.clone();
                    e[i] ^= 1 << (step / 50 % 8);
                    if let Ok(t) = Minions::decode_state(&e) {
                        assert_ne!(t.key(), st.key(), "byte {i} did not change the key");
                    }
                }
            }
            let mvs = Minions::legal_moves(&st);
            st = Minions::apply(&st, &mvs[rng.gen_range(0..mvs.len())]);
        }
    }

    /// The pre-bitboard `spawn_cells`, walking the neighbour table.
    fn legacy_spawn_cells(s: &MinionsState, side: u8) -> Vec<u8> {
        let mut seen = [false; NUM_CELLS];
        let mut out = Vec::new();
        for (i, &c) in s.cells.iter().enumerate() {
            if c == 0 || cell_side(c) != side || !cell_unit(c).stats().spawn {
                continue;
            }
            if s.pstate[i] & EXHAUSTED != 0 {
                continue;
            }
            for &n in &NEIGHBORS[i] {
                if n != NO_CELL && s.cells[n as usize] == 0 && !seen[n as usize] {
                    seen[n as usize] = true;
                    out.push(n);
                }
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn spawn_bb_matches_the_neighbour_walk() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let mut checked = 0;
        for _ in 0..40 {
            let mut st = Minions::initial_state();
            // Single moves, so mid-phase states (spawners exhausted by a
            // spawn this turn) are covered too.
            for _ in 0..400 {
                if Minions::outcome(&st).is_terminal() {
                    break;
                }
                for side in 0..2 {
                    assert_eq!(st.spawn_cells(side), legacy_spawn_cells(&st, side));
                    checked += 1;
                }
                let mvs = Minions::legal_moves(&st);
                st = Minions::apply(&st, &mvs[rng.gen_range(0..mvs.len())]);
            }
        }
        assert!(checked > 1000, "only {checked} states checked");
    }

    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn move_targets_paired_ab() {
        let mut st = MinionsState::default();
        let mut prng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        for _ in 0..12 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = crate::turn::TurnSpec::random_playout(&st, &mut prng);
            st = crate::turn::realize(&st, &spec).next;
        }
        let own: Vec<u8> = (0..NUM_CELLS as u8)
            .filter(|&i| st.piece_at(i).map(|(s, _)| s == st.side).unwrap_or(false))
            .collect();
        // Identical output is the precondition for the timing to mean anything.
        for &i in &own {
            assert_eq!(st.move_targets(i), legacy_move_targets(&st, i), "cell {i}");
        }
        let reps = 20000;
        let run = |legacy: bool| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for &i in &own {
                    let v = if legacy { legacy_move_targets(&st, i) } else { st.move_targets(i) };
                    std::hint::black_box(v.len());
                }
            }
            t0.elapsed().as_secs_f64() * 1e9 / (reps as f64 * own.len() as f64)
        };
        run(true);
        run(false);
        let (mut la, mut nu) = (0.0, 0.0);
        for _ in 0..4 {
            la += run(true);
            nu += run(false);
            nu += run(false);
            la += run(true);
        }
        println!(
            "move_targets  legacy {:.0} ns   new {:.0} ns   {:+.1}%   ({} own units)",
            la / 8.0,
            nu / 8.0,
            (la - nu) / la * 100.0,
            own.len()
        );
    }

}

#[cfg(test)]
mod game_score_tests {
    use super::*;
    use playground_core::Game;

    fn at(turn: u16, winner: u8) -> MinionsState {
        let mut s = Minions::initial_state();
        s.turn = turn;
        s.winner = winner;
        s
    }

    #[test]
    fn speed_term_matches_the_stated_schedule() {
        // The schedule the score was specified by: 0 at the cap, 10 a turn
        // earlier, 100 at turn 10. Material is level from `initial_state`
        // except for the second player's starting money, so compare the
        // *difference* between two turns rather than the absolute value.
        let base = game_score(&at(20, 1), 0);
        assert_eq!(game_score(&at(19, 1), 0) - base, 10.0);
        assert_eq!(game_score(&at(10, 1), 0) - base, 100.0);
    }

    #[test]
    fn a_game_that_goes_the_distance_scores_material_only() {
        let s = at(MAX_TURNS, 1);
        let material = net_worth(&s, 0) as f32 - net_worth(&s, 1) as f32;
        assert_eq!(game_score(&s, 0), material, "no speed bonus at the cap");
    }

    #[test]
    fn score_is_antisymmetric_between_the_seats() {
        for (turn, winner) in [(12u16, 1u8), (17, 2), (20, 3), (14, 2)] {
            let s = at(turn, winner);
            assert_eq!(
                game_score(&s, 0),
                -game_score(&s, 1),
                "turn {turn} winner {winner}: one seat's score must be the other's negated"
            );
        }
    }

    #[test]
    fn losing_fast_is_worse_than_losing_slowly() {
        // The mirror of the speed bonus, and the reason `sign` is applied to it
        // rather than to the whole score: a net that is going to lose should
        // still prefer to lose late.
        assert!(game_score(&at(12, 2), 0) < game_score(&at(19, 2), 0));
    }

    #[test]
    fn a_draw_scores_no_speed_either_way() {
        let s = at(20, 3);
        let material = net_worth(&s, 0) as f32 - net_worth(&s, 1) as f32;
        assert_eq!(game_score(&s, 0), material);
    }

    #[test]
    fn an_early_win_can_outscore_a_material_deficit() {
        // The point of the speed term: finishing on turn 12 while behind on
        // material still reads as a better game than grinding to the cap.
        let mut fast = at(12, 1);
        // Hand side 1 a large material edge; side 0 still won.
        fast.money[1] = 40;
        let slow = at(20, 1);
        assert!(
            game_score(&fast, 0) > game_score(&slow, 0),
            "an 80-point speed bonus must outweigh a 40-point material deficit"
        );
    }
}
