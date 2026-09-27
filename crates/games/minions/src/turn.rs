//! The flat whole-turn action space, and the realizer that compiles it into
//! legal [`MinionsMove`]s.
//!
//! A turn is declared, not sequenced — three board-shaped fields:
//!
//! | field | per cell | meaning |
//! |---|---|---|
//! | [`TurnSpec::moves`] | one of 37 offsets | where the unit standing here goes (offset 0 = stay) |
//! | [`TurnSpec::targets`] | stay / kill / unsummon | what happens to the enemy standing here |
//! | [`TurnSpec::spawns`] | none, or a unit | what is created here |
//!
//! Three properties of the rules make this the *exact* action space rather than
//! a lossy summary of it:
//!
//! - Max speed is 3 (warg) and there are exactly 37 hexes within distance 3, so
//!   a 37-way relative offset covers every unit's every destination.
//! - Damage clears at end of turn (`Minions::end_turn` zeroes every `pstate`),
//!   so damage that doesn't kill *this turn* is wasted. There is nothing between
//!   "kill" and "leave alone" worth expressing.
//! - Buying without spawning is dominated — money and reinforcements both
//!   persist — so a purchase is implied by its spawn and never declared.
//!
//! # Realization
//!
//! Moves and attacks interleave: an enemy may have to die before a friendly can
//! path through or occupy its hex, and that unblocked move may in turn put
//! another unit in range for the next kill. So [`realize`] runs a fixed point,
//! not a pipeline of passes.
//!
//! It always terminates: each unit moves at most once and attacks at most once
//! (every unit here has one attack), so at most `2n` actions fire and the loop
//! runs at most `2n + 1` times.
//!
//! It is additionally *confluent* — the result does not depend on the order
//! cells are visited — **provided the declared destinations are pairwise
//! distinct**, which the sampler guarantees by masking claimed hexes. Under that
//! hypothesis the process is monotone:
//!
//! - Within one side's turn friendly units are passable, so the movement mask is
//!   just `!enemy`, shared by every unit of that side. Moving a friendly never
//!   changes another friendly's reachability — only destination emptiness.
//! - Only a kill changes `!enemy`, and only ever by growing it. Kills grow the
//!   empty set too.
//!
//! so no action can un-enable a previously enabled action, and the loop reaches
//! the unique maximal executable subset. It settles in two or three rounds in
//! practice.
//!
//! Without distinctness the guarantee is weaker but still safe: two units naming
//! one hex contend for it, the lower cell index wins, and the loser's move is
//! dropped. The result stays deterministic and legal — just order-dependent.
//!
//! The realizer is deliberately **best-effort and deterministic, not optimal**.
//! It *decodes* a declaration; it does not *choose* the turn. The reference
//! engine reaches for a SAT solver because it is picking the combat itself —
//! here the policy already picked, and unachievable declarations are simply
//! dropped and reported back through [`Realized::spec`], which is what the nets
//! train on.

use crate::assign;
use crate::bitboard::{self, bit, has, Bb, FULL};
use crate::game::{Minions, MinionsMove, MinionsState, PHASE_SPAWN};
use crate::map::{cell_name, dist, NO_CELL, NUM_CELLS};
use crate::units::{Attack, Unit, BUYABLE};
use playground_core::Game;

/// No declared move for this cell.
pub const NO_MOVE: u8 = 255;

/// Node budget for the complete assignment search. Generous relative to how
/// rarely it runs (greedy leaves work undone on the order of 1e-6 of kills), but
/// bounded so a pathological melee cannot stall self-play.
pub const SOLVER_BUDGET: u64 = 20_000;

/// Hexes within distance 3 — the whole movement space of the fastest unit.
pub const N_OFFSETS: usize = 37;

/// Target classes: stay / kill / unsummon.
pub const N_TARGETS: usize = 3;
/// Spawn classes: nothing, plus the five buyable units.
pub const N_SPAWNS: usize = 6;
/// Planes describing a partially committed declaration, fed to the diffusion and
/// energy heads. Move destinations go in as `(dx, dy)` rather than a 37-way
/// one-hot — same information, 34 fewer planes.
///
/// `0,1` move dx/3, dy/3 · `2` move committed · `3..5` target one-hot ·
/// `6..11` spawn one-hot.
///
/// Only *moves* carry a commit flag. Target and spawn are one-hot over class
/// sets that already contain the "nothing" option, so all-zeros already means
/// uncommitted; their old commit planes were exactly the sum of their one-hots.
/// A move committed to offset 0 ("stay") writes (0, 0), which is why that one
/// cannot be inferred and keeps its flag.
pub const DECL_PLANES: usize = 12;

/// Planes for the **energy** head, which only ever scores complete
/// declarations. With everything committed the move-commit flag degenerates
/// into a liveness mask — something `h` already carries — so it is dropped:
/// `0,1` move dx/3, dy/3 · `2..4` target one-hot · `5..10` spawn one-hot.
pub const ENERGY_PLANES: usize = DECL_PLANES - 1;

pub const TARGET_STAY: u8 = 0;
pub const TARGET_KILL: u8 = 1;
pub const TARGET_UNSUMMON: u8 = 2;

/// Relative move offsets, `(dx, dy)`. Index 0 is `(0, 0)` — stay put.
pub const OFFSETS: [(i8, i8); N_OFFSETS] = build_offsets();

const fn build_offsets() -> [(i8, i8); N_OFFSETS] {
    let mut out = [(0i8, 0i8); N_OFFSETS];
    let mut n = 1; // slot 0 is (0, 0)
    let mut dy = -3i32;
    while dy <= 3 {
        let mut dx = -3i32;
        while dx <= 3 {
            let s = dx + dy;
            let ax = if dx < 0 { -dx } else { dx };
            let ay = if dy < 0 { -dy } else { dy };
            let asum = if s < 0 { -s } else { s };
            let mut m = ax;
            if ay > m {
                m = ay;
            }
            if asum > m {
                m = asum;
            }
            if m <= 3 && !(dx == 0 && dy == 0) {
                out[n] = (dx as i8, dy as i8);
                n += 1;
            }
            dx += 1;
        }
        dy += 1;
    }
    out
}

/// `MOVE_DEST[cell][offset]` — the destination cell, or [`NO_CELL`] off-board.
pub const MOVE_DEST: [[u8; N_OFFSETS]; NUM_CELLS] = build_move_dest();

/// The offset slot that walks `from` to `to`, if one exists.
///
/// The inverse of [`MOVE_DEST`]; a linear scan over 37 entries, which is cheaper
/// than carrying a second table.
pub fn offset_index(from: u8, to: u8) -> Option<u8> {
    (0..N_OFFSETS as u8).find(|&k| MOVE_DEST[from as usize][k as usize] == to)
}

const fn build_move_dest() -> [[u8; N_OFFSETS]; NUM_CELLS] {
    // Board width comes from `map::N`; it used to be hardcoded to 10, which
    // silently produced wrong destinations (and off-board indices) the moment
    // the board was resized.
    let n = crate::map::N as i32;
    let mut t = [[NO_CELL; N_OFFSETS]; NUM_CELLS];
    let mut c = 0;
    while c < NUM_CELLS {
        let x = (c as i32) % n;
        let y = (c as i32) / n;
        let mut k = 0;
        while k < N_OFFSETS {
            let nx = x + OFFSETS[k].0 as i32;
            let ny = y + OFFSETS[k].1 as i32;
            if nx >= 0 && nx < n && ny >= 0 && ny < n {
                t[c][k] = (ny * n + nx) as u8;
            }
            k += 1;
        }
        c += 1;
    }
    t
}

// ── The declaration ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TurnSpec {
    /// Offset index into [`OFFSETS`] for the unit here, or [`NO_MOVE`].
    pub moves: [u8; NUM_CELLS],
    /// [`TARGET_STAY`] / [`TARGET_KILL`] / [`TARGET_UNSUMMON`] for the enemy here.
    pub targets: [u8; NUM_CELLS],
    /// 0 = nothing, else `unit as u8` (1..=5) to create here.
    pub spawns: [u8; NUM_CELLS],
}

impl Default for TurnSpec {
    fn default() -> Self {
        Self::empty()
    }
}

/// For each move offset, the index of its negation — the same displacement seen
/// from the rotated frame. Built at compile time from [`OFFSETS`].
pub const NEG_OFFSET: [u8; N_OFFSETS] = build_neg_offsets();

const fn build_neg_offsets() -> [u8; N_OFFSETS] {
    let mut out = [0u8; N_OFFSETS];
    let mut i = 0;
    while i < N_OFFSETS {
        let (dx, dy) = OFFSETS[i];
        let mut j = 0;
        // The offset set is closed under negation (asserted in tests), so this
        // always finds a match.
        while j < N_OFFSETS {
            if OFFSETS[j].0 == -dx && OFFSETS[j].1 == -dy {
                out[i] = j as u8;
                break;
            }
            j += 1;
        }
        i += 1;
    }
    out
}

impl TurnSpec {
    /// The same declaration expressed in the 180°-rotated frame.
    ///
    /// Cells reverse and every move offset becomes its negation, so a turn
    /// planned in the mover's canonical view maps back onto the real board.
    /// Self-inverse.
    pub fn rotated(&self) -> Self {
        let mut out = Self::empty();
        for i in 0..NUM_CELLS {
            let j = crate::map::rot(i);
            out.moves[j] = match self.moves[i] {
                NO_MOVE => NO_MOVE,
                k if (k as usize) < N_OFFSETS => NEG_OFFSET[k as usize],
                other => other,
            };
            out.targets[j] = self.targets[i];
            out.spawns[j] = self.spawns[i];
        }
        out
    }
}

impl TurnSpec {
    pub const fn empty() -> Self {
        Self {
            moves: [NO_MOVE; NUM_CELLS],
            targets: [TARGET_STAY; NUM_CELLS],
            spawns: [0; NUM_CELLS],
        }
    }

    /// Wire form: three bytes per cell. Used for replay records and dedupe keys.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(3 * NUM_CELLS);
        v.extend_from_slice(&self.moves);
        v.extend_from_slice(&self.targets);
        v.extend_from_slice(&self.spawns);
        v
    }

    pub fn from_bytes(b: &[u8]) -> anyhow::Result<Self> {
        if b.len() != 3 * NUM_CELLS {
            anyhow::bail!("turn spec: expected {} bytes, got {}", 3 * NUM_CELLS, b.len());
        }
        let mut s = Self::empty();
        s.moves.copy_from_slice(&b[..NUM_CELLS]);
        s.targets.copy_from_slice(&b[NUM_CELLS..2 * NUM_CELLS]);
        s.spawns.copy_from_slice(&b[2 * NUM_CELLS..]);
        Ok(s)
    }

    /// A declaration drawn uniformly over the whole space, ignoring legality
    /// entirely. Stress input for the realizer — most of it is unachievable and
    /// must be dropped rather than panicked on.
    pub fn wild<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let mut s = Self::empty();
        for c in 0..NUM_CELLS {
            s.moves[c] = rng.gen_range(0..N_OFFSETS as u8);
            s.targets[c] = rng.gen_range(0..3u8);
            s.spawns[c] = rng.gen_range(0..6u8);
        }
        s
    }

    /// A declaration sampled only from things that are individually plausible:
    /// reachable destinations, live enemies, affordable units on real spawn
    /// cells. Joint feasibility is still not guaranteed — that is the
    /// realizer's job — but this is a usable random *player*, where [`wild`]
    /// is not.
    ///
    /// [`wild`]: TurnSpec::wild
    /// A random turn built by **actually playing one**: pick uniformly among the
    /// engine's legal moves, apply, repeat until the turn ends, then read the
    /// declaration back off what happened.
    ///
    /// This exists because [`random_plausible`](Self::random_plausible) declares
    /// kill/unsummon per enemy *at random*, with no regard for whether any
    /// attacker could achieve it — and the soak measures only **3.7% of those
    /// declared kills surviving realization** (against 100% of its moves). The
    /// injected candidates were therefore move-only in practice, leaving combat
    /// — the coordinated "three units converge and that skeleton dies" that the
    /// joint sampler exists for — essentially uncovered. Support collapse in the
    /// combat dimension is exactly what random injection is supposed to prevent.
    ///
    /// Playing the turn out instead makes every declaration achievable by
    /// construction, so fidelity is ~100% on all three fields and the candidate
    /// genuinely samples the reachable turn space.
    ///
    /// It costs more than `random_plausible` — a `legal_moves` enumeration per
    /// action rather than a few bitboard ops — but it uses the engine's own
    /// definition of legality rather than reimplementing the rules, which is the
    /// right trade for a correctness-critical exploration path.
    pub fn random_playout<R: rand::Rng + ?Sized>(state: &MinionsState, rng: &mut R) -> Self {
        Self::from_play(state, |s| {
            let moves = Minions::legal_moves(s);
            if moves.is_empty() {
                None
            } else {
                Some(moves[rng.gen_range(0..moves.len())])
            }
        })
    }

    /// Play one turn with `choose` and read back the declaration describing what
    /// happened.
    ///
    /// Shared by [`Self::random_playout`] and by the greedy seat in `eval.rs`,
    /// which needs the same thing for a different chooser. Reconstructing this
    /// by *diffing* the before and after boards does not work — with several
    /// units moving there is no way to tell which one ended where — so the
    /// origin of each unit is tracked as it moves.
    pub fn from_play<F>(state: &MinionsState, mut choose: F) -> Self
    where
        F: FnMut(&MinionsState) -> Option<MinionsMove>,
    {
        let me = state.side;
        let mut spec = Self::empty();

        // Where each of our units started, tracked as they move so a declaration
        // can be written against the unit's *original* cell.
        let mut origin = [NO_CELL; NUM_CELLS];
        for c in 0..NUM_CELLS {
            if let Some((side, _)) = state.piece_at(c as u8) {
                if side == me {
                    origin[c] = c as u8;
                }
            }
        }

        let mut cur = state.clone();
        // Each unit moves at most once and attacks at most once, plus spawns and
        // two phase closes; the cap is a guard against a rules change making this
        // unbounded, not an expected limit.
        let cap = 4 * NUM_CELLS;
        for _ in 0..cap {
            if cur.side != me || Minions::outcome(&cur) != playground_core::Outcome::InProgress {
                break;
            }
            let Some(mv) = choose(&cur) else { break };
            match mv {
                MinionsMove::Move { from, to } => {
                    let o = origin[from as usize];
                    if o != NO_CELL {
                        if let Some(k) = offset_index(o, to) {
                            spec.moves[o as usize] = k;
                        }
                        origin[to as usize] = o;
                        origin[from as usize] = NO_CELL;
                    }
                }
                MinionsMove::Attack { from, to } => {
                    // Record the *outcome*, not the blow: a strike that only
                    // chips is worth nothing (damage clears at end of turn) and
                    // must stay `TARGET_STAY`.
                    let before = cur.piece_at(to);
                    let after = Minions::apply(&cur, &mv);
                    if before.is_some() && after.piece_at(to) != before {
                        let unsummon = cur
                            .piece_at(from)
                            .is_some_and(|(_, u)| matches!(u.stats().attack, Attack::Unsummon));
                        spec.targets[to as usize] =
                            if unsummon { TARGET_UNSUMMON } else { TARGET_KILL };
                    }
                    cur = after;
                    continue;
                }
                MinionsMove::Spawn { unit, at } => {
                    spec.spawns[at as usize] = unit as u8;
                }
                MinionsMove::Buy(_) | MinionsMove::EndPhase => {}
            }
            cur = Minions::apply(&cur, &mv);
        }
        spec
    }

    pub fn random_plausible<R: rand::Rng + ?Sized>(state: &MinionsState, rng: &mut R) -> Self {
        let mut s = Self::empty();
        let me = state.side;
        let enemy = side_bb(state, 1 - me);
        let mut claimed: Bb = 0;

        for c in 0..NUM_CELLS as u8 {
            let Some((side, unit)) = state.piece_at(c) else { continue };
            if side != me {
                continue;
            }
            let st = unit.stats();
            let pass = if st.flying { FULL } else { FULL & !enemy };
            // Destination must be empty and unclaimed, so declarations don't
            // collide — the same masking the diffusion sampler will do.
            let open = bitboard::reachable(bit(c), st.speed, pass) & empty_bb(state) & !claimed;
            let choices: Vec<u8> = (0..N_OFFSETS as u8)
                .filter(|&k| {
                    let d = MOVE_DEST[c as usize][k as usize];
                    k == 0 || (d != NO_CELL && bitboard::has(open, d))
                })
                .collect();
            let k = choices[rng.gen_range(0..choices.len())];
            s.moves[c as usize] = k;
            let d = MOVE_DEST[c as usize][k as usize];
            if k != 0 && d != NO_CELL {
                claimed |= bit(d);
            }
        }

        for c in bitboard::cells(enemy) {
            let (_, unit) = state.piece_at(c).expect("enemy bitboard is in sync");
            // Unsummon only removes non-persistent units, and only the
            // necromancer can do it at all.
            let can_unsummon = !unit.stats().persistent;
            s.targets[c as usize] = match rng.gen_range(0..3u8) {
                0 => TARGET_STAY,
                1 => TARGET_KILL,
                _ if can_unsummon => TARGET_UNSUMMON,
                _ => TARGET_KILL,
            };
        }

        let mut purse = state.money[me as usize];
        for at in state.spawn_cells(me) {
            let affordable: Vec<Unit> = BUYABLE
                .iter()
                .copied()
                .filter(|u| {
                    let slot = u.reinforcement_slot().expect("buyable");
                    state.reinforcements[me as usize][slot] > 0 || u.stats().cost <= purse
                })
                .collect();
            if affordable.is_empty() || rng.gen_bool(0.5) {
                continue;
            }
            let u = affordable[rng.gen_range(0..affordable.len())];
            s.spawns[at as usize] = u as u8;
            purse = purse.saturating_sub(u.stats().cost);
        }
        s
    }
}

impl std::fmt::Debug for TurnSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TurnSpec {{")?;
        for (c, &m) in self.moves.iter().enumerate() {
            let (t, sp) = (self.targets[c], self.spawns[c]);
            if m == NO_MOVE && t == TARGET_STAY && sp == 0 {
                continue;
            }
            write!(f, "  {}:", cell_name(c as u8))?;
            if m != NO_MOVE && m != 0 {
                let d = MOVE_DEST[c][m as usize];
                if d != NO_CELL {
                    write!(f, " ->{}", cell_name(d))?;
                }
            }
            match t {
                TARGET_KILL => write!(f, " kill")?,
                TARGET_UNSUMMON => write!(f, " unsummon")?,
                _ => {}
            }
            if sp != 0 {
                if let Some(u) = Unit::from_index(sp) {
                    write!(f, " spawn {}", u.stats().name)?;
                }
            }
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

/// Which declaration variables actually exist in a position. Cells outside these
/// masks have nothing to declare, so the sampler never proposes for them and the
/// trainer never masks them.
#[derive(Clone, Copy, Debug)]
pub struct LiveMask {
    /// Cells holding one of our units — these have a move to declare.
    pub moves: Bb,
    /// Cells holding an enemy — these have a fate to declare.
    pub targets: Bb,
    /// Empty cells next to one of our spawners.
    pub spawns: Bb,
}

impl LiveMask {
    pub fn of(state: &MinionsState) -> Self {
        let me = state.side;
        let spawns = state.spawn_bb(me);
        Self { moves: side_bb(state, me), targets: side_bb(state, 1 - me), spawns }
    }

    /// Total number of variables to decide — the `n` that sets `T = n` in the
    /// sampler's exact-density diagnostic mode.
    pub fn count(&self) -> usize {
        (self.moves.count_ones() + self.targets.count_ones() + self.spawns.count_ones()) as usize
    }
}

/// A declaration under construction: values, plus which of them are committed.
#[derive(Clone, Copy)]
pub struct Partial {
    pub spec: TurnSpec,
    /// Bit 0 move, bit 1 target, bit 2 spawn.
    pub committed: [u8; NUM_CELLS],
}

pub const C_MOVE: u8 = 1;
pub const C_TARGET: u8 = 2;
pub const C_SPAWN: u8 = 4;

impl Default for Partial {
    fn default() -> Self {
        Self::new()
    }
}

impl Partial {
    pub const fn new() -> Self {
        Self { spec: TurnSpec::empty(), committed: [0; NUM_CELLS] }
    }

    #[inline]
    pub fn is_committed(&self, cell: u8, field: u8) -> bool {
        self.committed[cell as usize] & field != 0
    }

    #[inline]
    pub fn commit(&mut self, cell: u8, field: u8, value: u8) {
        match field {
            C_MOVE => self.spec.moves[cell as usize] = value,
            C_TARGET => self.spec.targets[cell as usize] = value,
            _ => self.spec.spawns[cell as usize] = value,
        }
        self.committed[cell as usize] |= field;
    }

    /// A [`Partial`] with every live variable of `spec` marked committed — the
    /// form the energy head scores, since `f` sees whole declarations.
    pub fn fully_committed(spec: &TurnSpec, live: &LiveMask) -> Self {
        let mut p = Self { spec: *spec, committed: [0; NUM_CELLS] };
        for c in crate::bitboard::cells(live.moves) {
            p.committed[c as usize] |= C_MOVE;
        }
        for c in crate::bitboard::cells(live.targets) {
            p.committed[c as usize] |= C_TARGET;
        }
        for c in crate::bitboard::cells(live.spawns) {
            p.committed[c as usize] |= C_SPAWN;
        }
        p
    }

    /// Write the `DECL_PLANES × NUM_CELLS` conditioning planes, channel-major to
    /// match a `(C, H, W)` conv input.
    pub fn write_planes(&self, out: &mut [f32]) {
        self.write_planes_inner(out, true)
    }

    /// As [`Self::write_planes`] but without the move-commit plane, for the
    /// energy head — see [`ENERGY_PLANES`].
    pub fn write_energy_planes(&self, out: &mut [f32]) {
        self.write_planes_inner(out, false)
    }

    fn write_planes_inner(&self, out: &mut [f32], commit_plane: bool) {
        let width = if commit_plane { DECL_PLANES } else { ENERGY_PLANES };
        debug_assert_eq!(out.len(), width * NUM_CELLS);
        out.fill(0.0);
        // Everything after the move fields shifts down one when the commit
        // plane is absent.
        let base = if commit_plane { 3 } else { 2 };
        let plane = |i: usize, c: usize| i * NUM_CELLS + c;
        for c in 0..NUM_CELLS {
            if self.committed[c] & C_MOVE != 0 {
                let k = self.spec.moves[c];
                if (k as usize) < N_OFFSETS {
                    let (dx, dy) = OFFSETS[k as usize];
                    out[plane(0, c)] = dx as f32 / 3.0;
                    out[plane(1, c)] = dy as f32 / 3.0;
                }
                if commit_plane {
                    out[plane(2, c)] = 1.0;
                }
            }
            // No separate commit plane for targets or spawns: both are one-hot
            // over a class set that already includes the "nothing" option, so an
            // uncommitted variable is all-zeros and a committed one always has
            // exactly one bit set. The old planes 6 and 13 were exactly
            // `sum(planes 3..5)` and `sum(planes 7..12)`.
            //
            // Moves are different, and plane 2 stays: offset 0 is "stay", so a
            // cell committed to staying writes (0, 0) — indistinguishable from
            // the uncommitted default without the flag.
            if self.committed[c] & C_TARGET != 0 {
                let t = (self.spec.targets[c] as usize).min(N_TARGETS - 1);
                out[plane(base + t, c)] = 1.0;
            }
            if self.committed[c] & C_SPAWN != 0 {
                let sp = (self.spec.spawns[c] as usize).min(N_SPAWNS - 1);
                out[plane(base + N_TARGETS + sp, c)] = 1.0;
            }
        }
    }
}

// ── Realization ───────────────────────────────────────────────────────────────

/// How much of a declaration actually happened. Low fidelity means the policy is
/// declaring fantasies and the realizer is quietly papering over it, so this is
/// logged per generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fidelity {
    pub declared_moves: u32,
    pub done_moves: u32,
    pub declared_kills: u32,
    pub done_kills: u32,
    pub declared_spawns: u32,
    pub done_spawns: u32,
    /// Fixed-point iterations used, for tuning the loop bound.
    pub iterations: u32,
    /// Times the *complete search* ran — greedy fell short and the cheap
    /// deficiency certificate could not settle it.
    pub solver_calls: u32,
    /// Times that search found an assignment greedy had missed.
    pub solver_rescued: u32,
    /// Times the declaration was *proven* unrealizable — by the deficiency
    /// bound or by an exhausted search — so dropping it is correct rather than
    /// a guess.
    pub certified_infeasible: u32,
    /// Times it ran out of budget — neither realized nor disproven. This is the
    /// only case where the realizer is genuinely guessing, so watch it.
    pub unresolved: u32,
}

#[derive(Clone, Debug)]
pub struct Realized {
    /// The legal atomic sequence, including the `EndPhase`s that close the turn.
    pub actions: Vec<MinionsMove>,
    /// State after applying `actions`.
    pub next: MinionsState,
    /// What actually happened — the spec the nets score and train on.
    pub spec: TurnSpec,
    pub fidelity: Fidelity,
}

#[inline]
fn side_bb(s: &MinionsState, side: u8) -> Bb {
    let mut b = 0;
    for c in 0..NUM_CELLS as u8 {
        if s.piece_at(c).is_some_and(|(sd, _)| sd == side) {
            b |= bit(c);
        }
    }
    b
}

#[inline]
fn empty_bb(s: &MinionsState) -> Bb {
    let mut b = 0;
    for c in 0..NUM_CELLS as u8 {
        if s.cells[c as usize] == 0 {
            b |= bit(c);
        }
    }
    b
}

/// Damage `attacker` contributes toward killing `target`.
///
/// `Unsummon` is target-dependent, which is easy to get wrong: against a
/// *non-persistent* unit it removes the unit outright rather than damaging it —
/// that is a [`TARGET_UNSUMMON`] declaration, not a contribution to a damage
/// pool, and the owner keeps the unit instead of paying a rebate. Against a
/// *persistent* unit it deals 1, which can be the finishing blow: a ghost
/// (defense 4) dies to a serpent's 3 plus a necromancer's 1.
#[inline]
pub(crate) fn kill_damage_against(attacker: Unit, target: Unit) -> u8 {
    match attacker.stats().attack {
        Attack::Damage(d) => d,
        Attack::Unsummon => u8::from(target.stats().persistent),
    }
}

/// Compile `spec` into a legal turn from `state`.
/// Field weights for [`distance`].
///
/// Deliberately asymmetric. Movement is smooth in value — a unit one hex off is
/// a nearly equivalent turn — but combat is not: a declaration that kills the
/// necromancer and one that does not are worlds apart no matter how similar the
/// movement. Blurring those together is the way a kernel makes search *worse*
/// than plain PUCT, so target mismatches count double.
pub const W_MOVE: f32 = 1.0;
pub const W_TARGET: f32 = 2.0;
pub const W_SPAWN: f32 = 1.0;

/// Resolve a declared move to its destination cell, or the origin for "stay".
#[inline]
fn declared_dest(c: u8, k: u8) -> u8 {
    if k == NO_MOVE || k as usize >= N_OFFSETS {
        return c;
    }
    let d = MOVE_DEST[c as usize][k as usize];
    if d == NO_CELL {
        c
    } else {
        d
    }
}

/// Distance between two turn declarations at the same state, in `[0, W_MOVE +
/// W_TARGET + W_SPAWN]`.
///
/// This is the metric the KR-PUCT kernel regresses over: turns that move the
/// same units to nearby hexes, assign the same fates to the same enemies and
/// buy the same things are "near", and near turns share search statistics.
///
/// Indexed by **origin cell** throughout, because within one turn each unit
/// moves at most once, so its origin identifies it for the whole turn — the
/// same invariant `realize` relies on. A unit that stays has `dest == origin`,
/// so "moved vs did not move" needs no special case; it falls out as distance.
///
/// Denominators are properties of the *state*, not of the pair, so the metric
/// is comparable across candidates at one node — which is all the kernel needs.
pub fn distance(state: &MinionsState, a: &TurnSpec, b: &TurnSpec) -> f32 {
    let me = state.side;

    let (mut move_sum, mut n_units) = (0.0f32, 0u32);
    let (mut target_diff, mut n_enemies) = (0u32, 0u32);
    for c in 0..NUM_CELLS as u8 {
        match state.piece_at(c) {
            Some((sd, _)) if sd == me => {
                n_units += 1;
                let da = declared_dest(c, a.moves[c as usize]);
                let db = declared_dest(c, b.moves[c as usize]);
                move_sum += crate::map::dist(da, db) as f32;
            }
            Some(_) => {
                n_enemies += 1;
                if a.targets[c as usize] != b.targets[c as usize] {
                    target_diff += 1;
                }
            }
            None => {}
        }
    }

    let mut spawn_diff = 0u32;
    let mut n_spawn = 0u32;
    for c in state.spawn_cells(me) {
        n_spawn += 1;
        if a.spawns[c as usize] != b.spawns[c as usize] {
            spawn_diff += 1;
        }
    }

    // MAX_SPEED normalises the move term: 3 is the largest hex distance any unit
    // can declare, so a unit sent to opposite extremes contributes 1.0.
    const MAX_SPEED: f32 = 3.0;
    let mv = if n_units == 0 { 0.0 } else { move_sum / (MAX_SPEED * n_units as f32) };
    let tg = if n_enemies == 0 { 0.0 } else { target_diff as f32 / n_enemies as f32 };
    let sp = if n_spawn == 0 { 0.0 } else { spawn_diff as f32 / n_spawn as f32 };
    W_MOVE * mv + W_TARGET * tg + W_SPAWN * sp
}

/// Gaussian kernel over [`distance`]. `sigma <= 0` yields a hard delta, which
/// makes KR-PUCT collapse to exactly the PUCT it replaces — the identity that
/// keeps the flat search available as a mode.
#[inline]
pub fn kernel(state: &MinionsState, a: &TurnSpec, b: &TurnSpec, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return if a == b { 1.0 } else { 0.0 };
    }
    let d = distance(state, a, b);
    (-(d * d) / (2.0 * sigma * sigma)).exp()
}

/// Execute declaration `spec` at `state`: the legal turn it describes, as far as
/// the position allows.
///
/// Resolved in the mover's canonical frame (the side to move as side 0), then
/// rotated back. Resolution is order-dependent -- units move in cell order, and
/// a unit that moves first can free or take a cell another one wanted -- and
/// the 180° rotation reverses that order. Resolved in the real frame, the same
/// declaration reached a different position for 2.1% of side-1 turns than it
/// did in the canonical frame the sampler proposes and scores in, so the
/// search played a turn the nets never saw. Pinning the frame to the mover
/// makes `realize` commute with the rotation exactly.
pub fn realize(state: &MinionsState, spec: &TurnSpec) -> Realized {
    if state.side == 0 {
        return realize_canonical(state, spec);
    }
    let r = realize_canonical(&state.rotated(), &spec.rotated());
    Realized {
        actions: r.actions.iter().map(MinionsMove::rotated).collect(),
        next: r.next.rotated(),
        spec: r.spec.rotated(),
        fidelity: r.fidelity,
    }
}

/// [`realize`] for a position whose mover is side 0.
fn realize_canonical(state: &MinionsState, spec: &TurnSpec) -> Realized {
    realize_canonical_to(state, spec, false)
}

/// The attack part of [`realize`] only, for a side-0 (canonical) state: `next`
/// is the position right after the attack phase closes -- the spawn phase, or
/// the next turn on turn 0, or a finished game. Spawns are not declared.
pub fn realize_attack(state: &MinionsState, spec: &TurnSpec) -> Realized {
    debug_assert_eq!(state.side, 0);
    realize_canonical_to(state, spec, true)
}

#[inline]
fn realize_canonical_to(state: &MinionsState, spec: &TurnSpec, stop_at_spawn: bool) -> Realized {
    debug_assert_eq!(state.side, 0);
    let me = state.side;
    let mut s = state.clone();
    let mut actions = Vec::new();
    let mut out = TurnSpec::empty();
    let mut fid = Fidelity::default();

    // Pending declared moves, indexed by *origin*: each unit moves at most once,
    // so the origin identifies it for the whole turn.
    let mut want = [NO_CELL; NUM_CELLS];
    for c in 0..NUM_CELLS as u8 {
        if s.piece_at(c).is_none_or(|(sd, _)| sd != me) {
            continue;
        }
        let k = spec.moves[c as usize];
        if k == NO_MOVE || k as usize >= N_OFFSETS {
            continue;
        }
        let d = MOVE_DEST[c as usize][k as usize];
        if d != NO_CELL && d != c {
            want[c as usize] = d;
            fid.declared_moves += 1;
        } else if k == 0 {
            out.moves[c as usize] = 0; // an honoured "stay"
        }
    }
    fid.declared_kills = (0..NUM_CELLS)
        .filter(|&c| spec.targets[c] != TARGET_STAY && is_enemy(&s, c as u8, me))
        .count() as u32;

    loop {
        fid.iterations += 1;
        let mut progress = false;

        // ── M: every pending move whose destination is now free and reachable.
        // `enemy` is fixed for this pass (only kills change it), so the movement
        // mask is computed once; only emptiness shifts as units move.
        let enemy = side_bb(&s, 1 - me);
        let ground_pass = FULL & !enemy;
        let mut empty = empty_bb(&s);
        for o in 0..NUM_CELLS as u8 {
            let d = want[o as usize];
            if d == NO_CELL {
                continue;
            }
            let Some((sd, unit)) = s.piece_at(o) else {
                want[o as usize] = NO_CELL; // died before it could move
                continue;
            };
            if sd != me || !s.can_move_piece(o) {
                continue;
            }
            if !bitboard::has(empty, d) {
                continue; // occupied for now; a later iteration may free it
            }
            let st = unit.stats();
            let pass = if st.flying { FULL } else { ground_pass };
            if !bitboard::has(bitboard::reachable(bit(o), st.speed, pass), d) {
                continue; // blocked; killing the blocker may open it later
            }
            let mv = MinionsMove::Move { from: o, to: d };
            s = Minions::apply(&s, &mv);
            actions.push(mv);
            empty = (empty | bit(o)) & !bit(d);
            out.moves[o as usize] = spec.moves[o as usize];
            want[o as usize] = NO_CELL;
            fid.done_moves += 1;
            progress = true;
        }

        // ── A: every declared kill/unsummon that can now be completed.
        progress |= attack_pass(&mut s, spec, &mut out, &mut actions, &mut fid, &want, me);

        if !progress || s.winner != 0 {
            break;
        }
    }

    // ── Close the attack phase. On turn 0 this ends the turn outright.
    if s.winner == 0 {
        actions.push(MinionsMove::EndPhase);
        s = Minions::apply(&s, &MinionsMove::EndPhase);
    }

    // ── S: spawns.
    if s.winner == 0 && s.phase == PHASE_SPAWN && !stop_at_spawn {
        fid.declared_spawns = spec
            .spawns
            .iter()
            .filter(|&&c| c != 0 && Unit::from_index(c).is_some())
            .count() as u32;
        // One pass suffices: spawn chaining is illegal, so the set of open cells
        // is fixed for the whole phase and cannot grow as units are placed.
        let open = s.spawn_bb(me);
        for at in 0..NUM_CELLS as u8 {
            let code = spec.spawns[at as usize];
            if code == 0 || !has(open, at) {
                continue;
            }
            let Some(unit) = Unit::from_index(code) else { continue };
            let Some(slot) = unit.reinforcement_slot() else { continue };
            if s.reinforcements[me as usize][slot] == 0 {
                if s.money[me as usize] < unit.stats().cost {
                    continue;
                }
                let buy = MinionsMove::Buy(unit);
                s = Minions::apply(&s, &buy);
                actions.push(buy);
            }
            let sp = MinionsMove::Spawn { unit, at };
            s = Minions::apply(&s, &sp);
            actions.push(sp);
            out.spawns[at as usize] = code;
            fid.done_spawns += 1;
        }
        actions.push(MinionsMove::EndPhase);
        s = Minions::apply(&s, &MinionsMove::EndPhase);
    }

    Realized { actions, next: s, spec: out, fidelity: fid }
}

#[inline]
fn is_enemy(s: &MinionsState, c: u8, me: u8) -> bool {
    s.piece_at(c).is_some_and(|(sd, _)| sd != me)
}

/// One attack pass: execute every declared kill/unsummon that the current
/// position can complete. Returns whether anything fired.
///
/// Assignment is greedy, **most-constrained target first** — a target with only
/// one possible attacker is served before one with five, which is what keeps a
/// shared attacker from being squandered. Exact assignment is bin covering and
/// NP-hard in general, but range is 1 so the bipartite graph has degree ≤ 6 and
/// components are 2–5 units wide; a per-component exact DP is a cheap upgrade if
/// the fidelity metric ever says greedy is leaving kills on the table.
fn attack_pass(
    s: &mut MinionsState,
    spec: &TurnSpec,
    out: &mut TurnSpec,
    actions: &mut Vec<MinionsMove>,
    fid: &mut Fidelity,
    want: &[u8; NUM_CELLS],
    me: u8,
) -> bool {
    let mut fired = false;
    let mut used: Bb = 0;

    // Unsummon first: only the necromancer can do it, so it competes with
    // nothing, and spending it early never costs a kill.
    for e in 0..NUM_CELLS as u8 {
        if spec.targets[e as usize] != TARGET_UNSUMMON || !is_enemy(s, e, me) {
            continue;
        }
        let (_, target) = s.piece_at(e).expect("checked");
        if target.stats().persistent {
            continue; // would be 1 wasted damage, not a removal
        }
        let Some(n) = find_necromancer(s, me) else { continue };
        if bitboard::has(used, n) || !s.can_attack_piece(n, Unit::Necromancer) {
            continue;
        }
        if dist(n, e) > Unit::Necromancer.stats().range as u32 {
            continue;
        }
        let mv = MinionsMove::Attack { from: n, to: e };
        *s = Minions::apply(s, &mv);
        actions.push(mv);
        used |= bit(n);
        out.targets[e as usize] = TARGET_UNSUMMON;
        fid.done_kills += 1;
        fired = true;
        if s.winner != 0 {
            return true;
        }
    }

    // Kills: plan first, then execute. Greedy handles essentially everything;
    // the complete search only runs on the residual, and only ever *adds*
    // certainty — it either improves on greedy or proves the declaration was
    // unrealizable in the first place.
    let kill_cells: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&e| spec.targets[e as usize] == TARGET_KILL && is_enemy(s, e, me))
        .collect();
    if kill_cells.is_empty() {
        return fired;
    }
    let atk_cells: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&a| {
            !bitboard::has(used, a)
                && s.piece_at(a).is_some_and(|(sd, u)| sd == me && s.can_attack_piece(a, u))
        })
        .collect();

    let attackers: Vec<assign::Attacker> =
        atk_cells.iter().map(|&c| assign::Attacker { cell: c, attacks: 1 }).collect();
    let targets: Vec<assign::Target> = kill_cells
        .iter()
        .map(|&e| {
            let (_, tgt) = s.piece_at(e).expect("live target");
            let mut reach: Vec<(usize, u32)> = atk_cells
                .iter()
                .enumerate()
                .filter_map(|(i, &a)| {
                    let (_, u) = s.piece_at(a)?;
                    let d = kill_damage_against(u, tgt) as u32;
                    (d > 0 && dist(a, e) <= u.stats().range as u32).then_some((i, d))
                })
                .collect();
            // Preference order, honoured by the greedy tier: units with no
            // pending move first, since spending their attack costs nothing.
            reach.sort_by_key(|&(i, d)| {
                (want[atk_cells[i] as usize] != NO_CELL, std::cmp::Reverse(d), i)
            });
            assign::Target {
                cell: e,
                need: tgt.stats().defense.saturating_sub(s.damage_on(e)) as u32,
                reach,
            }
        })
        .collect();

    let mut plan = assign::greedy(&targets, &attackers);
    if plan.iter().any(|p| p.is_none()) {
        // Cheap certificate first. Most unrealized declarations are wildly
        // impossible — a kill declared with nothing in range — and the
        // deficiency bound settles those in O(targets x reach) without paying
        // for a search. Counting them separately keeps `solver_calls` an honest
        // measure of how often the expensive path actually runs.
        if assign::hall_violation(&targets, &attackers).is_some() {
            fid.certified_infeasible += 1;
        } else {
            fid.solver_calls += 1;
            match assign::solve(&targets, &attackers, SOLVER_BUDGET) {
                assign::Assignment::Found(full) => {
                    plan = full.into_iter().map(Some).collect();
                    fid.solver_rescued += 1;
                }
                // Proven impossible: greedy's partial result stands, and
                // dropping the rest is correct rather than a fallback.
                assign::Assignment::Infeasible => fid.certified_infeasible += 1,
                assign::Assignment::Unknown => fid.unresolved += 1,
            }
        }
    }

    for (ti, slot) in plan.iter().enumerate() {
        let Some(chosen) = slot else { continue };
        let e = kill_cells[ti];
        for &ai in chosen {
            let mv = MinionsMove::Attack { from: atk_cells[ai], to: e };
            *s = Minions::apply(s, &mv);
            actions.push(mv);
            if s.winner != 0 {
                out.targets[e as usize] = TARGET_KILL;
                fid.done_kills += 1;
                return true;
            }
        }
        out.targets[e as usize] = TARGET_KILL;
        fid.done_kills += 1;
        fired = true;
    }
    fired
}


fn find_necromancer(s: &MinionsState, side: u8) -> Option<u8> {
    (0..NUM_CELLS as u8)
        .find(|&c| s.piece_at(c).is_some_and(|(sd, u)| sd == side && u.stats().necromancer))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{idx, NEIGHBORS};
    use playground_core::Outcome;
    use rand::SeedableRng;

    fn offset_of(dx: i8, dy: i8) -> u8 {
        OFFSETS.iter().position(|&o| o == (dx, dy)).expect("offset exists") as u8
    }

    /// The invariant the whole design rests on: whatever the declaration, every
    /// emitted action is legal in the state it is played from.
    fn assert_all_legal(state: &MinionsState, r: &Realized) {
        let mut s = state.clone();
        for mv in &r.actions {
            let legal = Minions::legal_moves(&s);
            assert!(
                legal.contains(mv),
                "illegal action {mv} in\n{}",
                s.render()
            );
            s = Minions::apply(&s, mv);
        }
        assert_eq!(Minions::encode_state(&s), Minions::encode_state(&r.next));
    }

    #[test]
    fn offsets_are_the_37_hexes_within_distance_three() {
        assert_eq!(OFFSETS[0], (0, 0));
        let mut seen = std::collections::HashSet::new();
        for &(dx, dy) in &OFFSETS {
            let (dx, dy) = (dx as i32, dy as i32);
            let d = dx.abs().max(dy.abs()).max((dx + dy).abs());
            assert!(d <= 3, "({dx},{dy}) is distance {d}");
            assert!(seen.insert((dx, dy)), "duplicate offset ({dx},{dy})");
        }
        assert_eq!(seen.len(), N_OFFSETS);
    }

    #[test]
    fn move_dest_agrees_with_hex_distance() {
        for c in 0..NUM_CELLS as u8 {
            for &d in &MOVE_DEST[c as usize] {
                if d != NO_CELL {
                    assert!(dist(c, d) <= 3);
                }
            }
            assert_eq!(MOVE_DEST[c as usize][0], c, "offset 0 must be stay");
        }
    }

    #[test]
    fn wild_declarations_always_realize_to_a_legal_turn() {
        // The load-bearing property test: uniform garbage in, legal turn out.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(20260824);
        let mut s = Minions::initial_state();
        for _ in 0..60 {
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            let spec = TurnSpec::wild(&mut rng);
            let r = realize(&s, &spec);
            assert_all_legal(&s, &r);
            s = r.next;
        }
    }

    #[test]
    fn plausible_declarations_always_realize_to_a_legal_turn() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut s = Minions::initial_state();
        let mut turns = 0;
        while !Minions::outcome(&s).is_terminal() {
            let spec = TurnSpec::random_plausible(&s, &mut rng);
            let r = realize(&s, &spec);
            assert_all_legal(&s, &r);
            s = r.next;
            turns += 1;
            assert!(turns < 200, "random-spec self-play failed to terminate");
        }
        assert!(Minions::outcome(&s).is_terminal());
    }

    #[test]
    fn an_empty_declaration_just_passes_the_turn() {
        let s = Minions::initial_state();
        let r = realize(&s, &TurnSpec::empty());
        assert_all_legal(&s, &r);
        assert_eq!(r.next.side, 1, "the turn should have changed hands");
        assert_eq!(r.fidelity.done_moves, 0);
        assert_eq!(r.fidelity.done_kills, 0);
    }

    #[test]
    fn a_declared_move_is_executed() {
        let s = Minions::initial_state();
        let mut spec = TurnSpec::empty();
        // b2 holds an opening zombie; step it NE to b3.
        let from = idx(1, 2);
        spec.moves[from as usize] = offset_of(0, 1);
        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.next.piece_at(idx(1, 3)), Some((0, Unit::Zombie)));
        assert_eq!(r.next.cells[from as usize], 0);
        assert_eq!(r.fidelity.done_moves, 1);
        assert_eq!(r.spec.moves[from as usize], offset_of(0, 1));
    }

    #[test]
    fn two_units_combine_to_make_a_declared_kill() {
        // A skeleton does 5, a serpent 3; a ghost has defense 4, so either alone
        // fails on damage or overkills — the pool has to be assembled.
        let mut s = Minions::initial_state();
        let target = idx(5, 5);
        let a = idx(5, 4);
        let b = idx(4, 5);
        s.cells[target as usize] = 1 + Unit::Ghost as u8 + 6; // second player's
        s.cells[a as usize] = 1 + Unit::Serpent as u8; // 3 damage
        s.cells[b as usize] = 1 + Unit::Serpent as u8; // 3 damage
        let mut spec = TurnSpec::empty();
        spec.targets[target as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.next.cells[target as usize], 0, "ghost should be dead");
        assert_eq!(r.fidelity.done_kills, 1);
    }

    #[test]
    fn the_complete_search_rescues_a_kill_greedy_would_miss() {
        // Two ghosts (defense 4). One skeleton (5) reaches both; a serpent (3)
        // and a zombie (1) reach only the second. Sending the skeleton to the
        // second ghost strands the first — greedy's most-constrained rule gets
        // this right, but the escalation must produce both kills regardless.
        let mut s = Minions::initial_state();
        s.cells = [0; NUM_CELLS];
        let g1 = idx(5, 5);
        let g2 = idx(5, 3);
        let skel = idx(5, 4); // adjacent to both
        s.cells[g1 as usize] = 1 + Unit::Ghost as u8 + 6;
        s.cells[g2 as usize] = 1 + Unit::Ghost as u8 + 6;
        s.cells[skel as usize] = 1 + Unit::Skeleton as u8;
        // Serpent + zombie adjacent to g2 only.
        for (n, u) in NEIGHBORS[g2 as usize]
            .iter()
            .filter(|&&n| n != NO_CELL && n != skel && dist(n, g1) > 1)
            .take(2)
            .zip([Unit::Serpent, Unit::Zombie])
        {
            s.cells[*n as usize] = 1 + u as u8;
        }
        let mut spec = TurnSpec::empty();
        spec.targets[g1 as usize] = TARGET_KILL;
        spec.targets[g2 as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_eq!(r.fidelity.done_kills, 2, "both ghosts should die");
        assert_eq!(r.next.cells[g1 as usize], 0);
        assert_eq!(r.next.cells[g2 as usize], 0);
        assert_eq!(r.fidelity.unresolved, 0, "never guess when it can be settled");
    }

    #[test]
    fn an_impossible_kill_is_certified_not_guessed() {
        // A ghost (defense 4) with only a serpent (3) in range: genuinely
        // unrealizable, and the realizer should *prove* that rather than fall
        // back blindly.
        let mut s = Minions::initial_state();
        s.cells = [0; NUM_CELLS];
        let ghost = idx(5, 5);
        s.cells[ghost as usize] = 1 + Unit::Ghost as u8 + 6;
        s.cells[idx(5, 4) as usize] = 1 + Unit::Serpent as u8;
        let mut spec = TurnSpec::empty();
        spec.targets[ghost as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_eq!(r.fidelity.done_kills, 0);
        assert_eq!(r.fidelity.certified_infeasible, 1, "should be proven, not guessed");
        assert_eq!(r.fidelity.unresolved, 0);
    }

    #[test]
    fn an_unkillable_target_is_dropped_and_takes_no_wasted_damage() {
        // One serpent (3) against a ghost (defense 4) cannot finish the job, and
        // chip damage is wasted, so nothing should be spent at all.
        let mut s = Minions::initial_state();
        let target = idx(5, 5);
        let a = idx(5, 4);
        s.cells[target as usize] = 1 + Unit::Ghost as u8 + 6;
        s.cells[a as usize] = 1 + Unit::Serpent as u8;
        let mut spec = TurnSpec::empty();
        spec.targets[target as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.next.piece_at(target), Some((1, Unit::Ghost)), "still alive");
        assert_eq!(r.fidelity.done_kills, 0);
        assert_eq!(r.fidelity.declared_kills, 1);
        assert!(
            !r.actions.iter().any(|m| matches!(m, MinionsMove::Attack { .. })),
            "no attack should be spent on a kill that cannot complete"
        );
    }

    #[test]
    fn a_kill_unblocks_a_move_through_the_dead_unit_hex() {
        // This is the case a staged pipeline gets wrong: f4 is the only 2-step
        // route from f2 to f5-ish, and it is occupied by an enemy. Killing it
        // has to happen before the move can be realized, in the same turn.
        let mut s = Minions::initial_state();
        let blocker = idx(5, 3);
        let killer = idx(5, 2);
        let mover = idx(4, 2);
        s.cells[blocker as usize] = 1 + Unit::Warg as u8 + 6; // defense 1
        s.cells[killer as usize] = 1 + Unit::Serpent as u8; // 3 damage, adjacent
        s.cells[mover as usize] = 1 + Unit::Serpent as u8; // speed 2

        let mut spec = TurnSpec::empty();
        spec.targets[blocker as usize] = TARGET_KILL;
        spec.moves[mover as usize] = offset_of(1, 1); // e2 -> f3, the freed hex

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.fidelity.done_kills, 1);
        assert_eq!(r.next.piece_at(blocker), Some((0, Unit::Serpent)), "mover took the hex");
        assert!(r.fidelity.iterations >= 2, "should have needed a second pass");
    }

    #[test]
    fn a_move_waits_for_a_friend_to_vacate() {
        // B wants A's hex; A is leaving. One extra iteration resolves it.
        let mut s = Minions::initial_state();
        // Pick a clear column: the follower steps into the hex the leader leaves.
        let a = idx(3, 3);
        let b = idx(3, 2);
        s.cells[a as usize] = 1 + Unit::Serpent as u8;
        s.cells[b as usize] = 1 + Unit::Serpent as u8;
        let mut spec = TurnSpec::empty();
        spec.moves[a as usize] = offset_of(0, 1); // d3 -> d4, vacating
        spec.moves[b as usize] = offset_of(0, 1); // d2 -> d3, into the vacated hex

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.fidelity.done_moves, 2, "both moves should land");
        assert_eq!(r.next.piece_at(a), Some((0, Unit::Serpent)));
    }

    #[test]
    fn a_lumbering_unit_that_moves_does_not_also_attack() {
        let mut s = Minions::initial_state();
        let z = idx(3, 3);
        let target = idx(3, 5);
        s.cells[z as usize] = 1 + Unit::Zombie as u8;
        s.cells[target as usize] = 1 + Unit::Warg as u8 + 6; // defense 1
        let mut spec = TurnSpec::empty();
        spec.moves[z as usize] = offset_of(0, 1); // move adjacent
        spec.targets[target as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.fidelity.done_moves, 1);
        assert_eq!(r.fidelity.done_kills, 0, "a zombie cannot move and attack");
    }

    #[test]
    fn a_declared_spawn_buys_and_places() {
        let mut s = Minions::initial_state();
        s = Minions::apply(&s, &MinionsMove::EndPhase); // past the opening turn
        s.money[1] = 10;
        let at = *s.spawn_cells(1).first().expect("second player has spawn cells");
        let mut spec = TurnSpec::empty();
        spec.spawns[at as usize] = Unit::Skeleton as u8;

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.next.piece_at(at), Some((1, Unit::Skeleton)));
        assert_eq!(r.fidelity.done_spawns, 1);
    }

    #[test]
    fn killing_the_necromancer_ends_the_turn_immediately() {
        let mut s = Minions::initial_state();
        let target = crate::map::NECRO_START[1];
        let mut spec = TurnSpec::empty();
        // Two skeletons (5 each) beat defense 7.
        for (i, &n) in crate::map::NEIGHBORS[target as usize].iter().take(2).enumerate() {
            let _ = i;
            s.cells[n as usize] = 1 + Unit::Skeleton as u8;
        }
        spec.targets[target as usize] = TARGET_KILL;

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(Minions::outcome(&r.next), Outcome::FirstWins);
        assert!(
            !matches!(r.actions.last(), Some(MinionsMove::EndPhase)),
            "no trailing EndPhase once the game is already won"
        );
    }

    #[test]
    fn spec_bytes_round_trip() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let spec = TurnSpec::wild(&mut rng);
        assert_eq!(TurnSpec::from_bytes(&spec.to_bytes()).unwrap(), spec);
        assert!(TurnSpec::from_bytes(&[0u8; 5]).is_err());
    }
}

// ── Turn-level game ───────────────────────────────────────────────────────────

/// Minions viewed as a game whose action is a **whole turn**.
///
/// The [`Game`] impl on [`Minions`] exposes atomic plies, which is what the
/// engine and the UI want. This is the view the learning stack searches over:
/// one action is one [`TurnSpec`], and the ~10^15–10^25 legal turns are never
/// enumerated — hence [`SampledGame`](playground_core::SampledGame)'s deliberate
/// lack of a `legal_moves`.
///
/// Moves carried through the search are *realized* specs, so applying one
/// reproduces exactly the position the search scored.
pub struct MinionsTurns;

impl playground_core::SampledGame for MinionsTurns {
    type State = MinionsState;
    type Move = TurnSpec;

    const PLANES: usize = crate::game::PLANES * NUM_CELLS;

    fn initial_state() -> MinionsState {
        Minions::initial_state()
    }

    fn to_move(state: &MinionsState) -> u8 {
        state.side + 1
    }

    fn apply(state: &MinionsState, spec: &TurnSpec) -> MinionsState {
        realize(state, spec).next
    }

    fn terminal_value(state: &MinionsState) -> Option<f32> {
        use playground_core::Outcome;
        match Minions::outcome(state) {
            Outcome::InProgress => None,
            Outcome::Draw => Some(0.0),
            // Value is for the side to move, and `winner` is absolute.
            Outcome::FirstWins => Some(if state.side == 0 { 1.0 } else { -1.0 }),
            Outcome::SecondWins => Some(if state.side == 1 { 1.0 } else { -1.0 }),
        }
    }

    fn encode_planes(state: &MinionsState, out: &mut [f32]) {
        for (o, b) in out.iter_mut().zip(Minions::to_features(state)) {
            *o = b as f32;
        }
    }

    fn state_key(state: &MinionsState) -> u128 {
        state.key()
    }

    /// The KR-PUCT kernel's metric over whole-turn declarations — see
    /// [`distance`]. Only consulted when `SampledParams::kernel_sigma > 0`.
    fn move_distance(state: &MinionsState, a: &TurnSpec, b: &TurnSpec) -> f32 {
        distance(state, a, b)
    }
}

#[cfg(test)]
mod turn_game_tests {
    use super::*;
    use playground_core::{SampledGame, SampledModel, SampledParams};
    use rand::SeedableRng;

    /// `realize` commutes with the seat rotation, for both movers, and the
    /// realized spec reproduces its own position. The sampler realizes in the
    /// canonical frame and the search applies the rotated realized spec in the
    /// real one, so both have to hold exactly for the two to see one position.
    /// Before `realize` pinned the frame, 2.1% of side-1 candidates disagreed.
    #[test]
    fn realize_commutes_with_the_seat_rotation() {
        use playground_core::SampledGame;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(21);
        let key = MinionsTurns::state_key;
        let mut checked = [0usize; 2];
        let mut unfixed = 0usize;
        for _ in 0..60 {
            let mut st = crate::game::Minions::initial_state();
            for _ in 0..40 {
                if crate::game::Minions::outcome(&st).is_terminal() {
                    break;
                }
                let rot = st.rotated();
                for _ in 0..8 {
                    let spec = TurnSpec::random_playout(&st, &mut rng);
                    let r = realize(&st, &spec);
                    let q = realize(&rot, &spec.rotated());
                    assert_eq!(key(&q.next), key(&r.next.rotated()), "side {}", st.side);
                    assert_eq!(q.spec, r.spec.rotated());
                    let back: Vec<_> = q.actions.iter().map(|a| a.rotated()).collect();
                    assert_eq!(back, r.actions);
                    // The actions are the turn: legal in sequence, ending at `next`.
                    let mut s = st.clone();
                    for a in &r.actions {
                        assert!(crate::game::Minions::legal_moves(&s).contains(a), "illegal {a}");
                        s = crate::game::Minions::apply(&s, a);
                    }
                    assert_eq!(key(&s), key(&r.next));
                    unfixed += (key(&realize(&st, &r.spec).next) != key(&r.next)) as usize;
                    checked[st.side as usize] += 1;
                }
                let spec = TurnSpec::random_playout(&st, &mut rng);
                st = realize(&st, &spec).next;
            }
        }
        assert!(checked[0] > 1000 && checked[1] > 1000, "{checked:?}");
        assert_eq!(unfixed, 0, "realized specs that do not reproduce their position");
    }

    /// Proposes plausible random declarations — the stand-in for the diffusion
    /// sampler, and enough to exercise the search end to end.
    struct RandomSampler(std::sync::Mutex<rand_chacha::ChaCha8Rng>);
    impl SampledModel<MinionsTurns> for RandomSampler {
        fn expand(
            &self,
            state: &MinionsState,
            k: usize,
        ) -> anyhow::Result<playground_core::sampled_mcts::Expansion<MinionsTurns>> {
            let mut rng = self.0.lock().unwrap();
            let cands = (0..k)
                .map(|_| {
                    let spec = TurnSpec::random_plausible(state, &mut *rng);
                    (realize(state, &spec).spec, 1.0, false)
                })
                .collect();
            Ok((0.0, cands))
        }
    }

    /// CPU cost of a whole search, with the random sampler standing in for
    /// the nets: selection, backup, dedupe and realize, and no GPU. Large
    /// candidate sets like production roots (~160-300 children).
    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn search_cpu_cost() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let mut positions = Vec::new();
        let mut s = Minions::initial_state();
        for _ in 0..16 {
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            positions.push(s.clone());
            s = realize(&s, &TurnSpec::random_playout(&s, &mut rng)).next;
        }
        let params = SampledParams {
            simulations: 800,
            k_candidates: 96,
            seed: 7,
            ..Default::default()
        };
        let run = || {
            let model = RandomSampler(std::sync::Mutex::new(
                rand_chacha::ChaCha8Rng::seed_from_u64(5),
            ));
            let t0 = std::time::Instant::now();
            for (i, p) in positions.iter().enumerate() {
                let r = playground_core::sampled_mcts::search::<MinionsTurns, _>(
                    &model, p.clone(), &params, i as u32,
                )
                .unwrap();
                std::hint::black_box(r.chosen);
            }
            t0.elapsed().as_secs_f64() * 1e3 / positions.len() as f64
        };
        run();
        let mut t: Vec<f64> = (0..5).map(|_| run()).collect();
        t.sort_by(f64::total_cmp);
        println!("search_cpu_cost: median {:.2} ms/search over {} positions", t[2], positions.len());
    }

    #[test]
    fn realizing_a_realized_spec_reproduces_the_position() {
        // The search stores realized specs and re-applies them, so this has to
        // hold or the tree would score one position and play another.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let mut s = Minions::initial_state();
        for _ in 0..25 {
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            let declared = TurnSpec::random_plausible(&s, &mut rng);
            let once = realize(&s, &declared);
            let twice = realize(&s, &once.spec);
            assert_eq!(
                Minions::encode_state(&once.next),
                Minions::encode_state(&twice.next),
                "replaying a realized spec must reproduce the position"
            );
            s = once.next;
        }
    }

    #[test]
    fn terminal_value_is_from_the_side_to_move() {
        let mut s = Minions::initial_state();
        s.winner = 1; // first player has won
        s.side = 0;
        assert_eq!(MinionsTurns::terminal_value(&s), Some(1.0));
        s.side = 1;
        assert_eq!(MinionsTurns::terminal_value(&s), Some(-1.0));
        s.winner = 3;
        assert_eq!(MinionsTurns::terminal_value(&s), Some(0.0));
        s.winner = 0;
        assert_eq!(MinionsTurns::terminal_value(&s), None);
    }

    /// The search backs a child's value up negating **only across a real change
    /// of side** (`sampled_mcts::backup`). An unconditional negation assumes
    /// strict alternation, which does not hold here.
    ///
    /// It does not always alternate here: `end_turn` flips the side, but a
    /// necromancer kill sets `winner` mid-turn and `realize` stops before
    /// `EndPhase`, so the winner is still "to move" in the terminal state.
    /// Graveyard wins and adjudication both happen *inside* `end_turn` and do
    /// flip. This test pins the invariant the backup relies on.
    #[test]
    fn terminal_child_value_is_negative_for_the_side_that_just_moved() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(20260831);
        let mut checked = 0;
        let mut violations = Vec::new();
        for game in 0..400u64 {
            let mut st = MinionsState::default();
            let mut guard = 0;
            while !Minions::outcome(&st).is_terminal() && guard < 60 {
                guard += 1;
                let mover = st.side;
                let spec = TurnSpec::random_playout(&st, &mut rng);
                let next = realize(&st, &spec).next;
                if Minions::encode_state(&next) == Minions::encode_state(&st) {
                    st = realize(&st, &TurnSpec::empty()).next;
                    continue;
                }
                if Minions::outcome(&next).is_terminal() {
                    let v = MinionsTurns::terminal_value(&next).expect("terminal has a value");
                    // Who actually won, absolutely.
                    let winner_side = match Minions::outcome(&next) {
                        playground_core::Outcome::FirstWins => Some(0u8),
                        playground_core::Outcome::SecondWins => Some(1u8),
                        _ => None,
                    };
                    if let Some(w) = winner_side {
                        checked += 1;
                        // The backup negates only across a real change of side
                        // (see `sampled_mcts::backup`). Apply the same rule here
                        // and the mover must receive +1 exactly when it won.
                        let flipped = next.side != mover;
                        let mover_gets = if flipped { -v } else { v };
                        let expected = if w == mover { 1.0 } else { -1.0 };
                        if (mover_gets - expected).abs() > 1e-6 {
                            violations.push((game, next.side, mover, w, v));
                        }
                    }
                }
                st = next;
            }
        }
        eprintln!("checked {checked} terminal transitions, {} violations", violations.len());
        for (g, cs, mv, w, v) in violations.iter().take(5) {
            eprintln!("  game {g}: child.side={cs} mover={mv} winner={w} terminal_value={v}");
        }
        assert!(checked > 0, "fixture produced no terminal transitions");
        assert!(
            violations.is_empty(),
            "{} of {checked} terminal transitions back up with the wrong sign",
            violations.len()
        );
    }

    #[test]
    fn state_key_separates_positions_and_matches_equal_ones() {
        let a = Minions::initial_state();
        let b = Minions::apply(&a, &MinionsMove::EndPhase);
        assert_ne!(MinionsTurns::state_key(&a), MinionsTurns::state_key(&b));
        assert_eq!(MinionsTurns::state_key(&a), MinionsTurns::state_key(&a.clone()));
    }

    #[test]
    fn planes_width_matches_the_feature_encoding() {
        let s = Minions::initial_state();
        let mut buf = vec![0f32; MinionsTurns::PLANES];
        MinionsTurns::encode_planes(&s, &mut buf);
        assert_eq!(buf.len(), Minions::to_features(&s).len());
        assert!(buf.iter().any(|&v| v != 0.0), "features should not be empty");
    }

    #[test]
    fn a_search_over_sampled_turns_plays_a_legal_game() {
        // End to end: sampled search drives a whole game, and every turn it
        // plays is legal in the position it was played from.
        let model = RandomSampler(std::sync::Mutex::new(
            rand_chacha::ChaCha8Rng::seed_from_u64(5),
        ));
        let params = SampledParams {
            simulations: 24,
            k_candidates: 6,
            temp_moves: 2,
            seed: 5,
            ..Default::default()
        };
        let mut s = Minions::initial_state();
        let mut turns = 0u32;
        while !Minions::outcome(&s).is_terminal() {
            let r = playground_core::sampled_mcts::search::<MinionsTurns, _>(
                &model, s.clone(), &params, turns,
            )
            .unwrap();
            assert!(!r.candidates.is_empty(), "sampler should always propose something");
            let spec = &r.candidates[r.chosen].0;

            let realized = realize(&s, spec);
            let mut check = s.clone();
            for mv in &realized.actions {
                assert!(
                    Minions::legal_moves(&check).contains(mv),
                    "search played an illegal action {mv}"
                );
                check = Minions::apply(&check, mv);
            }
            s = MinionsTurns::apply(&s, spec);
            turns += 1;
            assert!(turns < 100, "turn-level game should end inside the cap");
        }
        assert!(Minions::outcome(&s).is_terminal());
    }
}

#[cfg(test)]
mod coverage_tests {
    //! Does every legal turn have a preimage? These pin the answer.
    use super::*;
    use crate::map::{idx, NEIGHBORS};

    /// **Counterexample to completeness.** The necromancer's `Unsummon` deals 1
    /// damage to a *persistent* target, so it can land the finishing blow — a
    /// ghost (defense 4) dies to a serpent's 3 plus the necromancer's 1. That is
    /// a legal turn, so the resulting position must have a preimage.
    #[test]
    fn necromancer_can_finish_a_persistent_target() {
        let mut s = Minions::initial_state();
        let ghost = idx(5, 5);
        let necro = idx(5, 4);
        let serpent = idx(4, 5);
        // Clear the default setup so only these three matter.
        s.cells = [0; NUM_CELLS];
        s.cells[ghost as usize] = 1 + Unit::Ghost as u8 + 6; // enemy, def 4, persistent
        s.cells[necro as usize] = 1 + Unit::Necromancer as u8;
        s.cells[serpent as usize] = 1 + Unit::Serpent as u8; // 3 damage
        assert_eq!(dist(necro, ghost), 1);
        assert_eq!(dist(serpent, ghost), 1);

        // The turn is legal when played by hand: 3 + 1 >= 4.
        let by_hand = Minions::apply(&s, &MinionsMove::Attack { from: serpent, to: ghost });
        let by_hand = Minions::apply(&by_hand, &MinionsMove::Attack { from: necro, to: ghost });
        assert_eq!(by_hand.cells[ghost as usize], 0, "the rules allow this kill");

        let mut spec = TurnSpec::empty();
        spec.targets[ghost as usize] = TARGET_KILL;
        let r = realize(&s, &spec);
        assert_eq!(
            r.next.cells[ghost as usize], 0,
            "declaring the kill must reach the same position the rules allow"
        );
    }

    /// Spawn chaining is **illegal**: a unit spawned this turn is not a spawn
    /// source until next turn. So the open set is fixed for the whole phase, and
    /// a single ascending pass loses nothing.
    #[test]
    fn a_spawn_does_not_open_further_spawn_cells() {
        // `realize` always starts at the beginning of a turn and closes the
        // attack phase itself, so hand it a turn boundary — not a state already
        // sitting in the spawn phase.
        let mut s = Minions::initial_state();
        s = Minions::apply(&s, &MinionsMove::EndPhase); // opening turn has no spawn phase
        assert_eq!((s.side, s.phase), (1, crate::game::PHASE_ATTACK));
        s.money[1] = 20;

        // Find a spawn cell whose own neighbour is lower-indexed and not yet a
        // spawn cell — spawning on the first opens the second.
        let cells = s.spawn_cells(1);
        let mut found = None;
        for &c in &cells {
            for &n in &NEIGHBORS[c as usize] {
                if n != NO_CELL && n < c && s.cells[n as usize] == 0 && !cells.contains(&n) {
                    found = Some((c, n));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let (high, low) = found.expect("setup should offer a would-be chained spawn");

        let mut spec = TurnSpec::empty();
        spec.spawns[high as usize] = Unit::Zombie as u8;
        spec.spawns[low as usize] = Unit::Zombie as u8;
        let r = realize(&s, &spec);
        assert_eq!(r.next.piece_at(high), Some((1, Unit::Zombie)));
        assert_eq!(
            r.next.piece_at(low),
            None,
            "{low} is only adjacent to the unit just spawned on {high}; chaining is illegal"
        );
        // And the rules agree: the freshly spawned unit is not a spawn source.
        let mid = Minions::apply(&s, &MinionsMove::EndPhase);
        let mid = Minions::apply(&mid, &MinionsMove::Buy(Unit::Zombie));
        let mid = Minions::apply(&mid, &MinionsMove::Spawn { unit: Unit::Zombie, at: high });
        assert!(!mid.spawn_cells(1).contains(&low));
    }
}

#[cfg(test)]
mod playout_tests {
    use super::*;
    use rand::SeedableRng;

    /// How often does forced exploration actually propose *buying*?
    ///
    /// Printed, not asserted: it answers whether the injected candidates cover
    /// the economic decision at all, which no aggregate over stored candidates
    /// can (those are already filtered to the visited top-K).
    #[test]
    fn playout_spawn_rate() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        for depth in [4usize, 12, 24] {
            let (mut with_spawn, mut total, mut spawns) = (0u32, 0u32, 0u32);
            for g in 0..40u64 {
                let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(g);
                let mut s = Minions::initial_state();
                for _ in 0..depth {
                    let spec = TurnSpec::random_plausible(&s, &mut r);
                    s = realize(&s, &spec).next;
                    if Minions::outcome(&s).is_terminal() {
                        break;
                    }
                }
                if Minions::outcome(&s).is_terminal() {
                    continue;
                }
                for _ in 0..8 {
                    let spec = TurnSpec::random_playout(&s, &mut rng);
                    let n = spec.spawns.iter().filter(|&&x| x != 0).count();
                    total += 1;
                    if n > 0 {
                        with_spawn += 1;
                        spawns += n as u32;
                    }
                }
            }
            eprintln!(
                "depth {depth:2}: {:.1}% of playouts spawn ({} of {}), {:.2} spawns each",
                100.0 * with_spawn as f64 / total.max(1) as f64,
                with_spawn,
                total,
                spawns as f64 / with_spawn.max(1) as f64
            );
        }
    }

    /// Cost comparison, printed not asserted — this is a hot-path generator and
    /// the number informs whether the coverage is worth its price.
    #[test]
    fn playout_generation_cost() {
        use std::time::Instant;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let mut states = Vec::new();
        for g in 0..8u64 {
            let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(g);
            let mut s = Minions::initial_state();
            for _ in 0..12 {
                let spec = TurnSpec::random_plausible(&s, &mut r);
                s = realize(&s, &spec).next;
                if Minions::outcome(&s).is_terminal() {
                    break;
                }
            }
            states.push(s);
        }
        // Late-game states too: unit count drives both `legal_moves` size and
        // playout length, so an opening-position measurement understates it.
        let mut late = Vec::new();
        for g in 100..108u64 {
            let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(g);
            let mut s = Minions::initial_state();
            for _ in 0..30 {
                let spec = TurnSpec::random_plausible(&s, &mut r);
                s = realize(&s, &spec).next;
                if Minions::outcome(&s).is_terminal() {
                    break;
                }
            }
            if !Minions::outcome(&s).is_terminal() {
                late.push(s);
            }
        }
        let n = 2000;
        let t = Instant::now();
        for i in 0..n {
            std::hint::black_box(TurnSpec::random_plausible(&states[i % states.len()], &mut rng));
        }
        let plaus = t.elapsed().as_nanos() as f64 / n as f64;
        let t = Instant::now();
        for i in 0..n {
            std::hint::black_box(TurnSpec::random_playout(&states[i % states.len()], &mut rng));
        }
        let play = t.elapsed().as_nanos() as f64 / n as f64;
        eprintln!("mid-game:  random_plausible {plaus:.0} ns   random_playout {play:.0} ns   ratio {:.1}x",
                  play / plaus);
        if !late.is_empty() {
            let t = Instant::now();
            for i in 0..n {
                std::hint::black_box(TurnSpec::random_playout(&late[i % late.len()], &mut rng));
            }
            let lp = t.elapsed().as_nanos() as f64 / n as f64;
            let t = Instant::now();
            for i in 0..n {
                std::hint::black_box(TurnSpec::random_plausible(&late[i % late.len()], &mut rng));
            }
            let lpl = t.elapsed().as_nanos() as f64 / n as f64;
            eprintln!("late-game: random_plausible {lpl:.0} ns   random_playout {lp:.0} ns   ratio {:.1}x",
                      lp / lpl);
        }
    }

    /// The point of [`TurnSpec::random_playout`]: a declaration built by playing
    /// the turn is achievable, so the realizer keeps it.
    ///
    /// `random_plausible` declares kill/unsummon per enemy at random and the soak
    /// measures only ~3.8% of those surviving realization, which left injected
    /// candidates effectively move-only and combat uncovered. Playouts measure
    /// ~99.6%. This pins the gap so a regression cannot quietly reopen it.
    #[test]
    fn playout_declarations_survive_realization() {
        let (mut declared, mut done) = (0u64, 0u64);
        let (mut kills_declared, mut kills_done) = (0u64, 0u64);
        // Play whole games rather than sampling early positions: the armies only
        // make contact past the opening, and kills average ~0.14 a turn, so a
        // handful of turn-6 states legitimately sees none.
        for g in 0..12u64 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(g);
            let mut s = Minions::initial_state();
            while !Minions::outcome(&s).is_terminal() {
                let spec = TurnSpec::random_playout(&s, &mut rng);
                let r = realize(&s, &spec);
                let f = r.fidelity;
                declared += f.declared_moves as u64;
                done += f.done_moves as u64;
                kills_declared += f.declared_kills as u64;
                kills_done += f.done_kills as u64;
                s = r.next;
            }
        }
        assert!(declared > 0, "playouts must declare moves");
        let mv = done as f64 / declared as f64;
        assert!(mv > 0.95, "move fidelity {mv:.3} — playouts should be achievable");
        // Combat is the whole reason this generator exists; if no kill was ever
        // declared across 40 games the test is not measuring what it claims.
        assert!(kills_declared > 0, "playouts must sometimes kill");
        let kl = kills_done as f64 / kills_declared as f64;
        assert!(kl > 0.9, "kill fidelity {kl:.3} — far below the ~0.996 measured");
    }
}

/// Timing harness for the forced-exploration playout path (4-7% of self-play).
#[cfg(test)]
mod playout_bench {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn position(plies: usize) -> MinionsState {
        let mut st = MinionsState::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..plies {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = TurnSpec::random_playout(&st, &mut rng);
            st = realize(&st, &spec).next;
        }
        st
    }

    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn realize_cost() {
        for plies in [6usize, 12] {
            let st = position(plies);
            let mut rng = ChaCha8Rng::seed_from_u64(9);
            // Realistic specs: what the sampler actually feeds `realize`.
            let specs: Vec<TurnSpec> =
                (0..200).map(|_| TurnSpec::random_playout(&st, &mut rng)).collect();
            let reps = 2000;
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                std::hint::black_box(realize(&st, &specs[r % specs.len()]).actions.len());
            }
            let realize_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

            // Cost of the state clone that every applied action pays.
            let t1 = std::time::Instant::now();
            for _ in 0..reps * 20 {
                std::hint::black_box(st.clone());
            }
            let clone_ns = t1.elapsed().as_secs_f64() * 1e9 / (reps * 20) as f64;

            let t2 = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(st.spawn_cells(st.side).len());
            }
            let spawn_ns = t2.elapsed().as_secs_f64() * 1e9 / reps as f64;

            println!(
                "plies {plies:2}: realize {realize_us:6.2} us   state clone {clone_ns:5.1} ns                    spawn_cells {spawn_ns:6.1} ns"
            );
        }
    }

    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn playout_cost_breakdown() {
        for plies in [6usize, 12, 18] {
            let st = position(plies);
            let mut rng = ChaCha8Rng::seed_from_u64(5);
            let reps = 2000;

            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(TurnSpec::random_playout(&st, &mut rng));
            }
            let playout_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

            // How much of that is the per-action `legal_moves` enumeration?
            let t1 = std::time::Instant::now();
            let mut n = 0usize;
            for _ in 0..reps {
                n += Minions::legal_moves(&st).len();
            }
            let legal_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;

            // And how much of *that* is the two inner target enumerations?
            let t2 = std::time::Instant::now();
            let mut m = 0usize;
            for _ in 0..reps {
                for i in 0..crate::map::NUM_CELLS as u8 {
                    let Some((s, _)) = st.piece_at(i) else { continue };
                    if s != st.side {
                        continue;
                    }
                    m += st.move_targets(i).len() + st.attack_targets(i).len();
                }
            }
            let targets_us = t2.elapsed().as_secs_f64() * 1e6 / reps as f64;

            println!(
                "plies {plies:2}: playout {playout_us:6.1} us   one legal_moves {legal_us:5.2} us \
                 ({} moves)   its targets {targets_us:5.2} us   [{} {}]",
                n / reps,
                m / reps,
                if legal_us > 0.0 { format!("{:.0}% of legal_moves is targets", targets_us / legal_us * 100.0) } else { String::new() },
            );
        }
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;
    use crate::game::{Minions, MinionsState};
    use playground_core::Game;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Are the per-piece turn-state planes and the phase plane always constant
    /// at the positions the net actually sees?
    ///
    /// The net only ever evaluates *turn boundaries* — `MinionsTurns::apply` is
    /// a whole realized turn — and `end_turn` zeroes every `pstate`. If that
    /// holds, planes 12-15 (damage/moved/attacked/exhausted) and 17 (spawn
    /// phase) carry no information and can go.
    #[test]
    fn turn_boundary_states_have_no_piece_state() {
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let mut checked = 0;
        for game in 0..12u64 {
            let mut st = MinionsState::default();
            let mut rng2 = ChaCha8Rng::seed_from_u64(100 + game);
            while !Minions::outcome(&st).is_terminal() {
                assert!(
                    st.pstate.iter().all(|&p| p == 0),
                    "pstate not clear at a turn boundary: {:?}",
                    st.pstate.iter().enumerate().filter(|(_, &p)| p != 0).collect::<Vec<_>>()
                );
                assert_eq!(
                    st.phase,
                    crate::game::PHASE_ATTACK,
                    "a turn boundary was not in the attack phase"
                );
                checked += 1;
                let spec = TurnSpec::random_playout(&st, &mut rng2);
                st = realize(&st, &spec).next;
            }
            let _ = &mut rng;
        }
        println!("checked {checked} turn-boundary states: pstate all zero, phase always ATTACK");
    }

    /// The energy encoding must be exactly the diffusion one minus plane 2.
    ///
    /// Splitting them created a second Rust/Python surface that can drift; the
    /// validation bundle only covers `write_planes`. Python derives its version
    /// by slicing (`energy_planes` slices `decl_planes`), so pinning the Rust
    /// side to the same relation is what keeps the two languages together.
    #[test]
    fn energy_planes_are_decl_planes_without_the_commit_flag() {
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let mut st = MinionsState::default();
        for _ in 0..8 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = TurnSpec::random_playout(&st, &mut rng);
            let live = LiveMask::of(&st);
            let p = Partial::fully_committed(&spec, &live);

            let mut full = vec![0f32; DECL_PLANES * NUM_CELLS];
            p.write_planes(&mut full);
            let mut lean = vec![0f32; ENERGY_PLANES * NUM_CELLS];
            p.write_energy_planes(&mut lean);

            // Planes 0,1 carry over; plane 2 is dropped; 3.. shift down one.
            let mut want = Vec::with_capacity(ENERGY_PLANES * NUM_CELLS);
            want.extend_from_slice(&full[..2 * NUM_CELLS]);
            want.extend_from_slice(&full[3 * NUM_CELLS..]);
            assert_eq!(lean, want, "energy encoding is not decl-minus-plane-2");

            st = realize(&st, &spec).next;
        }
    }

    #[test]
    fn offsets_are_closed_under_negation() {
        for (i, &(dx, dy)) in OFFSETS.iter().enumerate() {
            let j = NEG_OFFSET[i] as usize;
            assert_eq!(
                (OFFSETS[j].0, OFFSETS[j].1),
                (-dx, -dy),
                "offset {i} = ({dx},{dy}) has no negation"
            );
            assert_eq!(NEG_OFFSET[j] as usize, i, "negation is not self-inverse at {i}");
        }
    }

    #[test]
    /// The kernel's metric must behave like one: zero on the diagonal, symmetric,
    /// and bounded by the sum of its weights.
    #[test]
    fn distance_is_zero_symmetric_and_bounded() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut st = MinionsState::default();
        for _ in 0..8 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let a = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
            let b = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
            assert_eq!(distance(&st, &a, &a), 0.0, "a turn is at distance 0 from itself");
            let ab = distance(&st, &a, &b);
            assert!((ab - distance(&st, &b, &a)).abs() < 1e-6, "distance must be symmetric");
            assert!(
                (0.0..=W_MOVE + W_TARGET + W_SPAWN + 1e-6).contains(&ab),
                "distance {ab} out of range"
            );
            st = realize(&st, &a).next;
        }
    }

    /// The 180-degree automorphism is how both seats share one net, so the
    /// kernel must not see a different geometry after canonicalisation. If it
    /// did, two candidates would be "near" for yellow and "far" for blue and the
    /// search would disagree with itself between seats.
    #[test]
    fn distance_is_invariant_under_the_board_rotation() {
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let mut st = MinionsState::default();
        for _ in 0..8 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let a = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
            let b = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
            let plain = distance(&st, &a, &b);
            let rotated = distance(&st.rotated(), &a.rotated(), &b.rotated());
            assert!(
                (plain - rotated).abs() < 1e-5,
                "distance changed under rotation: {plain} vs {rotated}"
            );
            st = realize(&st, &a).next;
        }
    }

    /// `sigma <= 0` must reproduce a hard delta exactly, because that is the
    /// identity value that keeps the current flat PUCT search available as a
    /// mode rather than being replaced by the kernel one.
    #[test]
    fn zero_bandwidth_kernel_is_a_delta() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let st = MinionsState::default();
        let a = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
        let b = realize(&st, &TurnSpec::random_playout(&st, &mut rng)).spec;
        assert_ne!(a, b, "the fixture needs two distinct turns");
        assert_eq!(kernel(&st, &a, &a, 0.0), 1.0);
        assert_eq!(kernel(&st, &a, &b, 0.0), 0.0);
        // And a positive bandwidth must actually share something between them.
        assert!(kernel(&st, &a, &b, 1.0) > 0.0, "a wide kernel must not be a delta");
    }

    fn canonical_round_trips() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let mut st = MinionsState::default();
        for _ in 0..9 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            // Both seats, so the side==1 branch is exercised.
            assert_eq!(st.rotated().rotated(), st, "rotation must be self-inverse");
            assert_eq!(
                st.canonical().canonical(),
                st.canonical(),
                "canonicalisation must be idempotent"
            );
            assert_eq!(st.canonical().side, 0, "the mover is side 0 in the canonical frame");
            let spec = TurnSpec::random_playout(&st, &mut rng);
            assert_eq!(spec.rotated().rotated(), spec, "rotation must be self-inverse");
            st = realize(&st, &spec).next;
        }
    }

    /// The load-bearing claim: planning in the canonical frame and rotating the
    /// answer back is the same as planning on the real board.
    ///
    /// If this fails the net is being trained on one board and played on another.
    #[test]
    fn planning_canonically_matches_planning_directly() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut st = MinionsState::default();
        for step in 0..10 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = TurnSpec::random_playout(&st, &mut rng);
            let direct = realize(&st, &spec);

            // The same declaration, made in the mover's canonical view.
            let canon = st.canonical();
            let canon_spec = if st.side == 1 { spec.rotated() } else { spec };
            let via = realize(&canon, &canon_spec);

            assert_eq!(
                direct.actions.len(),
                via.actions.len(),
                "step {step}: canonical realization played a different number of actions"
            );
            // The resulting positions must agree once mapped into a common frame.
            assert_eq!(
                Minions::encode_state(&direct.next.canonical()),
                Minions::encode_state(&via.next.canonical()),
                "step {step}: canonical and direct planning diverged"
            );
            st = direct.next;
        }
    }
}
