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
//! It terminates, and no ordering heuristic can deadlock it, because the
//! process is *monotone*:
//!
//! - Within one side's turn friendly units are passable, so the movement mask is
//!   just `!enemy`, shared by every unit of that side. Moving a friendly never
//!   changes another friendly's reachability — only destination emptiness.
//! - Only a kill changes `!enemy`, and only ever by growing it. Kills grow the
//!   empty set too.
//!
//! So no action can un-enable a previously enabled action. Each unit moves at
//! most once and attacks at most once (every unit here has one attack), which
//! bounds the loop at `2n` iterations; it settles in two or three in practice.
//!
//! The realizer is deliberately **best-effort and deterministic, not optimal**.
//! It *decodes* a declaration; it does not *choose* the turn. The reference
//! engine reaches for a SAT solver because it is picking the combat itself —
//! here the policy already picked, and unachievable declarations are simply
//! dropped and reported back through [`Realized::spec`], which is what the nets
//! train on.

use crate::bitboard::{self, bit, Bb, FULL};
use crate::game::{Minions, MinionsMove, MinionsState, PHASE_SPAWN};
use crate::map::{cell_name, dist, NO_CELL, NUM_CELLS};
use crate::units::{Attack, Unit, BUYABLE};
use playground_core::Game;

/// No declared move for this cell.
pub const NO_MOVE: u8 = 255;

/// Hexes within distance 3 — the whole movement space of the fastest unit.
pub const N_OFFSETS: usize = 37;

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

const fn build_move_dest() -> [[u8; N_OFFSETS]; NUM_CELLS] {
    let mut t = [[NO_CELL; N_OFFSETS]; NUM_CELLS];
    let mut c = 0;
    while c < NUM_CELLS {
        let x = (c % 10) as i32;
        let y = (c / 10) as i32;
        let mut k = 0;
        while k < N_OFFSETS {
            let nx = x + OFFSETS[k].0 as i32;
            let ny = y + OFFSETS[k].1 as i32;
            if nx >= 0 && nx < 10 && ny >= 0 && ny < 10 {
                t[c][k] = (ny * 10 + nx) as u8;
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

/// Damage this unit contributes toward a kill. The necromancer's `Unsummon`
/// contributes nothing (it bounces non-persistent units outright and only dents
/// persistent ones), so it never joins a damage pool.
#[inline]
fn kill_damage(unit: Unit) -> u8 {
    match unit.stats().attack {
        Attack::Damage(d) => d,
        Attack::Unsummon => 0,
    }
}

/// Compile `spec` into a legal turn from `state`.
pub fn realize(state: &MinionsState, spec: &TurnSpec) -> Realized {
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
    if s.winner == 0 && s.phase == PHASE_SPAWN {
        for at in 0..NUM_CELLS as u8 {
            let code = spec.spawns[at as usize];
            if code == 0 {
                continue;
            }
            let Some(unit) = Unit::from_index(code) else { continue };
            let Some(slot) = unit.reinforcement_slot() else { continue };
            fid.declared_spawns += 1;
            if !s.spawn_cells(me).contains(&at) {
                continue;
            }
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

    // Kills, most-constrained first.
    loop {
        let mut best: Option<(usize, u8, Vec<u8>)> = None;
        for e in 0..NUM_CELLS as u8 {
            if spec.targets[e as usize] != TARGET_KILL || !is_enemy(s, e, me) {
                continue;
            }
            let attackers = available_attackers(s, e, me, used, want);
            let need = {
                let (_, t) = s.piece_at(e).expect("checked");
                t.stats().defense.saturating_sub(s.damage_on(e))
            };
            let total: u32 = attackers.iter().map(|&a| kill_damage(unit_at(s, a)) as u32).sum();
            if total < need as u32 {
                continue; // not killable this pass
            }
            let n = attackers.len();
            if best.as_ref().is_none_or(|(bn, _, _)| n < *bn) {
                best = Some((n, e, attackers));
            }
        }
        let Some((_, e, attackers)) = best else { break };

        // Spend the fewest units: biggest hitters first, then drop any that the
        // remainder no longer needs.
        let (_, target) = s.piece_at(e).expect("checked");
        let need = target.stats().defense.saturating_sub(s.damage_on(e)) as u32;
        let mut chosen: Vec<u8> = Vec::new();
        let mut acc = 0u32;
        for a in attackers {
            if acc >= need {
                break;
            }
            acc += kill_damage(unit_at(s, a)) as u32;
            chosen.push(a);
        }
        let mut i = 0;
        while i < chosen.len() {
            let d = kill_damage(unit_at(s, chosen[i])) as u32;
            if acc - d >= need {
                acc -= d;
                chosen.remove(i);
            } else {
                i += 1;
            }
        }

        for a in chosen {
            let mv = MinionsMove::Attack { from: a, to: e };
            *s = Minions::apply(s, &mv);
            actions.push(mv);
            used |= bit(a);
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

#[inline]
fn unit_at(s: &MinionsState, c: u8) -> Unit {
    s.piece_at(c).expect("occupied").1
}

fn find_necromancer(s: &MinionsState, side: u8) -> Option<u8> {
    (0..NUM_CELLS as u8)
        .find(|&c| s.piece_at(c).is_some_and(|(sd, u)| sd == side && u.stats().necromancer))
}

/// Our units that could attack `e` right now, best-to-spend first: units with no
/// pending move come first (spending their attack costs nothing, since a unit
/// that attacks can no longer move), then bigger hitters.
fn available_attackers(
    s: &MinionsState,
    e: u8,
    me: u8,
    used: Bb,
    want: &[u8; NUM_CELLS],
) -> Vec<u8> {
    let mut v: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&a| {
            if bitboard::has(used, a) {
                return false;
            }
            let Some((sd, u)) = s.piece_at(a) else { return false };
            sd == me
                && !u.stats().necromancer
                && kill_damage(u) > 0
                && dist(a, e) <= u.stats().range as u32
                && s.can_attack_piece(a, u)
        })
        .collect();
    v.sort_by_key(|&a| {
        let pending = want[a as usize] != NO_CELL;
        (pending, std::cmp::Reverse(kill_damage(unit_at(s, a))), a)
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::idx;
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
        let from = idx(2, 3);
        spec.moves[from as usize] = offset_of(0, 1); // NE, to c4
        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.next.piece_at(idx(2, 4)), Some((0, Unit::Zombie)));
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
        let a = idx(5, 5);
        let b = idx(5, 4);
        s.cells[a as usize] = 1 + Unit::Serpent as u8;
        s.cells[b as usize] = 1 + Unit::Serpent as u8;
        let mut spec = TurnSpec::empty();
        spec.moves[a as usize] = offset_of(0, 1); // f5 -> f6, vacating
        spec.moves[b as usize] = offset_of(0, 1); // f4 -> f5, into the vacated hex

        let r = realize(&s, &spec);
        assert_all_legal(&s, &r);
        assert_eq!(r.fidelity.done_moves, 2, "both moves should land");
        assert_eq!(r.next.piece_at(a), Some((0, Unit::Serpent)));
    }

    #[test]
    fn a_lumbering_unit_that_moves_does_not_also_attack() {
        let mut s = Minions::initial_state();
        let z = idx(5, 4);
        let target = idx(5, 6);
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
