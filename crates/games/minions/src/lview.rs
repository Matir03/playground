//! The tier-2 head's view of a lattice node (2026-10-06, the user's design).
//!
//! Instead of the steps view's *projected* board (which overwrote an
//! undecided friend when a decided mover claimed its cell, and carried a
//! `next` marker the order-free lattice does not have), a node is the
//! turn's original position plus the decisions so far, side by side:
//!
//! | planes  | what                                                        |
//! |---------|-------------------------------------------------------------|
//! | 0..12   | original units (mover 0..6, opponent 6..12, by `Unit`)      |
//! | 12      | graveyards                                                  |
//! | 13..16  | decided enemy fates: stay, killed, unsummoned (on the enemy) |
//! | 16      | decided friends (on their origin)                           |
//! | 17..23  | final friendly units (by `Unit`), at their final cell: a     |
//! |         | decided move's destination (stays included) or a spawn       |
//! | 23      | the final unit there is a spawn                             |
//!
//! Scalars (the first 13 of the steps layout's 16): money of both sides and
//! reinforcements projected through the decisions (kill rebates, unsummoned
//! units' reinforcements, decided spawns' costs), and the turns left. At a
//! turn start nothing is decided, so planes 0..13 and the 13 scalars are the
//! steps view's position channels: what the trunk reads.
//!
//! Outputs are per cell in separate ranges per kind of variable: fates
//! (`TARGET_*`), moves (offsets), spawns (0 = nothing, else the unit code).
//! [`net_slot`] maps a variable's choice to its slot.

use crate::bitboard::{self, bit, Bb};
use crate::game::MinionsState;
use crate::map::{IS_GRAVEYARD, NUM_CELLS};
use crate::steps::{Stage, StepState, MAX_ACTIVE, N_SCALARS};
use crate::turn::{TurnSpec, MOVE_DEST, N_OFFSETS, TARGET_KILL, TARGET_STAY, TARGET_UNSUMMON};
use crate::units::Unit;

pub const LV_UNITS: usize = 0;
pub const LV_GRAVE: usize = 12;
pub const LV_FATE: usize = 13;
pub const LV_DECIDED: usize = 16;
pub const LV_FINAL: usize = 17;
pub const LV_SPAWNED: usize = 23;
pub const LV_PLANES: usize = 24;
pub const LV_SCALARS: usize = 13;

pub const FATE_SLOTS: usize = 3;
pub const MOVE_SLOTS: usize = N_OFFSETS;
pub const SPAWN_SLOTS: usize = 6;
pub const FATE_OFF: usize = 0;
pub const MOVE_OFF: usize = FATE_SLOTS;
pub const SPAWN_OFF: usize = FATE_SLOTS + MOVE_SLOTS;
/// Output slots per cell.
pub const NET_SLOTS: usize = FATE_SLOTS + MOVE_SLOTS + SPAWN_SLOTS;

/// The output slot of `choice` for a variable decided at stage `stage`.
pub fn net_slot(stage: Stage, choice: u8) -> u8 {
    let off = match stage {
        Stage::Fates => FATE_OFF,
        Stage::Moves => MOVE_OFF,
        Stage::Spawns | Stage::Terminal => SPAWN_OFF,
    };
    (off + choice as usize) as u8
}

/// The decisions of a (partial) turn: which enemies are fated, which
/// friends' moves and which spawn cells are decided; choices from `spec`.
struct Decided<'a> {
    spec: &'a TurnSpec,
    fated: Bb,
    movers: Bb,
    spawns: Bb,
}

fn decided_of(s: &StepState) -> Decided<'_> {
    let all = |v: &[u8]| v.iter().fold(0, |b, &c| b | bit(c));
    let (fated, movers, spawns) = match s.stage {
        Stage::Fates => (all(&s.enemies[..s.idx]), 0, 0),
        Stage::Moves => (all(&s.enemies), all(&s.friends[..s.idx]), 0),
        Stage::Spawns => (
            all(&s.enemies),
            all(&s.friends),
            all(&s.spawn_cells[..s.idx]),
        ),
        Stage::Terminal => (all(&s.enemies), all(&s.friends), all(&s.spawn_cells)),
    };
    Decided {
        spec: &s.spec,
        fated,
        movers,
        spawns,
    }
}

fn encode(base: &MinionsState, d: &Decided, act: &mut Vec<u16>, sc: &mut [f32; N_SCALARS]) {
    act.clear();
    let mut push = |p: usize, c: u8| act.push((p * NUM_CELLS + c as usize) as u16);
    for c in 0..NUM_CELLS as u8 {
        if let Some((side, u)) = base.piece_at(c) {
            push(LV_UNITS + side as usize * 6 + u as usize, c);
        }
        if IS_GRAVEYARD[c as usize] {
            push(LV_GRAVE, c);
        }
    }
    let mut money = base.money;
    let mut reinf = base.reinforcements;
    for e in bitboard::cells(d.fated) {
        let t = d.spec.targets[e as usize];
        let plane = match t {
            TARGET_STAY => 0,
            TARGET_KILL => 1,
            TARGET_UNSUMMON => 2,
            _ => continue,
        };
        push(LV_FATE + plane, e);
        if let Some((side, u)) = base.piece_at(e) {
            if t == TARGET_KILL {
                money[side as usize] = money[side as usize].saturating_add(u.stats().rebate);
            } else if t == TARGET_UNSUMMON {
                if let Some(slot) = u.reinforcement_slot() {
                    reinf[side as usize][slot] = reinf[side as usize][slot].saturating_add(1);
                }
            }
        }
    }
    for f in bitboard::cells(d.movers) {
        push(LV_DECIDED, f);
        let k = d.spec.moves[f as usize];
        let dest = MOVE_DEST[f as usize][k as usize];
        if let (Some((_, u)), true) = (base.piece_at(f), (dest as usize) < NUM_CELLS) {
            push(LV_FINAL + u as usize, dest);
        }
    }
    for c in bitboard::cells(d.spawns) {
        let code = d.spec.spawns[c as usize];
        let Some(u) = (code != 0).then(|| Unit::from_index(code)).flatten() else {
            continue;
        };
        push(LV_FINAL + u as usize, c);
        push(LV_SPAWNED, c);
        // A spawn takes a reinforcement, or buys one first.
        if let Some(slot) = u.reinforcement_slot() {
            if reinf[0][slot] > 0 {
                reinf[0][slot] -= 1;
            } else {
                money[0] = money[0].saturating_sub(u.stats().cost);
            }
        }
    }
    act.truncate(MAX_ACTIVE);
    sc.fill(0.0);
    sc[0] = f32::from(money[0]) / 10.0;
    sc[1] = f32::from(money[1]) / 10.0;
    for (side, r) in reinf.iter().enumerate() {
        for (slot, &n) in r.iter().enumerate() {
            sc[2 + side * 5 + slot] = f32::from(n);
        }
    }
    sc[12] = crate::game::MAX_TURNS.saturating_sub(base.turn) as f32 / 30.0;
}

/// A lattice node's features (a turn start included: nothing decided).
pub fn node_features(s: &StepState, act: &mut Vec<u16>, sc: &mut [f32; N_SCALARS]) {
    encode(&s.base, &decided_of(s), act, sc);
    // In the spawn phase the realized position is known: exact money and
    // reinforcements (the spawns so far applied).
    if let (Stage::Spawns, Some(p)) = (s.stage, s.post.as_ref()) {
        sc[0] = f32::from(p.money[0]) / 10.0;
        sc[1] = f32::from(p.money[1]) / 10.0;
        for (side, r) in p.reinforcements.iter().enumerate() {
            for (slot, &n) in r.iter().enumerate() {
                sc[2 + side * 5 + slot] = f32::from(n);
            }
        }
    }
}

/// The features of a complete declaration `spec` of the turn starting at
/// `start` (every variable decided): what the energy is read from.
pub fn decl_features(
    start: &StepState,
    spec: &TurnSpec,
    act: &mut Vec<u16>,
    sc: &mut [f32; N_SCALARS],
) {
    let all = |v: &[u8]| v.iter().fold(0, |b, &c| b | bit(c));
    let mut spawns: Bb = 0;
    for c in 0..NUM_CELLS as u8 {
        if spec.spawns[c as usize] != 0 {
            spawns |= bit(c);
        }
    }
    let d = Decided {
        spec,
        fated: all(&start.enemies),
        movers: all(&start.friends),
        spawns,
    };
    encode(&start.base, &d, act, sc);
}

/// The turn starting at `start` with every variable decided as in `spec`
/// (a complete declaration): its [`node_features`] are [`decl_features`],
/// so an expansion request on it (no kids) reads the declaration's energy.
pub fn decl_state(start: &StepState, spec: &TurnSpec) -> StepState {
    let mut s = start.clone();
    s.spec = *spec;
    s.spawn_cells = (0..NUM_CELLS as u8)
        .filter(|&c| spec.spawns[c as usize] != 0)
        .collect();
    s.stage = Stage::Terminal;
    s.idx = 0;
    s.post = None;
    s
}

/// Sparse features as a record's bytes (`MAX_ACTIVE` u16 indices, then the
/// scalars as f32): the active count.
pub fn push_features(act: &[u16], sc: &[f32; N_SCALARS], out: &mut Vec<u8>) -> u8 {
    for i in 0..MAX_ACTIVE {
        out.extend_from_slice(&act.get(i).copied().unwrap_or(0).to_le_bytes());
    }
    for x in sc {
        out.extend_from_slice(&x.to_le_bytes());
    }
    act.len().min(MAX_ACTIVE) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::MinionsState;

    /// At a turn start the lattice view's position channels are the steps
    /// view's (what the trunk reads), and nothing is decided.
    #[test]
    fn turn_start_matches_steps_position_channels() {
        for bid in [0, 2] {
            let s = StepState::begin(&MinionsState::with_bid(bid), 3);
            let (mut a, mut sa) = (Vec::new(), [0f32; N_SCALARS]);
            node_features(&s, &mut a, &mut sa);
            let (mut b, mut sb) = (Vec::new(), [0f32; N_SCALARS]);
            s.view().features(&mut b, &mut sb);
            let pos = |v: &[u16]| {
                let mut x: Vec<u16> = v
                    .iter()
                    .copied()
                    .filter(|&i| (i as usize) < 13 * NUM_CELLS)
                    .collect();
                x.sort();
                x
            };
            assert_eq!(pos(&a), pos(&b));
            assert!(a.iter().all(|&i| (i as usize) < 13 * NUM_CELLS));
            assert_eq!(&sa[..LV_SCALARS], &sb[..LV_SCALARS]);
        }
    }
}
