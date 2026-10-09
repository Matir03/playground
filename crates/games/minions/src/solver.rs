//! A greedy built-in mover.
//!
//! There is no Minions net yet, so `RequestMove` with the SOLVER seat lands
//! here. This is a one-ply heuristic, not a search: it scores every legal
//! action in the current position and plays the best one, and because a turn is
//! a sequence of actions it gets called again for each. That is enough to give
//! the web UI an opponent that takes free kills, walks toward graveyards and
//! spends its money, which is all a placeholder needs to be.
//!
//! Deliberate quirks, so its play is not mistaken for a bug: it never moves its
//! necromancer, and it only attacks with the necromancer when the unsummon
//! actually removes something.

use playground_core::{Game, SearchOutcome};

use crate::game::{Minions, MinionsMove, MinionsState, PHASE_ATTACK};
use crate::map::{dist, IS_GRAVEYARD, NECRO_START, NUM_CELLS};
use crate::units::{Attack, Unit};

/// Rough worth of a unit, used both for choosing targets and for the value the
/// solver reports back to the UI.
fn unit_value(u: Unit) -> i32 {
    match u {
        Unit::Necromancer => 1000,
        Unit::Skeleton => 40,
        Unit::Serpent => 30,
        Unit::Ghost => 30,
        Unit::Warg => 25,
        Unit::Zombie => 10,
    }
}

/// Distance from `cell` to the nearest thing worth walking to: a graveyard we
/// do not already hold, or the enemy necromancer.
fn objective_distance(state: &MinionsState, side: u8, cell: u8) -> u32 {
    let mut best = u32::MAX;
    for i in 0..NUM_CELLS as u8 {
        let is_target = if IS_GRAVEYARD[i as usize] {
            state.piece_at(i).is_none_or(|(s, _)| s != side)
        } else {
            state
                .piece_at(i)
                .is_some_and(|(s, u)| s != side && u.stats().necromancer)
        };
        if is_target {
            best = best.min(dist(cell, i));
        }
    }
    best
}

/// What an attack from `from` onto `to` would do: `(damage, bounces, kills)`.
fn attack_effect(state: &MinionsState, from: u8, to: u8) -> (u8, bool, bool) {
    let (_, att) = state.piece_at(from).expect("attacker exists");
    let (_, def) = state.piece_at(to).expect("target exists");
    let def_stats = def.stats();
    let (damage, bounce) = match att.stats().attack {
        Attack::Damage(d) => (d, false),
        Attack::Unsummon => {
            if def_stats.persistent {
                (1, false)
            } else {
                (0, true)
            }
        }
    };
    let kills = !bounce && state.damage_on(to) + damage >= def_stats.defense;
    (damage, bounce, kills)
}

fn score(state: &MinionsState, mv: &MinionsMove) -> i32 {
    let side = state.side;
    match *mv {
        MinionsMove::EndPhase => 0,

        // The greedy bot never cycles (its scores are per unit), so it plays
        // as it did before cyclic moves existed.
        MinionsMove::Cycle { .. } => -1,

        MinionsMove::Attack { from, to } => {
            let (_, att) = state.piece_at(from).expect("attacker exists");
            let (_, def) = state.piece_at(to).expect("target exists");
            let (damage, bounce, kills) = attack_effect(state, from, to);
            if def.stats().necromancer && kills {
                return 100_000;
            }
            if kills {
                return 200 + unit_value(def);
            }
            if bounce {
                return 150 + unit_value(def);
            }
            // Chip damage only, and never worth exposing the necromancer for.
            if att.stats().necromancer {
                -1
            } else {
                30 + damage as i32 * 10
            }
        }

        MinionsMove::Move { from, to } => {
            let (_, unit) = state.piece_at(from).expect("mover exists");
            if unit.stats().necromancer {
                return -1; // the necromancer stays home
            }
            let gain = objective_distance(state, side, from) as i32
                - objective_distance(state, side, to) as i32;
            let mut s = 5 + gain * 8;
            if IS_GRAVEYARD[to as usize] {
                s += 15;
            }
            // Stepping off a graveyard we hold costs income; only do it if the
            // move buys something better.
            if IS_GRAVEYARD[from as usize] {
                s -= 15;
            }
            s
        }

        MinionsMove::Buy(unit) => 40 + unit_value(unit) / 5,

        MinionsMove::Spawn { unit, at } => {
            let mut s = 60 + unit_value(unit) / 10;
            if IS_GRAVEYARD[at as usize] {
                s += 10;
            }
            // Break ties toward the enemy, so spawns do not pile up behind.
            let enemy_necro = NECRO_START[1 - side as usize];
            s += 9i32.saturating_sub(dist(at, enemy_necro) as i32);
            s
        }
    }
}

/// Material + territory + treasury, from the side to move's point of view,
/// squashed into `[-1, 1]`.
fn evaluate(state: &MinionsState) -> f64 {
    let side = state.side;
    let mut material = 0i32;
    for i in 0..NUM_CELLS as u8 {
        if let Some((s, u)) = state.piece_at(i) {
            if u.stats().necromancer {
                continue; // both are alive, or the game is already over
            }
            material += if s == side {
                unit_value(u)
            } else {
                -unit_value(u)
            };
        }
    }
    let territory =
        5 * (state.graveyards_held(side) as i32 - state.graveyards_held(1 - side) as i32);
    let treasury = (state.money[side as usize] as i32 - state.money[1 - side as usize] as i32) / 2;
    ((material + territory + treasury) as f64 / 200.0).tanh()
}

pub fn greedy_move(state: &MinionsState) -> SearchOutcome<MinionsMove> {
    let legal = Minions::legal_moves(state);
    assert!(
        !legal.is_empty(),
        "greedy_move called on a terminal position"
    );

    // Buying only makes sense if there is somewhere to put the unit; without
    // this the bot would bank money it can never spend.
    let spawn_room = if state.phase == PHASE_ATTACK {
        0
    } else {
        let held: u32 = state.reinforcements[state.side as usize]
            .iter()
            .map(|&n| n as u32)
            .sum();
        (state.spawn_cells(state.side).len() as u32).saturating_sub(held)
    };

    let mut best = legal[0];
    let mut best_score = i32::MIN;
    for mv in &legal {
        if matches!(mv, MinionsMove::Buy(_)) && spawn_room == 0 {
            continue;
        }
        let s = score(state, mv);
        if s > best_score {
            best_score = s;
            best = *mv;
        }
    }

    SearchOutcome {
        best_move: best,
        value: evaluate(state),
        nodes: legal.len() as u64,
    }
}
