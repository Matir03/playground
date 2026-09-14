//! The six unit types of the simplified game.
//!
//! Stats are copied verbatim from the reference implementation
//! (`minions/core/src/main/scala/Units.scala`, mirrored in
//! `spooky/src/core/units.rs`): the basic necromancer, the always-acquired
//! `zombie`, and the first four techline cards — `skeleton`, `serpent`, `warg`,
//! `ghost`. Everything past `ghost` is out of scope, and so is the techline
//! itself: all five buyable units are unlocked from move one.
//!
//! One deliberate change from the reference: the warg costs 3 and rebates 1,
//! not 4 and 2.
//!
//! Two stats from the full game are omitted because no unit here uses them:
//! `num_attacks` (every unit below attacks once) and `blink`. `Deathtouch`
//! likewise never appears — the only non-damage attack in the roster is the
//! necromancer's `Unsummon`.

/// What an attack does when it lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attack {
    /// Deal this much damage. Damage accumulates within a turn and is wiped at
    /// end of turn; a piece dies the moment it reaches its defense.
    Damage(u8),
    /// Send the target back to its owner's reinforcements. `persistent` targets
    /// take 1 damage instead.
    Unsummon,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitStats {
    pub name: &'static str,
    /// Uppercase FEN-style letter; lowercase is the second player's.
    pub ch: char,
    pub attack: Attack,
    pub defense: u8,
    /// Hexes of movement per turn.
    pub speed: u8,
    /// Hex distance an attack reaches (no line of sight is required).
    pub range: u8,
    pub cost: u16,
    /// Refunded to the *owner* when the unit dies (not when it is unsummoned).
    pub rebate: u16,
    /// Losing this unit loses the board, and the board is the whole game here.
    pub necromancer: bool,
    /// Cannot move and attack in the same turn.
    pub lumbering: bool,
    /// Moves through enemy units (there is no water on this map, so that is all
    /// flying does).
    pub flying: bool,
    /// Immune to `Unsummon`; takes 1 damage from it instead.
    pub persistent: bool,
    /// New units may be spawned on this unit's empty neighbours.
    pub spawn: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Unit {
    Necromancer = 0,
    Zombie = 1,
    Skeleton = 2,
    Serpent = 3,
    Warg = 4,
    Ghost = 5,
}

pub const NUM_UNITS: usize = 6;

/// The units a player may buy — everything except the necromancer, which is
/// never purchasable and never leaves the board except by dying.
pub const BUYABLE: [Unit; 5] = [Unit::Zombie, Unit::Skeleton, Unit::Serpent, Unit::Warg, Unit::Ghost];

pub const UNIT_STATS: [UnitStats; NUM_UNITS] = [
    UnitStats {
        name: "necromancer", ch: 'N', attack: Attack::Unsummon,
        defense: 7, speed: 1, range: 1, cost: 0, rebate: 0,
        necromancer: true, lumbering: false, flying: false, persistent: true, spawn: true,
    },
    UnitStats {
        name: "zombie", ch: 'Z', attack: Attack::Damage(1),
        defense: 1, speed: 1, range: 1, cost: 2, rebate: 0,
        necromancer: false, lumbering: true, flying: false, persistent: false, spawn: true,
    },
    UnitStats {
        name: "skeleton", ch: 'S', attack: Attack::Damage(5),
        defense: 2, speed: 1, range: 1, cost: 4, rebate: 2,
        necromancer: false, lumbering: false, flying: true, persistent: true, spawn: false,
    },
    UnitStats {
        name: "serpent", ch: 'P', attack: Attack::Damage(3),
        defense: 1, speed: 2, range: 1, cost: 4, rebate: 2,
        necromancer: false, lumbering: false, flying: false, persistent: false, spawn: false,
    },
    UnitStats {
        name: "warg", ch: 'W', attack: Attack::Damage(1),
        defense: 1, speed: 3, range: 1, cost: 3, rebate: 1,
        necromancer: false, lumbering: false, flying: false, persistent: false, spawn: false,
    },
    UnitStats {
        name: "ghost", ch: 'G', attack: Attack::Damage(1),
        defense: 4, speed: 1, range: 1, cost: 3, rebate: 1,
        necromancer: false, lumbering: false, flying: true, persistent: true, spawn: false,
    },
];

impl Unit {
    #[inline]
    pub fn stats(self) -> &'static UnitStats {
        &UNIT_STATS[self as usize]
    }

    #[inline]
    pub fn from_index(i: u8) -> Option<Unit> {
        Some(match i {
            0 => Unit::Necromancer,
            1 => Unit::Zombie,
            2 => Unit::Skeleton,
            3 => Unit::Serpent,
            4 => Unit::Warg,
            5 => Unit::Ghost,
            _ => return None,
        })
    }

    /// Index into a side's reinforcement bag (necromancers are never in one).
    #[inline]
    pub fn reinforcement_slot(self) -> Option<usize> {
        match self {
            Unit::Necromancer => None,
            u => Some(u as usize - 1),
        }
    }
}
