//! Simplified Minions: the board fight from Matir03/minions, on one 10×10 hex
//! board, with the first four techline units permanently unlocked and no
//! techline, no generaling, and no board points.
//!
//! [`game`] has the rules and the scope of the simplification, [`units`] the
//! stat block, [`map`] the hex geometry and the ten graveyards, and [`solver`]
//! the greedy stand-in mover the UI plays against until there is a net.
//!
//! [`turn`] is the flat whole-turn action space the learning stack searches over
//! — a [`TurnSpec`] declares where every unit goes, what happens to every enemy
//! and what is spawned where, and [`realize`] compiles that into a legal
//! sequence of [`MinionsMove`]s. [`bitboard`] is the `u128` hex board it uses.

pub mod assign;
pub mod bitboard;
pub mod eval;
pub mod game;
pub mod genpos;
pub mod map;
pub mod net;
pub mod net_seat;
pub mod nnue;
pub mod ort_net;
pub mod sampler;
pub mod search_config;
pub mod selfplay;
pub mod solver;
pub mod turn;
pub mod units;

pub use game::{Minions, MinionsMove, MinionsState};
pub use turn::{realize, MinionsTurns, Realized, TurnSpec};
pub use units::Unit;
