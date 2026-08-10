//! Hex 13×13: rules ([`game`]), the 169-bit board ([`bitboard`]) and the MoHex
//! bridge ([`mohex`]).
//!
//! Plus the plain-AlphaZero stack built on them: the net ([`net`]), the PUCT
//! search ([`mcts`]), self-play generation ([`selfplay`]) and the two ways to
//! rate a checkpoint ([`eval`]).

pub mod bitboard;
pub mod eval;
pub mod game;
pub mod mcts;
pub mod mohex;
pub mod net;
pub mod ort_net;
pub mod selfplay;
