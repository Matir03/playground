//! Hex 13×13: rules ([`game`]), the 169-bit board ([`bitboard`]) and the MoHex
//! bridge ([`mohex`]).
//!
//! A library as well as a binary because the AZ pieces that come next (net,
//! search, self-play, and any eval tooling) all need the rules and the
//! symmetry/plane helpers.

pub mod bitboard;
pub mod game;
pub mod mohex;
