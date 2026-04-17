//! Amazons game implementation. Stub — fill in at M5.
//!
//! A move in Amazons is a compound (queen-move, arrow-shot) tuple; this is
//! the first game that exercises the reifier on non-trivial compound actions.

use playground_core::{Game, Outcome, Player};

#[derive(Clone, Debug, Default)]
pub struct AmazonsState;

#[derive(Clone, Debug)]
pub struct AmazonsMove;

pub struct Amazons;

impl Game for Amazons {
    type State = AmazonsState;
    type Move = AmazonsMove;

    fn id() -> u32 { 3 }
    fn initial_state() -> Self::State { AmazonsState }
    fn to_move(_: &Self::State) -> Player { Player::First }
    fn outcome(_: &Self::State) -> Outcome { Outcome::InProgress }
    fn legal_moves(_: &Self::State) -> Vec<Self::Move> { vec![] }
    fn apply(s: &Self::State, _: &Self::Move) -> Self::State { s.clone() }
    fn to_features(_: &Self::State) -> Vec<u8> { vec![] }
    fn encode_state(_: &Self::State) -> Vec<u8> { vec![] }
    fn decode_state(_: &[u8]) -> anyhow::Result<Self::State> { Ok(AmazonsState) }
    fn encode_move(_: &Self::Move) -> Vec<u8> { vec![] }
    fn decode_move(_: &[u8]) -> anyhow::Result<Self::Move> { Ok(AmazonsMove) }
}
