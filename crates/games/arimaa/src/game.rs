//! Arimaa game implementation. Stub — fill in at M6.
//!
//! An Arimaa turn is up to 4 steps, each step itself compound (piece + direction,
//! possibly with a push/pull). This is the ultimate stress test for the
//! reifier + KR-UCT progressive widening approach.

use playground_core::{Game, Outcome, Player};

#[derive(Clone, Debug, Default)]
pub struct ArimaaState;

#[derive(Clone, Debug)]
pub struct ArimaaMove;

pub struct Arimaa;

impl Game for Arimaa {
    type State = ArimaaState;
    type Move = ArimaaMove;

    fn id() -> u32 { 4 }
    fn initial_state() -> Self::State { ArimaaState }
    fn to_move(_: &Self::State) -> Player { Player::First }
    fn outcome(_: &Self::State) -> Outcome { Outcome::InProgress }
    fn legal_moves(_: &Self::State) -> Vec<Self::Move> { vec![] }
    fn apply(s: &Self::State, _: &Self::Move) -> Self::State { s.clone() }
    fn to_features(_: &Self::State) -> Vec<u8> { vec![] }
    fn encode_state(_: &Self::State) -> Vec<u8> { vec![] }
    fn decode_state(_: &[u8]) -> anyhow::Result<Self::State> { Ok(ArimaaState) }
    fn encode_move(_: &Self::Move) -> Vec<u8> { vec![] }
    fn decode_move(_: &[u8]) -> anyhow::Result<Self::Move> { Ok(ArimaaMove) }
}
