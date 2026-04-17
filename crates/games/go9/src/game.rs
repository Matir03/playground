//! 9x9 Go game implementation. Stub — fill in at M3.

use playground_core::{Game, Outcome, Player};

#[derive(Clone, Debug, Default)]
pub struct Go9State;

#[derive(Clone, Debug)]
pub struct Go9Move;

pub struct Go9;

impl Game for Go9 {
    type State = Go9State;
    type Move = Go9Move;

    fn id() -> u32 { 2 }
    fn initial_state() -> Self::State { Go9State }
    fn to_move(_: &Self::State) -> Player { Player::First }
    fn outcome(_: &Self::State) -> Outcome { Outcome::InProgress }
    fn legal_moves(_: &Self::State) -> Vec<Self::Move> { vec![] }
    fn apply(s: &Self::State, _: &Self::Move) -> Self::State { s.clone() }
    fn to_features(_: &Self::State) -> Vec<u8> { vec![] }
    fn encode_state(_: &Self::State) -> Vec<u8> { vec![] }
    fn decode_state(_: &[u8]) -> anyhow::Result<Self::State> { Ok(Go9State) }
    fn encode_move(_: &Self::Move) -> Vec<u8> { vec![] }
    fn decode_move(_: &[u8]) -> anyhow::Result<Self::Move> { Ok(Go9Move) }
}
