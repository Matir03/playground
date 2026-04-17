//! Connect 4 game implementation. Stub — fill in at M1.

use playground_core::{Game, Outcome, Player};

#[derive(Clone, Debug, Default)]
pub struct Connect4State;

#[derive(Clone, Debug)]
pub struct Connect4Move;

pub struct Connect4;

impl Game for Connect4 {
    type State = Connect4State;
    type Move = Connect4Move;

    fn id() -> u32 {
        // GameId::GAME_ID_CONNECT4
        1
    }

    fn initial_state() -> Self::State {
        Connect4State
    }

    fn to_move(_state: &Self::State) -> Player {
        Player::First
    }

    fn outcome(_state: &Self::State) -> Outcome {
        Outcome::InProgress
    }

    fn legal_moves(_state: &Self::State) -> Vec<Self::Move> {
        vec![]
    }

    fn apply(state: &Self::State, _mv: &Self::Move) -> Self::State {
        state.clone()
    }

    fn to_features(_state: &Self::State) -> Vec<u8> {
        vec![]
    }

    fn encode_state(_state: &Self::State) -> Vec<u8> {
        vec![]
    }

    fn decode_state(_bytes: &[u8]) -> anyhow::Result<Self::State> {
        Ok(Connect4State)
    }

    fn encode_move(_mv: &Self::Move) -> Vec<u8> {
        vec![]
    }

    fn decode_move(_bytes: &[u8]) -> anyhow::Result<Self::Move> {
        Ok(Connect4Move)
    }
}
