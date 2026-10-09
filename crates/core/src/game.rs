//! Game trait — the contract every concrete game implements.

use std::fmt::Debug;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Player {
    First,
    Second,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Player::First => Player::Second,
            Player::Second => Player::First,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Outcome {
    InProgress,
    FirstWins,
    SecondWins,
    Draw,
}

impl Outcome {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Outcome::InProgress)
    }

    /// Value from the perspective of `player`, in [-1, 1].
    pub fn value_for(self, player: Player) -> f64 {
        match (self, player) {
            (Outcome::FirstWins, Player::First) => 1.0,
            (Outcome::FirstWins, Player::Second) => -1.0,
            (Outcome::SecondWins, Player::Second) => 1.0,
            (Outcome::SecondWins, Player::First) => -1.0,
            (Outcome::Draw, _) => 0.0,
            (Outcome::InProgress, _) => 0.0,
        }
    }
}

/// A two-player abstract strategy game.
///
/// Implementations must be deterministic: applying the same `Move` to two
/// equal `State`s always yields the same next state.
pub trait Game: Send + Sync + 'static {
    type State: Clone + Debug + Send + Sync;
    type Move: Clone + Debug + Send + Sync;

    /// Stable identifier (matches `playground.game.GameId`).
    fn id() -> u32;

    fn initial_state() -> Self::State;

    fn to_move(state: &Self::State) -> Player;
    fn outcome(state: &Self::State) -> Outcome;

    fn legal_moves(state: &Self::State) -> Vec<Self::Move>;

    /// Whether `mv` is legal in `state`. The default looks it up in
    /// [`Self::legal_moves`]; a game whose legal set is too large to list
    /// in full (Minions' cyclic moves) overrides it with a direct check.
    fn is_legal(state: &Self::State, mv: &Self::Move) -> bool {
        let want = Self::encode_move(mv);
        Self::legal_moves(state).iter().any(|m| Self::encode_move(m) == want)
    }
    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State;

    /// Monotonic ply counter (matches `GameState.ply` in the proto).
    fn ply(state: &Self::State) -> u32;

    /// Feature bytes for the encoder input. Layout is game-specific and
    /// must match what the Python model server expects.
    fn to_features(state: &Self::State) -> Vec<u8>;

    /// Serialize state for wire transport (`GameState.payload` bytes).
    fn encode_state(state: &Self::State) -> Vec<u8>;
    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State>;

    /// Serialize a move for wire transport (`Move.payload` bytes).
    fn encode_move(mv: &Self::Move) -> Vec<u8>;
    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move>;
}
