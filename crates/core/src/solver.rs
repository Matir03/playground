//! Built-in (non-NN) move chooser.
//!
//! Lets the engine answer `RequestMove` without a model server — used for
//! strong baselines like the perfect Connect 4 solver. This is deliberately
//! separate from the MCTS+model path: a `Solver` is whatever can pick a move
//! for a position on its own.

use crate::Game;

/// Result of a built-in search.
#[derive(Clone, Debug)]
pub struct SearchOutcome<M> {
    /// The move to play. The caller guarantees `state` was non-terminal.
    pub best_move: M,
    /// Value for the side to move, in [-1, 1]: +1 forced win, 0 draw,
    /// -1 forced loss. Magnitude may encode distance-to-result.
    pub value: f64,
    /// Positions examined, for telemetry / UI.
    pub nodes: u64,
}

/// Per-move budget for a solver, as asked for by `RequestMove`. Built-in
/// solvers may ignore it; external engines usually map it onto their own
/// time/playout controls.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolverBudget {
    pub max_millis: Option<u32>,
    pub max_visits: Option<u32>,
}

/// A game that can choose a move without a neural-network model.
pub trait Solver: Game {
    /// Choose a move for a non-terminal `state`.
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move>;

    /// Choose a move using an *external* reference engine the game crate drives
    /// itself — MoHex over GTP for Hex, and the obvious home for GNU Go in go9.
    /// `None` (the default) means no such engine is configured, and the engine
    /// server turns that into a failed-precondition rather than silently
    /// falling back to [`Solver::choose_move`]: a caller that asked to play the
    /// reference engine must not be answered by a different one.
    fn choose_move_external(
        _state: &Self::State,
        _budget: SolverBudget,
    ) -> Option<anyhow::Result<SearchOutcome<Self::Move>>> {
        None
    }
}

// ── Analysis ──────────────────────────────────────────────────────────────────

/// What the auxiliary heads see at a position, as probabilities.
///
/// Sigmoids are applied here rather than by each consumer, so the UI and any
/// other reader agree on what the numbers mean.
///
/// **Both arrays are indexed `[mover, opponent]`, not `[first, second].`** The
/// training target is built with `sides = [side_to_move, 1 - side_to_move]`, so
/// index 0 follows whoever is to move. Treating index 0 as a fixed player
/// swaps the two sides on every position where the other one moves.
pub struct AnalysisAux {
    /// Per-cell probability that the side holds the hex over the rest of the
    /// game, row-major.
    pub hex_control: [Vec<f32>; 2],
    /// Per-side-turn probability the necromancer is *dead* by that turn.
    ///
    /// Hazard, not survival: the target stores `u8::from(dead)`, so the curve
    /// rises as death becomes more likely.
    pub necro_hazard: [Vec<f32>; 2],
    /// Trajectory-balance log-partition estimate for this position.
    pub log_z: f32,
    /// Predicted terminal `game_score` in points, from the mover's seat: the
    /// net-worth differential plus a bonus for finishing early. Distinct from
    /// `value`, which saturates at +/-1 once the game is decided and so stops
    /// saying *how* decisively.
    pub score: f32,
}

/// One candidate edge of an analysed search tree.
///
/// Moves travel as `Game::encode_move` bytes plus a rendered label, which keeps
/// this type free of the game's move type — the engine server is generic over
/// `Game` and cannot name it.
pub struct AnalysisEdge {
    pub mv: Vec<u8>,
    pub label: String,
    /// Normalised over the node's candidate set.
    pub prior: f32,
    /// Raw `f(s,a)`, before any softmax. Only differences within a node mean
    /// anything: the head's loss is a softmax CE over the candidate set, which
    /// is invariant to a per-state shift, so the absolute level is arbitrary.
    pub energy: f32,
    pub visits: u32,
    /// `None` when never visited — deliberately distinct from a value of 0.
    pub value: Option<f32>,
    pub child: Option<usize>,
    pub explore: bool,
    pub forced: u32,
    /// Sampler draws merged into this candidate (1 = drawn once).
    pub draws: u32,
}

pub struct AnalysisNode {
    pub parent: Option<usize>,
    pub total_visits: u32,
    pub value: Option<f32>,
    pub terminal: bool,
    pub term_value: f32,
    pub edges: Vec<AnalysisEdge>,
    /// Row-major `k x k` KR-PUCT kernel over `edges`, empty when the kernel is
    /// off. `(i, j)` is how much of `j`'s statistics `i` borrows in selection.
    pub kernel: Vec<f32>,
}

/// A position's evaluation, and optionally the tree a search built on it.
pub struct Analysis {
    /// Value head, from the side to move's perspective.
    pub value: f64,
    pub aux: Option<AnalysisAux>,
    /// Empty when only an evaluation was asked for. `nodes[0]` is the root.
    pub nodes: Vec<AnalysisNode>,
    /// Index into `nodes[0].edges` of the move the search would play.
    pub chosen: Option<usize>,
}

/// What to ask an [`Analyzable`] for.
///
/// A struct rather than positional arguments because the list only ever grows,
/// and `analyze(&s, b, None, false, 0.0)` says nothing about which flag is
/// which at the call site.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnalysisOptions {
    /// Candidates per expansion. `None` = the engine's configured search.
    pub candidates: Option<usize>,
    /// Skip the search and return only the position's evaluation.
    pub eval_only: bool,
    /// KR-PUCT kernel bandwidth. `None` = the engine's configured search.
    pub kernel_sigma: Option<f32>,
    /// Run the engine's fast search -- the cheaper one self-play plays most
    /// turns with -- instead of its full one.
    pub fast: bool,
}

/// A game's search settings as JSON, for a client to show and edit field by
/// field without knowing the game.
#[derive(Clone, Debug)]
pub struct SearchConfigJson {
    /// The search in effect.
    pub current: String,
    /// The search the engine started with.
    pub default: String,
}

/// A game that can report what its net and search see at a position.
///
/// Separate from [`Solver`] because analysing is not choosing: the point is to
/// look without playing, and a game with no net has nothing to report. The
/// default returns `None`, which the server surfaces as "not configured"
/// rather than quietly answering with something else.
pub trait Analyzable: crate::Game {
    fn analyze(
        _state: &Self::State,
        _budget: SolverBudget,
        _opts: AnalysisOptions,
    ) -> Option<anyhow::Result<Analysis>> {
        None
    }

    /// Apply one opaque turn encoding, as handed out in [`AnalysisEdge::mv`].
    ///
    /// Needed because a search action is not always a single `Game::Move`: in
    /// Minions it is a whole turn, a *sequence* of moves, so a client cannot
    /// replay a variation through `apply` without reimplementing the realizer.
    fn apply_encoded_turn(_state: &Self::State, _turn: &[u8]) -> anyhow::Result<Self::State> {
        anyhow::bail!("this game has no turn encoding")
    }

    /// The search `analyze` and the engine's seat run. `None` = this game has
    /// no configurable search.
    fn search_config() -> Option<anyhow::Result<SearchConfigJson>> {
        None
    }

    /// Replace that search (`Some(json)`, every setting given) or restore the
    /// one the engine started with (`None`), and report the result as
    /// [`Self::search_config`] does.
    fn set_search_config(_json: Option<&str>) -> Option<anyhow::Result<SearchConfigJson>> {
        None
    }
}
