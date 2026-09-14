//! tonic server implementing `EngineService` generically over a `Game`.
//!
//! Implements the M1 interactive-play RPCs (NewGame / GetState / ApplyMove).
//! RequestMove, PlayMatch, and StartSelfPlay are stubbed for M2.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::sync::Arc;

use playground_core::{Analyzable, Game, ModelClient, Outcome, Player, Solver, SolverBudget};
use playground_protocol::engine_pb::engine_service_server::{EngineService, EngineServiceServer};
use playground_protocol::engine_pb::*;
use playground_protocol::game_pb::{self, GameHandle, GameState};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

// ── Game registry ─────────────────────────────────────────────────────────────

type Registry = Arc<RwLock<HashMap<String, Vec<u8>>>>;

fn new_handle_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // Simple non-crypto ID: combine coarse time with a thread-local counter.
    format!(
        "{:08x}{:08x}",
        t,
        HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

static HANDLE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// ── Conversions ───────────────────────────────────────────────────────────────

fn player_to_proto(p: Player) -> i32 {
    match p {
        Player::First => game_pb::Player::First as i32,
        Player::Second => game_pb::Player::Second as i32,
    }
}

fn outcome_to_proto(o: Outcome) -> i32 {
    match o {
        Outcome::InProgress => game_pb::Outcome::InProgress as i32,
        Outcome::FirstWins => game_pb::Outcome::FirstWins as i32,
        Outcome::SecondWins => game_pb::Outcome::SecondWins as i32,
        Outcome::Draw => game_pb::Outcome::Draw as i32,
    }
}

fn state_to_proto<G: Game>(state: &G::State) -> GameState {
    GameState {
        game_id: G::id() as i32,
        payload: G::encode_state(state).into(),
        to_move: player_to_proto(G::to_move(state)),
        outcome: outcome_to_proto(G::outcome(state)),
        ply: G::ply(state),
    }
}

// ── Server struct ─────────────────────────────────────────────────────────────

pub struct EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    /// NN models by name (e.g. "random", "trained"). The NN player picks one
    /// per RequestMove; empty name = the first inserted.
    models: HashMap<String, Arc<M>>,
    games: Registry,
    _game: PhantomData<G>,
}

impl<G, M> EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    pub fn new(models: HashMap<String, M>) -> Self {
        Self {
            models: models.into_iter().map(|(k, v)| (k, Arc::new(v))).collect(),
            games: Arc::new(RwLock::new(HashMap::new())),
            _game: PhantomData,
        }
    }

    /// Resolve a model by name; empty name falls back to any configured model.
    fn model_for(&self, name: &str) -> Result<&Arc<M>, Status> {
        let chosen = if name.is_empty() {
            self.models.values().next()
        } else {
            self.models.get(name)
        };
        chosen.ok_or_else(|| {
            let mut names: Vec<&str> = self.models.keys().map(String::as_str).collect();
            names.sort_unstable();
            Status::failed_precondition(format!(
                "no NN model {:?} configured (available: {:?})",
                name, names
            ))
        })
    }
}

/// Result of picking a move: serialized move payload, value for the side to
/// move in [-1, 1], a node/visit count, and a reification count (NN only).
struct Chosen {
    payload: Vec<u8>,
    value: f64,
    nodes: u64,
    reifications: u32,
}

/// Neural-net move: LGS-space MCTS with on-demand reification, using the model's
/// encode/eval/policy/reify heads (see playground_core::mcts and
/// docs/algorithm.md). With random weights it plays weakly, but exercises the
/// full planned search path rather than a 1-ply lookahead.
async fn nn_choose_move<G, M>(
    model: &M,
    state: &G::State,
    budget: Option<&SearchBudget>,
) -> anyhow::Result<Chosen>
where
    G: Game,
    M: ModelClient<G>,
{
    let mut config = playground_core::mcts::SearchConfig::default();
    // A time budget takes precedence over a visit budget.
    if let Some(ms) = budget.and_then(|b| b.max_millis).filter(|&v| v > 0) {
        config.move_millis = ms as u64;
    } else if let Some(v) = budget.and_then(|b| b.max_visits).filter(|&v| v > 0) {
        config.simulations = v;
    }
    let result = playground_core::mcts::search::<G, M>(model, state, &config).await?;
    Ok(Chosen {
        payload: G::encode_move(&result.chosen),
        value: result.root_value,
        nodes: result.visits as u64,
        reifications: result.reifications,
    })
}

// ── RPC implementations ───────────────────────────────────────────────────────

#[tonic::async_trait]
impl<G, M> EngineService for EngineServer<G, M>
where
    G: Solver + Analyzable,
    M: ModelClient<G> + 'static,
{
    /// Commit one whole turn to the handle's game.
    ///
    /// The counterpart to `analyze`: that one looks at a line on a copy, this
    /// one plays it. Both take the same opaque encoding, so an analysis view
    /// can hand back the turn it was just shown without translating anything.
    ///
    /// Legality is enforced by the game's realizer rather than by checking
    /// against `legal_moves` the way `apply_move` does: a turn is a *sequence*
    /// of moves, so it has no entry in that list to match.
    async fn apply_turn(
        &self,
        req: Request<ApplyTurnRequest>,
    ) -> Result<Response<GameState>, Status> {
        let inner = req.into_inner();
        let handle = inner
            .handle
            .ok_or_else(|| Status::invalid_argument("apply_turn: no handle"))?;
        let handle_id = handle.handle_id.clone();

        let mut games = self.games.write().await;
        let bytes = games
            .get(&handle_id)
            .ok_or_else(|| Status::not_found(format!("unknown handle {handle_id}")))?
            .clone();

        let state = G::decode_state(&bytes).map_err(|e| Status::internal(e.to_string()))?;
        if G::outcome(&state).is_terminal() {
            return Err(Status::failed_precondition("game is already over"));
        }

        let next = G::apply_encoded_turn(&state, &inner.turn)
            .map_err(|e| Status::invalid_argument(format!("apply_turn: {e}")))?;
        games.insert(handle_id, G::encode_state(&next));

        Ok(Response::new(state_to_proto::<G>(&next)))
    }

    /// The search analysis and the seat run. See `Analyzable::search_config`.
    async fn get_search_config(
        &self,
        _req: Request<GetSearchConfigRequest>,
    ) -> Result<Response<SearchConfig>, Status> {
        let c = G::search_config()
            .ok_or_else(|| Status::failed_precondition("this engine has no configurable search"))?
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(SearchConfig {
            json: c.current,
            default_json: c.default,
        }))
    }

    /// Replace the search, or restore the one the engine started with. A
    /// setting the game refuses is the caller's error, so it comes back as an
    /// invalid argument naming the setting.
    async fn set_search_config(
        &self,
        req: Request<SetSearchConfigRequest>,
    ) -> Result<Response<SearchConfig>, Status> {
        let inner = req.into_inner();
        let json = (!inner.reset).then_some(inner.json.as_str());
        let c = G::set_search_config(json)
            .ok_or_else(|| Status::failed_precondition("this engine has no configurable search"))?
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        Ok(Response::new(SearchConfig {
            json: c.current,
            default_json: c.default,
        }))
    }

    /// Run the net and the search on a position and report what they saw.
    ///
    /// Deliberately does not touch the handle's stored state: the point of an
    /// analysis view is to look at a line without committing to it, so the
    /// requested `line` is applied to a *copy* and thrown away afterwards.
    async fn analyze(
        &self,
        req: Request<AnalyzeRequest>,
    ) -> Result<Response<AnalyzeResponse>, Status> {
        let inner = req.into_inner();
        let handle = inner
            .handle
            .ok_or_else(|| Status::invalid_argument("analyze: no handle"))?;

        let mut state = {
            let games = self.games.read().await;
            let bytes = games
                .get(&handle.handle_id)
                .ok_or_else(|| Status::not_found("analyze: unknown handle"))?;
            G::decode_state(bytes).map_err(|e| Status::internal(e.to_string()))?
        };

        // Walk the requested variation on the copy. A line that does not apply
        // is the client's bug, and reporting an analysis of a *different*
        // position than it asked about would be worse than refusing.
        for (i, turn) in inner.line.iter().enumerate() {
            state = G::apply_encoded_turn(&state, turn).map_err(|e| {
                Status::invalid_argument(format!("analyze: line[{i}] does not apply: {e}"))
            })?;
        }

        let budget = SolverBudget {
            max_visits: inner.budget.as_ref().and_then(|b| b.max_visits),
            max_millis: inner.budget.as_ref().and_then(|b| b.max_millis),
        };
        let opts = playground_core::AnalysisOptions {
            candidates: (inner.candidates > 0).then_some(inner.candidates as usize),
            eval_only: inner.eval_only,
            kernel_sigma: (inner.kernel_sigma > 0.0).then_some(inner.kernel_sigma),
            fast: inner.fast,
        };

        let analysis = G::analyze(&state, budget, opts)
            .ok_or_else(|| {
                Status::failed_precondition("analyze: this engine has no analysable net configured")
            })?
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(AnalyzeResponse {
            state: Some(state_to_proto::<G>(&state)),
            value: analysis.value,
            aux: analysis.aux.map(|a| AnalyzeAux {
                hex_control_mover: a.hex_control[0].iter().map(|x| *x as f64).collect(),
                hex_control_opponent: a.hex_control[1].iter().map(|x| *x as f64).collect(),
                necro_hazard_mover: a.necro_hazard[0].iter().map(|x| *x as f64).collect(),
                necro_hazard_opponent: a.necro_hazard[1].iter().map(|x| *x as f64).collect(),
                log_z: a.log_z as f64,
                score: a.score as f64,
            }),
            nodes: analysis
                .nodes
                .iter()
                .map(|n| AnalyzeNode {
                    parent: n.parent.map(|p| p as u32),
                    total_visits: n.total_visits,
                    value: n.value.map(|v| v as f64),
                    terminal: n.terminal,
                    term_value: n.term_value as f64,
                    edges: n
                        .edges
                        .iter()
                        .map(|e| AnalyzeEdge {
                            turn: e.mv.clone().into(),
                            label: e.label.clone(),
                            prior: e.prior as f64,
                            energy: e.energy as f64,
                            visits: e.visits,
                            value: e.value.map(|v| v as f64),
                            child: e.child.map(|c| c as u32),
                            explore: e.explore,
                            forced: e.forced,
                            draws: e.draws,
                        })
                        .collect(),
                    kernel: n.kernel.clone(),
                })
                .collect(),
            chosen: analysis.chosen.map(|c| c as u32),
        }))
    }

    async fn new_game(&self, req: Request<NewGameRequest>) -> Result<Response<GameHandle>, Status> {
        let inner = req.into_inner();
        let state = match inner.initial_state {
            Some(lgs) => G::decode_state(&lgs.payload)
                .map_err(|e| Status::invalid_argument(e.to_string()))?,
            None => G::initial_state(),
        };
        let id = new_handle_id();
        self.games
            .write()
            .await
            .insert(id.clone(), G::encode_state(&state));
        Ok(Response::new(GameHandle {
            handle_id: id,
            game_id: G::id() as i32,
        }))
    }

    async fn get_state(&self, req: Request<GameHandle>) -> Result<Response<GameState>, Status> {
        let handle_id = req.into_inner().handle_id;
        let games = self.games.read().await;
        let bytes = games
            .get(&handle_id)
            .ok_or_else(|| Status::not_found(format!("unknown handle {handle_id}")))?;
        let state = G::decode_state(bytes).map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(state_to_proto::<G>(&state)))
    }

    async fn apply_move(
        &self,
        req: Request<ApplyMoveRequest>,
    ) -> Result<Response<GameState>, Status> {
        let inner = req.into_inner();
        let handle = inner
            .handle
            .ok_or_else(|| Status::invalid_argument("missing handle"))?;
        let mv_proto = inner
            .r#move
            .ok_or_else(|| Status::invalid_argument("missing move"))?;

        let handle_id = handle.handle_id.clone();
        let mut games = self.games.write().await;
        let bytes = games
            .get(&handle_id)
            .ok_or_else(|| Status::not_found(format!("unknown handle {handle_id}")))?
            .clone();

        let state = G::decode_state(&bytes).map_err(|e| Status::internal(e.to_string()))?;
        if G::outcome(&state).is_terminal() {
            return Err(Status::failed_precondition("game is already over"));
        }

        let mv = G::decode_move(&mv_proto.payload)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        // Validate legality.
        let legal = G::legal_moves(&state);
        let is_legal = legal
            .iter()
            .any(|m| G::encode_move(m) == G::encode_move(&mv));
        if !is_legal {
            return Err(Status::invalid_argument("move is not legal"));
        }

        let next = G::apply(&state, &mv);
        let next_bytes = G::encode_state(&next);
        games.insert(handle_id, next_bytes);

        Ok(Response::new(state_to_proto::<G>(&next)))
    }

    type RequestMoveStream = tokio_stream::wrappers::ReceiverStream<Result<MoveProgress, Status>>;
    async fn request_move(
        &self,
        req: Request<RequestMoveRequest>,
    ) -> Result<Response<Self::RequestMoveStream>, Status> {
        let inner = req.into_inner();
        let player = inner.player();
        let handle = inner
            .handle
            .ok_or_else(|| Status::invalid_argument("missing handle"))?;
        let handle_id = handle.handle_id;

        let bytes = self
            .games
            .read()
            .await
            .get(&handle_id)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("unknown handle {handle_id}")))?;

        let state = G::decode_state(&bytes).map_err(|e| Status::internal(e.to_string()))?;
        if G::outcome(&state).is_terminal() {
            return Err(Status::failed_precondition("game is already over"));
        }

        let chosen_move = match player {
            EnginePlayer::Nn => {
                let model = self.model_for(&inner.model)?.clone();
                nn_choose_move::<G, M>(&model, &state, inner.budget.as_ref())
                    .await
                    .map_err(|e| Status::internal(format!("nn move failed: {e}")))?
            }
            // An external reference engine (MoHex, GNU Go, …), driven by the
            // game crate. Blocking (usually a subprocess over GTP), and absent
            // unless the engine was started with one configured.
            EnginePlayer::External => {
                let budget = SolverBudget {
                    max_millis: inner.budget.as_ref().and_then(|b| b.max_millis),
                    max_visits: inner.budget.as_ref().and_then(|b| b.max_visits),
                };
                tokio::task::spawn_blocking(move || G::choose_move_external(&state, budget))
                    .await
                    .map_err(|e| Status::internal(format!("external engine panicked: {e}")))?
                    .ok_or_else(|| {
                        Status::failed_precondition(
                            "no external engine configured (start the engine with one)",
                        )
                    })?
                    .map(|out| Chosen {
                        payload: G::encode_move(&out.best_move),
                        value: out.value,
                        nodes: out.nodes,
                        reifications: 0,
                    })
                    .map_err(|e| Status::internal(format!("external engine failed: {e}")))?
            }
            // Default / Solver: the built-in perfect solver. The solve can be
            // CPU-heavy, so keep it off the async runtime's worker threads.
            _ => tokio::task::spawn_blocking(move || {
                let out = G::choose_move(&state);
                Chosen {
                    payload: G::encode_move(&out.best_move),
                    value: out.value,
                    nodes: out.nodes,
                    reifications: 0,
                }
            })
            .await
            .map_err(|e| Status::internal(format!("solver panicked: {e}")))?,
        };

        let chosen = game_pb::Move {
            game_id: G::id() as i32,
            payload: chosen_move.payload,
        };
        let stats = SearchStats {
            visits: chosen_move.nodes.min(u32::MAX as u64) as u32,
            reifications: chosen_move.reifications,
            best_value: chosen_move.value,
        };

        // Buffer both messages synchronously and close the channel before
        // returning — a deterministic, already-complete stream (no spawn race),
        // which produces cleaner gRPC-Web framing for browser clients.
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let _ = tx.try_send(Ok(MoveProgress {
            kind: Some(move_progress::Kind::Stats(stats)),
        }));
        let _ = tx.try_send(Ok(MoveProgress {
            kind: Some(move_progress::Kind::Chosen(chosen)),
        }));
        drop(tx);

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }

    type PlayMatchStream = tokio_stream::wrappers::ReceiverStream<Result<MatchProgress, Status>>;
    async fn play_match(
        &self,
        _req: Request<MatchSpec>,
    ) -> Result<Response<Self::PlayMatchStream>, Status> {
        Err(Status::unimplemented("PlayMatch: implement in M2"))
    }

    type StartSelfPlayStream =
        tokio_stream::wrappers::ReceiverStream<Result<SelfPlayProgress, Status>>;
    async fn start_self_play(
        &self,
        _req: Request<SelfPlaySpec>,
    ) -> Result<Response<Self::StartSelfPlayStream>, Status> {
        Err(Status::unimplemented("StartSelfPlay: implement in M2"))
    }

    async fn health(
        &self,
        _req: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_games: vec![G::id() as i32],
        }))
    }
}

// ── Server runner ─────────────────────────────────────────────────────────────

/// Serve `EngineService` on `addr` with gRPC-Web + permissive CORS so the
/// browser can call it directly.
pub async fn run<G, M>(addr: SocketAddr, models: HashMap<String, M>) -> anyhow::Result<()>
where
    G: Solver + Analyzable,
    M: ModelClient<G> + 'static,
{
    let model_names: Vec<String> = models.keys().cloned().collect();
    let svc = EngineServiceServer::new(EngineServer::<G, M>::new(models));

    tracing::info!(%addr, game_id = G::id(), models = ?model_names, "engine server listening");

    tonic::transport::Server::builder()
        .accept_http1(true)
        // CORS must wrap (be outer to) the gRPC-Web layer so browser preflight
        // OPTIONS requests get answered before gRPC-Web tries to handle them.
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any),
        )
        .layer(tonic_web::GrpcWebLayer::new())
        .add_service(svc)
        .serve(addr)
        .await?;

    Ok(())
}
