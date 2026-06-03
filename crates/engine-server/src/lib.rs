//! tonic server implementing `EngineService` generically over a `Game`.
//!
//! Implements the M1 interactive-play RPCs (NewGame / GetState / ApplyMove).
//! RequestMove, PlayMatch, and StartSelfPlay are stubbed for M2.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::sync::Arc;

use playground_core::{Game, ModelClient, Outcome, Player, Solver};
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
    format!("{:08x}{:08x}", t, HANDLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

static HANDLE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// ── Conversions ───────────────────────────────────────────────────────────────

fn player_to_proto(p: Player) -> i32 {
    match p {
        Player::First  => game_pb::Player::First as i32,
        Player::Second => game_pb::Player::Second as i32,
    }
}

fn outcome_to_proto(o: Outcome) -> i32 {
    match o {
        Outcome::InProgress  => game_pb::Outcome::InProgress as i32,
        Outcome::FirstWins   => game_pb::Outcome::FirstWins as i32,
        Outcome::SecondWins  => game_pb::Outcome::SecondWins as i32,
        Outcome::Draw        => game_pb::Outcome::Draw as i32,
    }
}

fn state_to_proto<G: Game>(state: &G::State) -> GameState {
    GameState {
        game_id:  G::id() as i32,
        payload:  G::encode_state(state).into(),
        to_move:  player_to_proto(G::to_move(state)),
        outcome:  outcome_to_proto(G::outcome(state)),
        ply:      G::ply(state),
    }
}

// ── Server struct ─────────────────────────────────────────────────────────────

pub struct EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    model: Arc<M>,
    games:  Registry,
    _game:  PhantomData<G>,
}

impl<G, M> EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            model: Arc::new(model),
            games:  Arc::new(RwLock::new(HashMap::new())),
            _game:  PhantomData,
        }
    }
}

/// Result of picking a move: serialized move payload, value for the side to
/// move in [-1, 1], and a node/visit count for telemetry.
struct Chosen {
    payload: Vec<u8>,
    value: f64,
    nodes: u64,
}

/// Neural-net move: 1-ply lookahead using the model's encode+eval. Terminal
/// children use their exact value; others are evaluated by the NN. With random
/// weights this plays weakly but exercises the full engine↔model-server path.
/// (The full KR-UCT MCTS over GS arrives with M4 — see docs/algorithm.md.)
async fn nn_choose_move<G, M>(model: &M, state: &G::State) -> anyhow::Result<Chosen>
where
    G: Game,
    M: ModelClient<G>,
{
    let mover = G::to_move(state);
    let mut best: Option<G::Move> = None;
    let mut best_value = f64::NEG_INFINITY;
    let mut nodes = 0u64;
    for mv in G::legal_moves(state) {
        let child = G::apply(state, &mv);
        let value = if G::outcome(&child).is_terminal() {
            G::outcome(&child).value_for(mover)
        } else {
            let gs = model.encode(&child).await?;
            // eval is from the child's side-to-move (the opponent) perspective.
            -model.eval(&gs).await?
        };
        nodes += 1;
        if value > best_value {
            best_value = value;
            best = Some(mv);
        }
    }
    let mv = best.ok_or_else(|| anyhow::anyhow!("no legal moves"))?;
    Ok(Chosen { payload: G::encode_move(&mv), value: best_value, nodes })
}

// ── RPC implementations ───────────────────────────────────────────────────────

#[tonic::async_trait]
impl<G, M> EngineService for EngineServer<G, M>
where
    G: Solver,
    M: ModelClient<G> + 'static,
{
    async fn new_game(&self, req: Request<NewGameRequest>) -> Result<Response<GameHandle>, Status> {
        let inner = req.into_inner();
        let state = match inner.initial_state {
            Some(gs) => G::decode_state(&gs.payload)
                .map_err(|e| Status::invalid_argument(e.to_string()))?,
            None => G::initial_state(),
        };
        let id = new_handle_id();
        self.games.write().await.insert(id.clone(), G::encode_state(&state));
        Ok(Response::new(GameHandle {
            handle_id: id,
            game_id:   G::id() as i32,
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
        let is_legal = legal.iter().any(|m| {
            G::encode_move(m) == G::encode_move(&mv)
        });
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
            EnginePlayer::Nn => nn_choose_move::<G, M>(&self.model, &state)
                .await
                .map_err(|e| Status::internal(format!("nn move failed: {e}")))?,
            // Default / Solver: the built-in perfect solver. The solve can be
            // CPU-heavy, so keep it off the async runtime's worker threads.
            _ => tokio::task::spawn_blocking(move || {
                let out = G::choose_move(&state);
                Chosen {
                    payload: G::encode_move(&out.best_move),
                    value: out.value,
                    nodes: out.nodes,
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
            reifications: 0,
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

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
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
            version:          env!("CARGO_PKG_VERSION").to_string(),
            supported_games:  vec![G::id() as i32],
        }))
    }
}

// ── Server runner ─────────────────────────────────────────────────────────────

/// Serve `EngineService` on `addr` with gRPC-Web + permissive CORS so the
/// browser can call it directly.
pub async fn run<G, M>(addr: SocketAddr, model: M) -> anyhow::Result<()>
where
    G: Solver,
    M: ModelClient<G> + 'static,
{
    let svc = EngineServiceServer::new(EngineServer::<G, M>::new(model));

    tracing::info!(%addr, game_id = G::id(), "engine server listening");

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
