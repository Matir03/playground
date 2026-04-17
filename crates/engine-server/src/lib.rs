//! tonic server implementing `EngineService` generically over a `Game`.
//!
//! Game binaries (`crates/games/<name>`) wire up a concrete Game + a
//! ModelClient and call `run`.

use std::marker::PhantomData;
use std::net::SocketAddr;

use playground_core::{Game, ModelClient};
use playground_protocol::engine_v1::engine_service_server::{EngineService, EngineServiceServer};
use playground_protocol::engine_v1::*;
use playground_protocol::game_v1::{GameHandle, GameState};
use tonic::{Request, Response, Status};

/// Long-running engine server. Holds a ModelClient and supports the three
/// modes described in docs/protocols.md (interactive, match, batched self-play).
pub struct EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    // In a real impl: handle registry, self-play worker pool, etc.
    _model: std::sync::Arc<M>,
    _game: PhantomData<G>,
}

impl<G, M> EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            _model: std::sync::Arc::new(model),
            _game: PhantomData,
        }
    }
}

#[tonic::async_trait]
impl<G, M> EngineService for EngineServer<G, M>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    async fn new_game(&self, _req: Request<NewGameRequest>) -> Result<Response<GameHandle>, Status> {
        Err(Status::unimplemented("NewGame: implement in M1"))
    }

    async fn get_state(&self, _req: Request<GameHandle>) -> Result<Response<GameState>, Status> {
        Err(Status::unimplemented("GetState: implement in M1"))
    }

    async fn apply_move(&self, _req: Request<ApplyMoveRequest>) -> Result<Response<GameState>, Status> {
        Err(Status::unimplemented("ApplyMove: implement in M1"))
    }

    type RequestMoveStream = tokio_stream::wrappers::ReceiverStream<Result<MoveProgress, Status>>;
    async fn request_move(
        &self,
        _req: Request<RequestMoveRequest>,
    ) -> Result<Response<Self::RequestMoveStream>, Status> {
        Err(Status::unimplemented("RequestMove: implement in M2"))
    }

    type PlayMatchStream = tokio_stream::wrappers::ReceiverStream<Result<MatchProgress, Status>>;
    async fn play_match(
        &self,
        _req: Request<MatchSpec>,
    ) -> Result<Response<Self::PlayMatchStream>, Status> {
        Err(Status::unimplemented("PlayMatch: implement in M2"))
    }

    type StartSelfPlayStream = tokio_stream::wrappers::ReceiverStream<Result<SelfPlayProgress, Status>>;
    async fn start_self_play(
        &self,
        _req: Request<SelfPlaySpec>,
    ) -> Result<Response<Self::StartSelfPlayStream>, Status> {
        Err(Status::unimplemented("StartSelfPlay: implement in M2"))
    }

    async fn health(&self, _req: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_games: vec![G::id() as i32],
        }))
    }
}

/// Serve `EngineService` on `addr`, wrapped with tonic-web so the browser
/// can reach it via gRPC-Web. See docs/decisions.md#d4.
pub async fn run<G, M>(addr: SocketAddr, model: M) -> anyhow::Result<()>
where
    G: Game,
    M: ModelClient<G> + 'static,
{
    let svc = EngineServiceServer::new(EngineServer::<G, M>::new(model));

    tracing::info!(%addr, game_id = G::id(), "engine server listening");

    tonic::transport::Server::builder()
        .accept_http1(true) // required for tonic-web
        .layer(tonic_web::GrpcWebLayer::new())
        .add_service(svc)
        .serve(addr)
        .await?;

    Ok(())
}
