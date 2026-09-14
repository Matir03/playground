//! ModelClient implementation that calls the Python ModelService over gRPC.
//!
//! Responsible for micro-batching across concurrent requests so the model
//! server sees large-enough batches to amortize GPU cost. Stub for now —
//! batching wired up in M2. The `unary` path works end-to-end as soon as
//! a model server is running.

use std::marker::PhantomData;

use async_trait::async_trait;
use playground_core::{Game, LatentGameState, ModelClient, PolicyCandidate};
use playground_protocol::model_pb as pb;
use playground_protocol::model_pb::model_service_client::ModelServiceClient;
use tonic::transport::Channel;

pub struct RemoteModelClient<G: Game> {
    client: tokio::sync::Mutex<ModelServiceClient<Channel>>,
    _game: PhantomData<G>,
}

impl<G: Game> RemoteModelClient<G> {
    pub async fn connect(uri: &str) -> anyhow::Result<Self> {
        // connect_lazy defers the actual TCP connection until the first RPC;
        // this lets the engine start without a model server running (human play).
        let channel = Channel::from_shared(uri.to_string())?.connect_lazy();
        Ok(Self {
            client: tokio::sync::Mutex::new(ModelServiceClient::new(channel)),
            _game: PhantomData,
        })
    }
}

#[async_trait]
impl<G: Game> ModelClient<G> for RemoteModelClient<G> {
    async fn encode(&self, state: &G::State) -> anyhow::Result<LatentGameState> {
        let payload = G::encode_state(state);
        let req = pb::EncodeRequest {
            state: Some(playground_protocol::game_pb::GameState {
                game_id: G::id() as i32,
                payload,
                to_move: 0,
                outcome: 0,
                ply: 0,
            }),
        };
        let mut client = self.client.lock().await;
        let resp = client.encode(req).await?.into_inner();
        let lgs = resp
            .lgs
            .ok_or_else(|| anyhow::anyhow!("model server returned empty LGS"))?;
        Ok(LatentGameState::new(
            lgs.payload,
            lgs.shape,
            lgs.model_version,
        ))
    }

    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        let req = pb::EvalRequest {
            lgs: Some(to_proto_lgs(lgs)),
        };
        let mut client = self.client.lock().await;
        Ok(client.eval(req).await?.into_inner().value)
    }

    async fn policy(
        &self,
        lgs: &LatentGameState,
        top_b: u32,
    ) -> anyhow::Result<Vec<PolicyCandidate>> {
        let req = pb::PolicyRequest {
            lgs: Some(to_proto_lgs(lgs)),
            top_b,
        };
        let mut client = self.client.lock().await;
        let resp = client.policy(req).await?.into_inner();
        Ok(resp
            .candidates
            .into_iter()
            .filter_map(|c| {
                Some(PolicyCandidate {
                    prior: c.prior,
                    action_vector: c.action_vector.into(),
                    next_lgs: from_proto_lgs(c.next_lgs?),
                })
            })
            .collect())
    }

    async fn reify(
        &self,
        from_state: &G::State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<G::Move> {
        let req = pb::ReifyRequest {
            from_lgs: Some(to_proto_lgs(from_lgs)),
            to_lgs: Some(to_proto_lgs(to_lgs)),
            from_state: Some(playground_protocol::game_pb::GameState {
                game_id: G::id() as i32,
                payload: G::encode_state(from_state),
                to_move: 0,
                outcome: 0,
                ply: 0,
            }),
            action_vector: action_seed.map(|s| s.to_vec()),
            noise_scale,
        };
        let mut client = self.client.lock().await;
        let resp = client.reify(req).await?.into_inner();
        let mv = resp
            .r#move
            .ok_or_else(|| anyhow::anyhow!("reify returned empty move"))?;
        G::decode_move(&mv.payload)
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        let mut client = self.client.lock().await;
        let resp = client.health(pb::HealthRequest {}).await?.into_inner();
        Ok(resp.model_version)
    }
}

fn to_proto_lgs(lgs: &LatentGameState) -> playground_protocol::game_pb::LatentGameState {
    playground_protocol::game_pb::LatentGameState {
        payload: lgs.payload.to_vec(),
        shape: lgs.shape.clone(),
        model_version: lgs.model_version.clone(),
    }
}

fn from_proto_lgs(lgs: playground_protocol::game_pb::LatentGameState) -> LatentGameState {
    LatentGameState::new(lgs.payload, lgs.shape, lgs.model_version)
}
