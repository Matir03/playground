//! ModelClient implementation that calls the Python ModelService over gRPC.
//!
//! Responsible for micro-batching across concurrent requests so the model
//! server sees large-enough batches to amortize GPU cost. Stub for now —
//! batching wired up in M2. The `unary` path works end-to-end as soon as
//! a model server is running.

use std::marker::PhantomData;

use async_trait::async_trait;
use playground_core::{Game, Gs, ModelClient, PolicyCandidate};
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
    async fn encode(&self, state: &G::State) -> anyhow::Result<Gs> {
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
        let gs = resp.gs.ok_or_else(|| anyhow::anyhow!("model server returned empty GS"))?;
        Ok(Gs::new(gs.payload, gs.shape, gs.model_version))
    }

    async fn eval(&self, gs: &Gs) -> anyhow::Result<f64> {
        let req = pb::EvalRequest {
            gs: Some(to_proto_gs(gs)),
        };
        let mut client = self.client.lock().await;
        Ok(client.eval(req).await?.into_inner().value)
    }

    async fn policy(&self, gs: &Gs, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        let req = pb::PolicyRequest {
            gs: Some(to_proto_gs(gs)),
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
                    next_gs: from_proto_gs(c.next_gs?),
                })
            })
            .collect())
    }

    async fn reify(
        &self,
        from_state: &G::State,
        from_gs: &Gs,
        to_gs: &Gs,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<G::Move> {
        let req = pb::ReifyRequest {
            from_gs: Some(to_proto_gs(from_gs)),
            to_gs: Some(to_proto_gs(to_gs)),
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
        let mv = resp.r#move.ok_or_else(|| anyhow::anyhow!("reify returned empty move"))?;
        G::decode_move(&mv.payload)
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        let mut client = self.client.lock().await;
        let resp = client.health(pb::HealthRequest {}).await?.into_inner();
        Ok(resp.model_version)
    }
}

fn to_proto_gs(gs: &Gs) -> playground_protocol::game_pb::Gs {
    playground_protocol::game_pb::Gs {
        payload: gs.payload.to_vec(),
        shape: gs.shape.clone(),
        model_version: gs.model_version.clone(),
    }
}

fn from_proto_gs(gs: playground_protocol::game_pb::Gs) -> Gs {
    Gs::new(gs.payload, gs.shape, gs.model_version)
}
