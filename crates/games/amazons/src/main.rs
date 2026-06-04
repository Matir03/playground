//! Amazons engine binary. See connect4-engine for subcommand semantics.

mod game;

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use game::Amazons;
use playground_model_client::RemoteModelClient;

#[derive(Parser)]
#[command(name = "amazons-engine", version)]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7102")] addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7202")] model_uri: String,
    },
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7102")] addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7202")] model_uri: String,
        #[arg(long, default_value_t = 16)] workers: u32,
    },
    Match {
        #[arg(long)] first_player_uri: String,
        #[arg(long)] second_player_uri: String,
        #[arg(long, default_value_t = 1)] num_games: u32,
    },
    Stdio,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { addr, model_uri } | Cmd::SelfPlay { addr, model_uri, .. } => {
            let model = RemoteModelClient::<Amazons>::connect(&model_uri).await?;
            let models = std::collections::HashMap::from([("nn".to_string(), model)]);
            playground_engine_server::run::<Amazons, _>(addr, models).await?;
        }
        Cmd::Match { .. } => anyhow::bail!("match: implement in M2"),
        Cmd::Stdio => anyhow::bail!("stdio: implement when needed"),
    }
    Ok(())
}
