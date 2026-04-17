//! Connect 4 engine binary.
//!
//! See docs/protocols.md for subcommand semantics.

mod game;

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use game::Connect4;
use playground_model_client::RemoteModelClient;

#[derive(Parser)]
#[command(name = "connect4-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Long-lived gRPC server. Interactive play + matches.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7100")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7200")]
        model_uri: String,
    },
    /// As Serve, but also accepts StartSelfPlay requests.
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7100")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7200")]
        model_uri: String,
        #[arg(long, default_value_t = 16)]
        workers: u32,
    },
    /// Client-side match runner.
    Match {
        #[arg(long)]
        first_player_uri: String,
        #[arg(long)]
        second_player_uri: String,
        #[arg(long, default_value_t = 1)]
        num_games: u32,
    },
    /// Newline-delimited stdio protocol for debugging / tools.
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
            let model = RemoteModelClient::<Connect4>::connect(&model_uri).await?;
            playground_engine_server::run::<Connect4, _>(addr, model).await?;
        }
        Cmd::Match { .. } => {
            anyhow::bail!("match subcommand: implement in M2");
        }
        Cmd::Stdio => {
            anyhow::bail!("stdio subcommand: implement when a second tool needs it");
        }
    }
    Ok(())
}
