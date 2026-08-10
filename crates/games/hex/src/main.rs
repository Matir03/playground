//! Hex 13×13 engine binary. See connect4-engine for subcommand semantics.
//!
//! Beyond `serve`/`self-play`, this exposes the MoHex bridge: `verify` checks our
//! rules against MoHex over random games, `match` plays games against it.

use std::net::SocketAddr;
use std::time::Instant;

use clap::{Parser, Subcommand};
use playground_core::Game;
use playground_model_client::RemoteModelClient;

use hex_engine::game::Hex;
use hex_engine::mohex;

#[derive(Parser)]
#[command(name = "hex-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7104")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7204")]
        model_uri: String,
        /// Play MoHex for `RequestMove(player = EXTERNAL)` seats. Needs
        /// scripts/fetch-mohex.sh; without it those requests are refused.
        #[arg(long)]
        mohex_bin: Option<String>,
        /// Default MoHex playouts per move (a request's budget overrides it).
        #[arg(long, default_value_t = 10_000)]
        mohex_playouts: u32,
        /// Default MoHex seconds per move (0 = budget by playouts only).
        #[arg(long, default_value_t = 0.0)]
        mohex_time: f64,
    },
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7104")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7204")]
        model_uri: String,
        #[arg(long, default_value_t = 16)]
        workers: u32,
    },
    /// Cross-check our rules against MoHex over random games.
    Verify {
        #[arg(long, default_value = "mohex")]
        mohex_bin: String,
        #[arg(long, default_value_t = 20)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Play MoHex against a random-move stand-in (until the AZ net lands, this
    /// is the end-to-end check that the bridge drives a whole game).
    Match {
        #[arg(long, default_value = "mohex")]
        mohex_bin: String,
        #[arg(long, default_value_t = 2)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// MoHex playouts per move (0 = its default).
        #[arg(long, default_value_t = 1000)]
        mohex_playouts: u32,
        /// MoHex seconds per move (0 = unlimited, budget by playouts).
        #[arg(long, default_value_t = 0.0)]
        mohex_time: f64,
    },
    /// Random-playout throughput of the rules (moves/s), the AZ self-play floor.
    Bench {
        #[arg(long, default_value_t = 2000)]
        games: u32,
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
        Cmd::Serve { addr, model_uri, mohex_bin, mohex_playouts, mohex_time } => {
            if let Some(bin) = &mohex_bin {
                mohex::install(bin, mohex_playouts, mohex_time)?;
                tracing::info!(bin, mohex_playouts, mohex_time, "mohex opponent ready");
            }
            serve(addr, &model_uri).await?;
        }
        Cmd::SelfPlay { addr, model_uri, .. } => serve(addr, &model_uri).await?,
        Cmd::Verify { mohex_bin, games, seed } => {
            mohex::verify_rules(&mohex_bin, games, seed)?;
        }
        Cmd::Match { mohex_bin, games, seed, mohex_playouts, mohex_time } => {
            let mut rng = seed | 1;
            mohex::play_match(&mohex_bin, games, mohex_playouts, mohex_time, move |s| {
                let moves = Hex::legal_moves(s);
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                moves[(rng >> 33) as usize % moves.len()]
            })?;
        }
        Cmd::Bench { games } => bench(games),
        Cmd::Stdio => anyhow::bail!("stdio: implement when needed"),
    }
    Ok(())
}

async fn serve(addr: SocketAddr, model_uri: &str) -> anyhow::Result<()> {
    let model = RemoteModelClient::<Hex>::connect(model_uri).await?;
    let models = std::collections::HashMap::from([("nn".to_string(), model)]);
    playground_engine_server::run::<Hex, _>(addr, models).await
}

/// Random playouts to completion, measuring rules throughput (legal-move
/// generation + apply + the connectivity check that decides the winner).
fn bench(games: u32) {
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut moves_played = 0u64;
    let mut black_wins = 0u32;
    let start = Instant::now();
    for _ in 0..games {
        let mut state = Hex::initial_state();
        while state.winner == 0 {
            let moves = Hex::legal_moves(&state);
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mv = moves[(rng >> 33) as usize % moves.len()];
            state = Hex::apply(&state, &mv);
            moves_played += 1;
        }
        black_wins += u32::from(state.winner == 1);
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{games} random games in {secs:.2}s — {:.0} moves/s, {:.1} plies/game, black won {:.1}%",
        moves_played as f64 / secs,
        moves_played as f64 / f64::from(games),
        100.0 * f64::from(black_wins) / f64::from(games),
    );
}
