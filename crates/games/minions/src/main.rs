//! Simplified-Minions engine binary. Subcommand semantics follow
//! arimaa-engine: `serve` is the interactive gRPC-Web endpoint the web UI
//! talks to, and works without a model server running (the model channel is
//! lazy) — only the NN seat needs one.

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use minions_engine::game::Minions;
use playground_model_client::RemoteModelClient;

#[derive(Parser)]
#[command(name = "minions-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7105")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7205")]
        model_uri: String,
    },
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7105")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7205")]
        model_uri: String,
        #[arg(long, default_value_t = 16)]
        workers: u32,
    },
    /// Play the greedy solver against itself and print the final board.
    GreedyGame {
        #[arg(long, default_value_t = 1)]
        games: u32,
        /// Print the board after every completed turn.
        #[arg(long)]
        verbose: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { addr, model_uri } | Cmd::SelfPlay { addr, model_uri, .. } => {
            let model = RemoteModelClient::<Minions>::connect(&model_uri).await?;
            let models = std::collections::HashMap::from([("nn".to_string(), model)]);
            playground_engine_server::run::<Minions, _>(addr, models).await?;
        }
        Cmd::GreedyGame { games, verbose } => greedy_games(games, verbose),
    }
    Ok(())
}

fn greedy_games(games: u32, verbose: bool) {
    use playground_core::{Game, Outcome, Solver};

    let mut tally = [0u32; 3]; // first wins, second wins, draws
    for g in 0..games {
        let mut state = Minions::initial_state();
        let mut turn = state.turn;
        while Minions::outcome(&state) == Outcome::InProgress {
            let mv = Minions::choose_move(&state).best_move;
            state = Minions::apply(&state, &mv);
            if verbose && state.turn != turn {
                turn = state.turn;
                println!("{}", state.render());
            }
        }
        let outcome = Minions::outcome(&state);
        match outcome {
            Outcome::FirstWins => tally[0] += 1,
            Outcome::SecondWins => tally[1] += 1,
            _ => tally[2] += 1,
        }
        println!(
            "game {g}: {outcome:?} after {} turns / {} actions",
            state.turn, state.ply
        );
        if !verbose {
            println!("{}", state.render());
        }
    }
    println!("first {} / second {} / draw {}", tally[0], tally[1], tally[2]);
}
