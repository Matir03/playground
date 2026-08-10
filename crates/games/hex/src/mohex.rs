//! MoHex bridge — a GTP client for the reference Hex engine (benzene's MoHex
//! 2.0), mirroring go9's GNU Go bridge.
//!
//! MoHex serves two purposes here:
//!   * **rules oracle** — `mohex verify` plays random games through both
//!     implementations and asserts our `winner` agrees with MoHex's
//!     `final_score` on every game, which is the cheapest way to be sure the
//!     board geometry (neighbourhood, edges, colour-to-edge mapping) is right;
//!   * **absolute-strength opponent** — once the AZ net exists, the same client
//!     drives `hex-engine match`, exactly as `go9_vs_gnugo` does.
//!
//! Build it with `scripts/fetch-mohex.sh`. MoHex's `param_game allow_swap`
//! defaults to off, matching [`crate::game::ALLOW_SWAP`]; we set it explicitly
//! anyway so the two never drift.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use playground_core::{Game, Outcome};

use crate::game::{cell_name, parse_cell, Hex, HexMove, HexState, N};

/// Minimal GTP client over a MoHex subprocess.
pub struct Mohex {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Mohex {
    /// Spawn `bin` and set up a fresh 13×13 game. `max_games` is MoHex's
    /// playout budget per move (0 = leave its default), `max_time` its
    /// per-move seconds.
    pub fn new(bin: &str, max_games: u32, max_time: f64) -> anyhow::Result<Self> {
        let debug = std::env::var_os("HEX_GTP_DEBUG").is_some();
        let mut child = Command::new(bin)
            .arg("--quiet")
            // MoHex drops a `default.log` in the cwd unless told not to; keep it
            // when tracing, since its search log is where its reasoning lives.
            .args(if debug { ["--use-logfile", "1"] } else { ["--use-logfile", "0"] })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // MoHex's banner and search chatter go to stderr; surface it only
            // when tracing (see `send`).
            .stderr(if debug { Stdio::inherit() } else { Stdio::null() })
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawning {bin}: {e} (run scripts/fetch-mohex.sh)"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut m = Self { child, stdin, stdout };
        m.send(&format!("boardsize {N}"))?;
        m.send(&format!("param_game allow_swap {}", u8::from(crate::game::ALLOW_SWAP)))?;
        if max_games > 0 {
            m.send(&format!("param_mohex max_games {max_games}"))?;
        }
        if max_time > 0.0 {
            m.send(&format!("param_mohex max_time {max_time}"))?;
        }
        Ok(m)
    }

    /// Send a GTP command and return the response body. A `?`-prefixed reply
    /// (GTP failure) becomes an `Err`. Set `HEX_GTP_DEBUG=1` to trace the
    /// conversation on stderr.
    pub fn send(&mut self, cmd: &str) -> anyhow::Result<String> {
        let debug = std::env::var_os("HEX_GTP_DEBUG").is_some();
        if debug {
            eprintln!("gtp >> {cmd}");
        }
        writeln!(self.stdin, "{cmd}")?;
        self.stdin.flush()?;
        let mut out: Vec<String> = Vec::new();
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                anyhow::bail!("mohex closed its output during {cmd:?}");
            }
            let t = line.trim_end().to_string();
            if t.is_empty() {
                if !out.is_empty() {
                    break; // blank line terminates a GTP response
                }
                continue;
            }
            out.push(t);
        }
        let text = out.join("\n");
        if debug {
            eprintln!("gtp << {text}");
        }
        let body = text.get(1..).unwrap_or("").trim().to_string();
        if text.starts_with('?') {
            anyhow::bail!("mohex rejected {cmd:?}: {body}");
        }
        Ok(body)
    }

    /// Reset to an empty board (keeps the process and its parameters).
    pub fn clear_board(&mut self) -> anyhow::Result<()> {
        self.send("clear_board").map(|_| ())
    }

    /// Tell MoHex about a move made outside its search.
    pub fn play(&mut self, player: u8, mv: &HexMove) -> anyhow::Result<()> {
        self.send(&format!("play {} {}", color_name(player), move_name(mv)))?;
        Ok(())
    }

    /// Ask MoHex for a move. `None` means it resigned.
    pub fn genmove(&mut self, player: u8) -> anyhow::Result<Option<HexMove>> {
        let reply = self.send(&format!("genmove {}", color_name(player)))?;
        parse_move(&reply)
    }

    /// MoHex's verdict on the current position: `Some(1)` black, `Some(2)`
    /// white, `None` if neither side has connected yet.
    pub fn winner(&mut self) -> anyhow::Result<Option<u8>> {
        match self.send("final_score")?.trim().to_ascii_lowercase().as_str() {
            "b" | "black" | "b+" => Ok(Some(1)),
            "w" | "white" | "w+" => Ok(Some(2)),
            other => {
                if other.starts_with('b') {
                    Ok(Some(1))
                } else if other.starts_with('w') {
                    Ok(Some(2))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

impl Drop for Mohex {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn color_name(player: u8) -> &'static str {
    if player == 1 { "black" } else { "white" }
}

/// Our move → GTP move token (`swap-pieces` is benzene's spelling).
pub fn move_name(mv: &HexMove) -> String {
    match mv {
        HexMove::Play { r, c } => cell_name(*r as usize, *c as usize),
        HexMove::Swap => "swap-pieces".to_string(),
    }
}

/// GTP move token → our move. `Ok(None)` for a resignation.
pub fn parse_move(token: &str) -> anyhow::Result<Option<HexMove>> {
    let t = token.trim().to_ascii_lowercase();
    if t == "resign" {
        return Ok(None);
    }
    if t.starts_with("swap") {
        return Ok(Some(HexMove::Swap));
    }
    let (r, c) = parse_cell(&t).ok_or_else(|| anyhow::anyhow!("bad GTP move {token:?}"))?;
    Ok(Some(HexMove::Play { r: r as u8, c: c as u8 }))
}

/// Cross-check our rules against MoHex: play `games` random games, feeding
/// every move to MoHex, and assert that both sides call the same winner at the
/// same moment. Returns the number of games checked.
///
/// Random play is the point — it fills the board with ragged chains and exercises
/// the connectivity test far harder than engine play would.
pub fn verify_rules(bin: &str, games: u32, seed: u64) -> anyhow::Result<u32> {
    let mut mohex = Mohex::new(bin, 1, 0.0)?; // no search: we never call genmove
    let mut rng = seed | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for g in 0..games {
        mohex.clear_board()?;
        let mut state = Hex::initial_state();
        let mut plies = 0u32;
        while state.winner == 0 {
            let moves = Hex::legal_moves(&state);
            anyhow::ensure!(!moves.is_empty(), "game {g}: board full with no winner");
            let mv = moves[(next() >> 33) as usize % moves.len()];
            let mover = state.to_move;
            mohex.play(mover, &mv)?;
            state = Hex::apply(&state, &mv);
            plies += 1;

            // MoHex must agree at every step, not just at the end: it should see
            // no winner until the exact ply where we do.
            let theirs = mohex.winner()?;
            let ours = (state.winner != 0).then_some(state.winner);
            anyhow::ensure!(
                theirs == ours,
                "game {g} ply {plies}: mohex says {theirs:?}, we say {ours:?}\n{}",
                crate::game::render(&state)
            );
        }
        let outcome = Hex::outcome(&state);
        anyhow::ensure!(outcome.is_terminal() && outcome != Outcome::Draw, "hex has no draws");
        eprintln!(
            "  game {g}: {} wins after {plies} plies (mohex agrees)",
            if state.winner == 1 { "black" } else { "white" },
        );
    }
    println!("hex rules verified against mohex over {games} random games");
    Ok(games)
}

/// Play `games` games between MoHex and the built-in placeholder engine,
/// alternating colours. Returns MoHex's score in [0,1] — which will be 1.0
/// until a real net is wired in; the point today is that a full game runs
/// end-to-end through the bridge.
pub fn play_match(
    bin: &str,
    games: u32,
    max_games: u32,
    max_time: f64,
    mut our_move: impl FnMut(&HexState) -> HexMove,
) -> anyhow::Result<f64> {
    let mut mohex = Mohex::new(bin, max_games, max_time)?;
    let mut score = 0.0;
    for g in 0..games {
        mohex.clear_board()?;
        let mohex_black = g % 2 == 0;
        let mut state = Hex::initial_state();
        let mut resigned = false;
        while state.winner == 0 {
            let black_to_move = state.to_move == 1;
            let mv = if black_to_move == mohex_black {
                match mohex.genmove(state.to_move)? {
                    Some(mv) => mv,
                    None => {
                        resigned = true;
                        break;
                    }
                }
            } else {
                let mv = our_move(&state);
                mohex.play(state.to_move, &mv)?;
                mv
            };
            state = Hex::apply(&state, &mv);
        }
        let mohex_won = if resigned {
            false
        } else {
            (state.winner == 1) == mohex_black
        };
        score += f64::from(u8::from(mohex_won));
        eprintln!(
            "  game {g}: mohex={} -> {} ({} plies)",
            if mohex_black { "black" } else { "white" },
            if mohex_won { "win" } else { "loss" },
            state.ply,
        );
    }
    let s = score / f64::from(games);
    println!("mohex_score {s:.3}  ({games} games)");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtp_move_tokens_roundtrip() {
        let mv = HexMove::Play { r: 4, c: 8 };
        assert_eq!(move_name(&mv), "i5");
        assert_eq!(parse_move("I5").unwrap(), Some(mv));
        assert_eq!(parse_move("swap-pieces").unwrap(), Some(HexMove::Swap));
        assert_eq!(parse_move("resign").unwrap(), None);
        assert!(parse_move("z9").is_err());
    }
}
