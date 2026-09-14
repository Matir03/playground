//! MoHex bridge — a GTP client for the reference Hex engine (benzene's MoHex
//! 2.0), mirroring go9's GNU Go bridge.
//!
//! MoHex serves three purposes here:
//!   * **rules oracle** — `mohex verify` plays random games through both
//!     implementations and asserts our `winner` agrees with MoHex's
//!     `final_score` on every game, which is the cheapest way to be sure the
//!     board geometry (neighbourhood, edges, colour-to-edge mapping) is right;
//!   * **absolute-strength opponent** — once the AZ net exists, the same client
//!     drives `hex-engine match`, exactly as `go9_vs_gnugo` does;
//!   * **interactive opponent** — [`install`] hands a MoHex process to the
//!     engine server, which answers `RequestMove(player = EXTERNAL)` with it,
//!     so the web UI can offer MoHex as a seat.
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
            .args(if debug {
                ["--use-logfile", "1"]
            } else {
                ["--use-logfile", "0"]
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // MoHex's banner and search chatter go to stderr; surface it only
            // when tracing (see `send`).
            .stderr(if debug {
                Stdio::inherit()
            } else {
                Stdio::null()
            })
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawning {bin}: {e} (run scripts/fetch-mohex.sh)"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut m = Self {
            child,
            stdin,
            stdout,
        };
        m.send(&format!("boardsize {N}"))?;
        m.send(&format!(
            "param_game allow_swap {}",
            u8::from(crate::game::ALLOW_SWAP)
        ))?;
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

    /// Set MoHex's per-move budget: playouts and/or seconds (0 = leave as is).
    pub fn set_budget(&mut self, max_games: u32, max_time: f64) -> anyhow::Result<()> {
        if max_games > 0 {
            self.send(&format!("param_mohex max_games {max_games}"))?;
        }
        if max_time > 0.0 {
            self.send(&format!("param_mohex max_time {max_time}"))?;
        }
        Ok(())
    }

    /// Make MoHex's board match `state`.
    ///
    /// The usual case is that `state` extends the position we last synced by a
    /// move or two, so we send just those — MoHex keeps its search tree, which
    /// is worth real strength. Otherwise (a new game, an undo, a handle we've
    /// never seen) we replay from an empty board. Hex is order-independent —
    /// no captures, and the winner depends only on the stones — so any replay
    /// order gives the same position; we alternate colours anyway so MoHex's
    /// move history looks like a real game.
    pub fn set_position(
        &mut self,
        state: &HexState,
        last: Option<&HexState>,
    ) -> anyhow::Result<()> {
        let empty = HexState::default();
        let (prev, reset) = match last.filter(|p| extends(p, state)) {
            Some(prev) => (prev, false),
            None => (&empty, true),
        };
        if reset {
            self.clear_board()?;
        }
        for (player, mv) in replay_moves(prev, state) {
            self.play(player, &mv)?;
        }
        Ok(())
    }

    /// MoHex's verdict on the current position: `Some(1)` black, `Some(2)`
    /// white, `None` if neither side has connected yet.
    pub fn winner(&mut self) -> anyhow::Result<Option<u8>> {
        match self
            .send("final_score")?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
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

/// Is `state` reachable from `prev` by adding stones? (Then MoHex only needs the
/// new ones, and keeps its search tree.)
fn extends(prev: &HexState, state: &HexState) -> bool {
    prev.ply <= state.ply
        && prev.black.andnot(state.black).is_empty()
        && prev.white.andnot(state.white).is_empty()
}

/// The stones `state` has and `prev` doesn't, as plays. Alternates colours where
/// it can and plays out whatever is left over otherwise (a swap leaves the
/// counts lopsided). Hex is order-independent — no captures, and the winner
/// depends only on the stones — so any order reproduces the position; we
/// alternate so MoHex's move history looks like a real game.
fn replay_moves(prev: &HexState, state: &HexState) -> Vec<(u8, HexMove)> {
    let mut pools = [
        state.black.andnot(prev.black).iter().collect::<Vec<_>>(),
        state.white.andnot(prev.white).iter().collect::<Vec<_>>(),
    ];
    let mut moves = Vec::with_capacity(pools[0].len() + pools[1].len());
    let mut side = prev.to_move;
    while !pools[0].is_empty() || !pools[1].is_empty() {
        if pools[side as usize - 1].is_empty() {
            side = 3 - side;
        }
        let i = pools[side as usize - 1].pop().expect("a non-empty pool");
        moves.push((
            side,
            HexMove::Play {
                r: (i / N) as u8,
                c: (i % N) as u8,
            },
        ));
        side = 3 - side;
    }
    moves
}

fn color_name(player: u8) -> &'static str {
    if player == 1 {
        "black"
    } else {
        "white"
    }
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
    Ok(Some(HexMove::Play {
        r: r as u8,
        c: c as u8,
    }))
}

// ── The interactive opponent ─────────────────────────────────────────────────

/// The MoHex process the engine server plays through, plus the position we last
/// synced to it (so the common case is a one-move update that keeps its tree).
struct Session {
    mohex: Mohex,
    last: Option<HexState>,
    default_playouts: u32,
    default_seconds: f64,
}

static SESSION: std::sync::OnceLock<std::sync::Mutex<Session>> = std::sync::OnceLock::new();

/// Spawn the MoHex that answers `RequestMove(player = EXTERNAL)`, with the
/// default per-move budget (a request may override it). One process, serialised
/// by a mutex — MoHex is a single GTP conversation, and interactive play is one
/// move at a time.
pub fn install(bin: &str, playouts: u32, seconds: f64) -> anyhow::Result<()> {
    let mohex = Mohex::new(bin, playouts, seconds)?;
    SESSION
        .set(std::sync::Mutex::new(Session {
            mohex,
            last: None,
            default_playouts: playouts,
            default_seconds: seconds,
        }))
        .map_err(|_| anyhow::anyhow!("mohex opponent already installed"))?;
    Ok(())
}

pub fn is_installed() -> bool {
    SESSION.get().is_some()
}

/// MoHex's move for `state`, or `None` if no MoHex was installed. `millis` /
/// `visits` override the startup budget when set.
pub fn choose_move(
    state: &HexState,
    millis: Option<u32>,
    visits: Option<u32>,
) -> Option<anyhow::Result<HexMove>> {
    let cell = SESSION.get()?;
    let mut guard = match cell.lock() {
        Ok(g) => g,
        // A previous call panicked mid-conversation, so the GTP stream may be
        // desynced; keep going but force a full resync.
        Err(poisoned) => {
            let mut g = poisoned.into_inner();
            g.last = None;
            g
        }
    };
    let session = &mut *guard;
    Some((|| {
        let playouts = visits.unwrap_or(session.default_playouts);
        let seconds = millis.map_or(session.default_seconds, |ms| f64::from(ms) / 1000.0);
        session.mohex.set_budget(playouts, seconds)?;
        session.mohex.set_position(state, session.last.as_ref())?;
        session.last = Some(*state);
        let mv = session
            .mohex
            .genmove(state.to_move)?
            // MoHex resigns only when it has proven the game lost; the UI has
            // no resignation concept, so play on with the placeholder move.
            .unwrap_or_else(|| crate::game::placeholder_move(state));
        session.last = Some(Hex::apply(state, &mv));
        Ok(mv)
    })())
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
        anyhow::ensure!(
            outcome.is_terminal() && outcome != Outcome::Draw,
            "hex has no draws"
        );
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

    fn play(state: &HexState, cell: &str) -> HexState {
        let (r, c) = crate::game::parse_cell(cell).unwrap();
        Hex::apply(
            state,
            &HexMove::Play {
                r: r as u8,
                c: c as u8,
            },
        )
    }

    #[test]
    fn syncing_an_extended_position_sends_only_the_new_stones() {
        let mut s = Hex::initial_state();
        for cell in ["g7", "d4", "h7", "e4"] {
            s = play(&s, cell);
        }
        let after = play(&play(&s, "i7"), "f4");

        assert!(extends(&s, &after));
        let moves = replay_moves(&s, &after);
        assert_eq!(moves.len(), 2, "only the two new stones");
        assert_eq!(moves[0].0, s.to_move, "the side to move plays first");
        assert_eq!(moves[1].0, 3 - s.to_move);

        // A full replay reproduces every stone, alternating from Black.
        let all = replay_moves(&HexState::default(), &after);
        assert_eq!(all.len(), 6);
        assert_eq!(all.iter().filter(|(p, _)| *p == 1).count(), 3);
        let mut rebuilt = Hex::initial_state();
        for (_, mv) in &all {
            rebuilt = Hex::apply(&rebuilt, mv);
        }
        assert_eq!(rebuilt.black, after.black);
        assert_eq!(rebuilt.white, after.white);
    }

    #[test]
    fn a_position_that_removes_or_recolours_a_stone_forces_a_resync() {
        let base = play(&play(&Hex::initial_state(), "g7"), "d4");
        assert!(!extends(&base, &Hex::initial_state()), "fewer stones");
        // Same cell, other colour: white g7 instead of black.
        let mut recoloured = Hex::initial_state();
        recoloured.white = base.black;
        recoloured.black = base.white;
        recoloured.ply = base.ply;
        assert!(!extends(&base, &recoloured));
    }

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
