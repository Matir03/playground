//! Absolute-strength eval vs GNU Go over GTP. The net moves via the **concrete**
//! `run_mcts` (value+policy, argmax, fixed node budget) — the same search self-play
//! uses — NOT the engine-server's latent reification search (which depends on the
//! `pol_next` reifier; with `--reif-weight 0` that head is stale, so a
//! reification-based eval would underrate the net). GNU Go plays via its GTP mode;
//! GNU Go also scores each finished game (it removes dead stones).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use playground_core::{Game, Outcome};

use crate::game::{area_outcome, Go9, Go9Move, Go9State};
use crate::mcts::{run_mcts, Params};
use crate::net::{action_to_move, LocalModel, ACTIONS};

const COLS: &[u8] = b"ABCDEFGHJ"; // GTP columns for 9×9 (skips 'I')

fn argmax(c: &[u32; ACTIONS]) -> usize {
    let mut best = 0;
    for i in 1..ACTIONS {
        if c[i] > c[best] {
            best = i;
        }
    }
    best
}

fn vertex_from_rc(r: usize, c: usize) -> String {
    format!("{}{}", COLS[c] as char, r + 1)
}

fn rc_from_vertex(v: &str) -> Option<(usize, usize)> {
    let b = v.as_bytes();
    let c = COLS.iter().position(|&x| x == b[0].to_ascii_uppercase())?;
    let r = v[1..].parse::<usize>().ok()?.checked_sub(1)?;
    Some((r, c))
}

/// Minimal GTP client over a `gnugo --mode gtp` subprocess.
struct Gtp {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Gtp {
    fn new(gnugo: &str, level: u32) -> anyhow::Result<Self> {
        let mut child = Command::new(gnugo)
            .args(["--mode", "gtp", "--level", &level.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Ok(Self { child, stdin, stdout })
    }

    /// Send a GTP command and return the response body (leading `=`/`?` stripped).
    fn send(&mut self, cmd: &str) -> anyhow::Result<String> {
        writeln!(self.stdin, "{cmd}")?;
        self.stdin.flush()?;
        let mut out: Vec<String> = Vec::new();
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                break; // EOF
            }
            let t = line.trim_end().to_string();
            if t.is_empty() {
                if !out.is_empty() {
                    break;
                }
                continue;
            }
            out.push(t);
        }
        let text = out.join("\n");
        Ok(text.get(1..).unwrap_or("").trim().to_string())
    }
}

impl Drop for Gtp {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.kill();
    }
}

/// Play `games` games of the net (concrete `run_mcts`, `node_budget` visits/move,
/// argmax) vs GNU Go at `level`; print and return the net's score in [0,1].
#[allow(clippy::too_many_arguments)]
pub fn run_gnugo_match(
    ckpt: &str,
    cuda: bool,
    gnugo: &str,
    level: u32,
    games: u32,
    node_budget: u32,
    c_puct: f32,
    max_ply: u32,
    seed: u64,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let model = LocalModel::load_on(ckpt, device)?;
    let eval = |s: &Go9State| model.policy_value(s);
    let params = Params {
        sims: 0, move_ms: 0.0, node_budget, c_puct,
        dir_alpha: 1.0, dir_eps: 0.0, temp_moves: 0, explore_eps: 0.0, max_ply,
        settle_plies: 0, resign_threshold: 0.0, resign_consecutive: 0, resign_playout_frac: 0.0,
    };
    eprintln!(
        "go9 vs GNU Go L{level}: {games} games, {node_budget} visits/move (concrete value+policy), \
         device {}, model {ckpt}",
        if cuda { "cuda" } else { "cpu" },
    );

    let mut score = 0.0f64;
    for g in 0..games {
        let net_black = g % 2 == 0; // alternate colors
        let mut gtp = Gtp::new(gnugo, level)?;
        gtp.send("boardsize 9")?;
        gtp.send("clear_board")?;
        gtp.send("komi 7.5")?;

        let mut state = Go9::initial_state();
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(g as u64));
        let mut ply = 0u32;
        let mut net_resign_win = false;
        while !Go9::outcome(&state).is_terminal() && ply < max_ply {
            let black_to_move = state.to_move == 1;
            let color = if black_to_move { "black" } else { "white" };
            if black_to_move == net_black {
                let (counts, _v, _n, _s) = run_mcts(&eval, &state, &params, &mut rng)?;
                let mv = action_to_move(argmax(&counts));
                state = Go9::apply(&state, &mv);
                let vtx = match mv {
                    Go9Move::Pass => "pass".to_string(),
                    Go9Move::Play { r, c } => vertex_from_rc(r as usize, c as usize),
                };
                gtp.send(&format!("play {color} {vtx}"))?;
            } else {
                let v = gtp.send(&format!("genmove {color}"))?;
                if v.eq_ignore_ascii_case("resign") {
                    net_resign_win = true;
                    break;
                }
                let mv = if v.eq_ignore_ascii_case("pass") {
                    Go9Move::Pass
                } else {
                    let (r, c) = rc_from_vertex(&v)
                        .ok_or_else(|| anyhow::anyhow!("bad GTP vertex {v:?}"))?;
                    Go9Move::Play { r: r as u8, c: c as u8 }
                };
                state = Go9::apply(&state, &mv);
            }
            ply += 1;
        }

        // Score: prefer GNU Go's final_score (it removes dead stones); else area score.
        let (net_won, detail) = if net_resign_win {
            (true, "gnugo resigned".to_string())
        } else {
            let fs = gtp.send("final_score").unwrap_or_default();
            let black_won = if fs.starts_with('B') {
                true
            } else if fs.starts_with('W') {
                false
            } else {
                matches!(area_outcome(&state), Outcome::FirstWins)
            };
            (black_won == net_black, fs)
        };
        let s = if net_won { 1.0 } else { 0.0 };
        score += s;
        eprintln!(
            "  game {g}: net={} {} -> {} ({detail})",
            if net_black { "B" } else { "W" },
            if ply >= max_ply { "[ply cap]" } else { "" },
            if net_won { "WIN" } else { "loss" },
        );
    }

    let net_score = score / games as f64;
    println!("net_vs_gnugo {net_score:.3}  ({games} games, {node_budget} visits, GNU Go L{level})");
    Ok(net_score)
}
