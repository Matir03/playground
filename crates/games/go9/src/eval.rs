//! Head-to-head match between two checkpoints, for ELO tracking. Plays A vs B
//! over many games (alternating colors, argmax moves, fixed node budget, turn
//! cap → area score) and reports A's score. Threaded with a two-model batched
//! inference server (each request tags A or B; the server batches per model) so
//! it uses the GPU efficiently — the same pattern as `selfplay.rs`'s server.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::Arc;
use std::time::Instant;

use playground_core::{Game, Outcome};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::{area_outcome, Go9, Go9State};
use crate::mcts::{run_mcts, Evaluator, Params};
use crate::net::{action_to_move, LocalModel, ACTIONS};

type EvalOut = (f32, [f32; ACTIONS]);

struct Req {
    is_a: bool,
    state: Go9State,
    resp: SyncSender<EvalOut>,
}

thread_local! {
    static RESP: (SyncSender<EvalOut>, Receiver<EvalOut>) = sync_channel(1);
}

/// Evaluator bound to one side's model (A or B) for a single move's search.
struct MatchEval<'a> {
    req_tx: &'a Sender<Req>,
    is_a: bool,
}

impl Evaluator for MatchEval<'_> {
    fn eval(&self, state: &Go9State) -> anyhow::Result<EvalOut> {
        RESP.with(|(tx, rx)| {
            self.req_tx
                .send(Req { is_a: self.is_a, state: state.clone(), resp: tx.clone() })
                .map_err(|_| anyhow::anyhow!("match server gone"))?;
            rx.recv().map_err(|_| anyhow::anyhow!("match response dropped"))
        })
    }
}

/// Two-model batched inference server: drain pending requests, batch the A-tagged
/// and B-tagged ones separately, one forward each, scatter the results.
fn server_loop(rx: Receiver<Req>, a: &LocalModel, b: &LocalModel, max_batch: usize) {
    loop {
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let mut batch = vec![first];
        while batch.len() < max_batch {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }
        // Partition into A and B, remembering each request's slot.
        let mut a_idx = vec![];
        let mut b_idx = vec![];
        for (i, r) in batch.iter().enumerate() {
            if r.is_a { a_idx.push(i) } else { b_idx.push(i) }
        }
        let mut out: Vec<Option<EvalOut>> = (0..batch.len()).map(|_| None).collect();
        for (model, idxs) in [(a, &a_idx), (b, &b_idx)] {
            if idxs.is_empty() {
                continue;
            }
            let states: Vec<&Go9State> = idxs.iter().map(|&i| &batch[i].state).collect();
            let res = model.policy_value_batch(&states).expect("match forward failed");
            for (slot, r) in idxs.iter().zip(res) {
                out[*slot] = Some(r);
            }
        }
        for (req, res) in batch.into_iter().zip(out) {
            let _ = req.resp.send(res.unwrap());
        }
    }
}

fn argmax(counts: &[u32; ACTIONS]) -> usize {
    let mut best = 0;
    for a in 1..ACTIONS {
        if counts[a] > counts[best] {
            best = a;
        }
    }
    best
}

/// Sample an action ~ the visit distribution (for opening diversity).
fn sample(counts: &[u32; ACTIONS], rng: &mut impl rand::Rng) -> usize {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return argmax(counts);
    }
    let r = rng.gen_range(0..total);
    let mut acc = 0u32;
    for a in 0..ACTIONS {
        acc += counts[a];
        if r < acc {
            return a;
        }
    }
    argmax(counts)
}

/// Play one game (A is black iff `a_is_black`); return A's score (1/0.5/0).
fn play_one(req_tx: &Sender<Req>, a_is_black: bool, p: &Params, seed: u64) -> f32 {
    let mut state = Go9::initial_state();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ply = 0u32;
    while !Go9::outcome(&state).is_terminal() && ply < p.max_ply {
        // Black is the side-to-move == 1; A plays black iff a_is_black.
        let black_to_move = state.to_move == 1;
        let is_a = black_to_move == a_is_black;
        let eval = MatchEval { req_tx, is_a };
        let (counts, _v, _n) = run_mcts(&eval, &state, p, &mut rng).expect("mcts");
        // Sample the opening for game diversity, then play the net's best move.
        let a = if ply < p.temp_moves { sample(&counts, &mut rng) } else { argmax(&counts) };
        let mv = action_to_move(a);
        state = Go9::apply(&state, &mv);
        ply += 1;
    }
    let outcome = match Go9::outcome(&state) {
        Outcome::InProgress => area_outcome(&state),
        o => o,
    };
    let a_color = if a_is_black { 1 } else { 2 };
    match outcome {
        Outcome::FirstWins => if a_color == 1 { 1.0 } else { 0.0 },
        Outcome::SecondWins => if a_color == 2 { 1.0 } else { 0.0 },
        _ => 0.5,
    }
}

/// Run a `games`-game match A vs B; print and return A's score in [0,1].
#[allow(clippy::too_many_arguments)]
pub fn run_match(
    ckpt_a: &str,
    ckpt_b: &str,
    cuda: bool,
    games: usize,
    params: Params,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
) -> anyhow::Result<f64> {
    let device = crate::net::pick_device(cuda)?;
    let a = Arc::new(LocalModel::load_on(ckpt_a, device.clone())?);
    let b = Arc::new(LocalModel::load_on(ckpt_b, device)?);
    eprintln!(
        "go9 eval-match: {games} games, {threads} threads, node-budget {}, max-ply {}\n  A={ckpt_a}\n  B={ckpt_b}",
        params.node_budget, params.max_ply,
    );

    let next = AtomicUsize::new(0);
    let score_x2 = AtomicU64::new(0); // sum of 2*score (win=2, draw=1, loss=0)
    let t0 = Instant::now();
    let (req_tx, req_rx) = channel::<Req>();
    std::thread::scope(|scope| {
        let (a, b) = (&*a, &*b);
        scope.spawn(move || server_loop(req_rx, a, b, max_batch));
        for _ in 0..threads {
            let req_tx = req_tx.clone();
            let next = &next;
            let score_x2 = &score_x2;
            let params = &params;
            scope.spawn(move || loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= games {
                    break;
                }
                let s = play_one(&req_tx, idx % 2 == 0, params, base_seed.wrapping_add(idx as u64));
                score_x2.fetch_add((s * 2.0) as u64, Ordering::Relaxed);
            });
        }
        drop(req_tx);
    });

    let a_score = score_x2.load(Ordering::Relaxed) as f64 / (2.0 * games as f64);
    println!("A_score {a_score:.4}  ({games} games, {:.1}s)", t0.elapsed().as_secs_f64());
    Ok(a_score)
}
