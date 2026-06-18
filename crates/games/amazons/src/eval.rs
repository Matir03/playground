//! Head-to-head match between two checkpoints for ELO tracking. Plays A vs B over
//! many games (alternating who moves first, argmax moves after a sampled opening,
//! fixed node budget) and reports A's score. Threaded with a two-model batched
//! inference server (each request tagged A or B), mirroring `go9/eval.rs`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::{self, AmazonsMove, AmazonsState};
use crate::mcts::{run_mcts, Evaluator, Params};
use crate::moves::POLICY_DIM;
use crate::net::LocalModel;

type EvalOut = (f32, Vec<f32>);

struct Req {
    is_a: bool,
    state: AmazonsState,
    resp: SyncSender<EvalOut>,
}

thread_local! {
    static RESP: (SyncSender<EvalOut>, Receiver<EvalOut>) = sync_channel(1);
}

struct MatchEval<'a> {
    req_tx: &'a Sender<Req>,
    is_a: bool,
}

impl Evaluator for MatchEval<'_> {
    fn eval(&self, state: &AmazonsState) -> anyhow::Result<EvalOut> {
        RESP.with(|(tx, rx)| {
            self.req_tx
                .send(Req { is_a: self.is_a, state: state.clone(), resp: tx.clone() })
                .map_err(|_| anyhow::anyhow!("match server gone"))?;
            rx.recv().map_err(|_| anyhow::anyhow!("match response dropped"))
        })
    }
}

/// Two-model batched server: partition pending requests by A/B, one forward each.
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
        let mut a_idx = vec![];
        let mut b_idx = vec![];
        for (i, r) in batch.iter().enumerate() {
            if r.is_a {
                a_idx.push(i)
            } else {
                b_idx.push(i)
            }
        }
        let mut out: Vec<Option<EvalOut>> = (0..batch.len()).map(|_| None).collect();
        for (model, idxs) in [(a, &a_idx), (b, &b_idx)] {
            if idxs.is_empty() {
                continue;
            }
            let states: Vec<&AmazonsState> = idxs.iter().map(|&i| &batch[i].state).collect();
            let (vals, logits) = model.value_policy_batch(&states).expect("match forward failed");
            for (slot, (&i, v)) in idxs.iter().zip(vals.iter()).enumerate() {
                let pol = logits[slot * POLICY_DIM..(slot + 1) * POLICY_DIM].to_vec();
                out[i] = Some((*v, pol));
            }
        }
        for (req, res) in batch.into_iter().zip(out) {
            let _ = req.resp.send(res.unwrap());
        }
    }
}

fn pick(moves: &[(AmazonsMove, u32)], sample: bool, rng: &mut ChaCha8Rng) -> AmazonsMove {
    let total: u32 = moves.iter().map(|&(_, n)| n).sum();
    if sample && total > 0 {
        let r = rng.gen_range(0..total);
        let mut acc = 0u32;
        for &(m, n) in moves {
            acc += n;
            if r < acc {
                return m;
            }
        }
    }
    moves.iter().max_by_key(|&&(_, n)| n).map(|&(m, _)| m).unwrap()
}

/// Play one game (A moves first iff `a_first`); return A's score (1/0.5/0).
fn play_one(req_tx: &Sender<Req>, a_first: bool, p: &Params, seed: u64) -> f32 {
    let mut state = AmazonsState::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ply = 0u32;
    while game::has_any_move(&state) && ply < p.max_ply {
        let first_to_move = state.to_move == 1;
        let is_a = first_to_move == a_first;
        let eval = MatchEval { req_tx, is_a };
        let moves = run_mcts(&eval, &state, p, &mut rng).expect("mcts");
        let mv = pick(&moves, ply < p.temp_moves, &mut rng);
        state = game::apply(&state, &mv);
        ply += 1;
    }
    // Whoever is to move with no legal move has lost. (If we hit the cap, draw.)
    if game::has_any_move(&state) {
        return 0.5;
    }
    let loser = state.to_move;
    let a_side = if a_first { 1 } else { 2 };
    if loser != a_side {
        1.0
    } else {
        0.0
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
        "amazons eval-match: {games} games, {threads} threads, node-budget {}, max-ply {}\n  A={ckpt_a}\n  B={ckpt_b}",
        params.node_budget, params.max_ply,
    );

    let next = AtomicUsize::new(0);
    let score_x2 = AtomicU64::new(0);
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
