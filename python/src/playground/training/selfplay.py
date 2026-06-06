"""Pure self-play training (AlphaZero-style) for a Connect 4 net, from scratch.

No solver involvement: the net plays itself with a PUCT MCTS (Dirichlet noise at
the root, temperature sampling early), and trains on its own games — policy
target = the MCTS visit distribution, value target = the game outcome. Iterated
over generations, the net bootstraps from random play.

MCTS runs in-process over concrete Connect 4 states (connect4.py) using the
net's policy + value, which for this game is equivalent to the engine's served
MCTS (reify is a structural column decode).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.net import (
    ConnectFourModel,
    ConnectFourResNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)

from . import mp_worker
from .connect4 import COLS, Board
from .mcts_core import run_mcts, self_play_game  # torch-free search + game loop

PLANES = 2 * 6 * 7


def _net_evaluate(net):
    """A single-board `evaluate` closure backed by one forward on `net`'s device."""
    dev = next(net.parameters()).device

    def evaluate(board):
        planes = torch.from_numpy(board.planes()).unsqueeze(0).to(dev)
        with torch.no_grad():
            v, probs = net.policy_value(planes)
        return float(v[0]), probs[0].cpu().numpy()

    return evaluate


# ── Multiprocessing self-play with a central GPU inference server ─────────────
# The GIL serialises Python MCTS tree work, capping a threaded batched evaluator.
# Instead run the games in worker PROCESSES (true-parallel tree work) and keep a
# single inference server — this (main) process, which owns the GPU net — that
# batches their leaf evaluations. IPC is shared memory: a request-flag array the
# server polls (a worker writes its planes, then raises its flag) and a per-slot
# Event the worker blocks on (the server sets it after the forward). No GIL
# contention across workers, and no weight syncing (the server holds the live net).


def _generate_games_mp(net, device, n_games, params, n_workers, max_batch, base_seed):
    """Generate `n_games` across `n_workers` processes fed by a batched GPU server."""
    ctx = multiprocessing.get_context("spawn")
    req_planes = ctx.RawArray("f", n_workers * PLANES)
    req_ready = ctx.RawArray("b", n_workers)
    val_arr = ctx.RawArray("f", n_workers)
    prob_arr = ctx.RawArray("f", n_workers * COLS)
    resp_events = [ctx.Event() for _ in range(n_workers)]
    counter = ctx.Value("i", 0)   # next game index workers atomically claim
    finished = ctx.Value("i", 0)  # games completed
    result_q = ctx.Queue()

    procs = []
    for slot in range(n_workers):
        p = ctx.Process(
            target=mp_worker.worker,
            args=(slot, n_games, base_seed, counter, finished, req_planes, req_ready,
                  val_arr, prob_arr, resp_events[slot], result_q, n_workers, params),
            daemon=True,
        )
        p.start()
        procs.append(p)

    ready_np = np.frombuffer(req_ready, dtype=np.int8)
    req_np = np.frombuffer(req_planes, dtype=np.float32).reshape(n_workers, 2, 6, 7)
    val_np = np.frombuffer(val_arr, dtype=np.float32)
    prob_np = np.frombuffer(prob_arr, dtype=np.float32).reshape(n_workers, COLS)

    nbatch = nsamp = 0
    while finished.value < n_games:
        # Snapshot the flags (workers set them concurrently); per-byte int8 reads
        # are atomic, so the copy is consistent and nonzero can't race.
        slots = np.nonzero(ready_np.copy())[0]
        if slots.size == 0:
            time.sleep(0.0002)
            continue
        if slots.size > max_batch:
            slots = slots[:max_batch]
        ready_np[slots] = 0  # clear before serving so workers can re-request
        planes = torch.from_numpy(req_np[slots]).to(device)
        with torch.no_grad():
            v, probs = net.policy_value(planes)
        val_np[slots] = v.cpu().numpy()
        prob_np[slots] = probs.cpu().numpy()
        nbatch += 1
        nsamp += int(slots.size)
        for s in slots:
            resp_events[int(s)].set()

    # Drain results before joining (a Queue with unread data deadlocks join).
    results: list = []
    for _ in range(n_workers):
        results.extend(result_q.get())
    for p in procs:
        p.join()
    return results, nsamp / max(1, nbatch)
    avg_batch = ev.nsamples / max(1, ev.nbatches)
    return results, avg_batch


# ── Process-pool self-play (CPU fallback) ─────────────────────────────────────
# When no GPU/concurrency is used, fan games out over a process pool, each worker
# pinned to one torch thread (batch-1 forwards on the tiny board don't benefit
# from intra-op threading).

_WORKER = {}


def _worker_init(ckpt_path: str, channels: int, blocks: int, params: dict) -> None:
    torch.set_num_threads(1)
    net = ConnectFourResNet(channels=channels, blocks=blocks)
    load_checkpoint(net, ckpt_path)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    _WORKER["net"] = net
    _WORKER["params"] = params


def _worker_play(seed: int):
    p = _WORKER["params"]
    rng = np.random.default_rng(seed)
    return self_play_game(
        _net_evaluate(_WORKER["net"]), p["sims"], p["c_puct"], p["dir_alpha"],
        p["dir_eps"], p["temp_moves"], rng, p["move_ms"],
    )


def _greedy_policy_move(evaluate, board: Board) -> int:
    _, probs = evaluate(board)
    legal = board.legal()
    return max(legal, key=lambda c: probs[c])


def eval_vs(net, opponent, games: int, sims: int, rng) -> float:
    """Win rate of `net` (MCTS) vs `opponent` (greedy policy), random openings."""
    net_eval = _net_evaluate(net)
    opp_eval = _net_evaluate(opponent)
    wins = draws = 0
    for g in range(games):
        board = Board()
        net_first = g % 2 == 0
        if board.legal():
            board = board.play(rng.choice(board.legal()))  # random opening for variety
        while not board.terminal()[0]:
            net_turn = (board.to_move == 1) == net_first
            if net_turn:
                counts, _ = run_mcts(board, net_eval, sims, 1.4, 0.0, 0.0, rng)
                col = int(counts.argmax())
            else:
                col = _greedy_policy_move(opp_eval, board)
            board = board.play(col)
        w = board.winner()
        if w == 0:
            draws += 1
        elif (w == 1) == net_first:
            wins += 1
    return (wins + 0.5 * draws) / games


def selfplay_train(
    out_dir: str,
    channels: int,
    blocks: int,
    generations: int,
    games_per_gen: int,
    sims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    buffer_gens: int,
    c_puct: float,
    dir_alpha: float,
    dir_eps: float,
    temp_moves: int,
    eval_games: int,
    seed: int,
    move_ms: float = 0.0,
    init: str | None = None,
    workers: int = 1,
    device: str = "auto",
    concurrent: int = 1,
    max_batch: int = 256,
) -> str:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ConnectFourResNet(channels=channels, blocks=blocks)
    # Resume support: if the output dir already holds generations, continue
    # numbering from the latest (and warm-start from it unless --init overrides).
    # Makes the run idempotent across interruptions (e.g. a host reboot).
    import glob

    existing = sorted(glob.glob(os.path.join(out_dir, "gen*")))
    start_gen = int(os.path.basename(existing[-1])[3:]) if existing else 0
    if init:
        load_checkpoint(net, init)  # warm-start from an explicit checkpoint
        print(f"warm-started from {init}", flush=True)
    elif start_gen > 0:
        resume = os.path.join(out_dir, f"gen{start_gen:03d}", "checkpoint.safetensors")
        load_checkpoint(net, resume)
        print(f"resumed from {resume} (continuing after gen {start_gen})", flush=True)
    net.to(device)
    print(f"device: {device}  concurrent-games: {concurrent}  max-batch: {max_batch}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # Optional baseline opponent for a progress signal.
    baseline = None
    base_path = "data/models/connect4/random/checkpoint.safetensors"
    if eval_games > 0 and os.path.exists(base_path):
        baseline = ConnectFourModel()
        load_checkpoint(baseline, base_path)
        baseline.eval()

    window: deque = deque(maxlen=buffer_gens)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")

    worker_ckpt = os.path.join(out_dir, "_worker.safetensors")
    play_params = {
        "sims": sims, "c_puct": c_puct, "dir_alpha": dir_alpha, "dir_eps": dir_eps,
        "temp_moves": temp_moves, "move_ms": move_ms,
    }

    for gen in range(generations):
        net.eval()
        t0 = time.time()
        gen_examples = []
        sims_acc = 0.0
        avg_batch = 0.0
        if concurrent > 1:
            # Worker PROCESSES (true-parallel tree work) fed by the central GPU
            # inference server — avoids the GIL cap of the threaded approach.
            results, avg_batch = _generate_games_mp(
                net, device, games_per_gen, play_params, concurrent, max_batch,
                seed * 1_000_000 + gen * 10_000,
            )
            for ex, avg_sims in results:
                gen_examples += ex
                sims_acc += avg_sims
        elif workers > 1:
            # Fan game generation out over a process pool reloading the current
            # weights. New pool per generation so workers pick up the latest net.
            from concurrent.futures import ProcessPoolExecutor

            save_checkpoint(net, worker_ckpt)
            seeds = [seed * 1_000_000 + gen * 10_000 + i for i in range(games_per_gen)]
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(worker_ckpt, channels, blocks, play_params),
            ) as pool:
                for ex, avg_sims in pool.map(_worker_play, seeds):
                    gen_examples += ex
                    sims_acc += avg_sims
        else:
            evaluate = _net_evaluate(net)
            for _ in range(games_per_gen):
                ex, avg_sims = self_play_game(
                    evaluate, sims, c_puct, dir_alpha, dir_eps, temp_moves, rng, move_ms
                )
                gen_examples += ex
                sims_acc += avg_sims
        avg_sims = sims_acc / max(1, games_per_gen)
        window.append(gen_examples)
        data = [ex for g in window for ex in g]

        planes = torch.from_numpy(np.stack([d[0] for d in data])).to(device)
        target_pi = torch.from_numpy(np.stack([d[1] for d in data])).to(device)
        target_z = torch.from_numpy(np.array([d[2] for d in data], dtype=np.float32)).to(device)

        net.train()
        idx = np.arange(len(data))
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                b = torch.from_numpy(idx[i : i + batch_size]).to(device)
                v_pred, logits = net(planes[b])
                loss_v = F.mse_loss(v_pred, target_z[b])
                loss_p = -(target_pi[b] * F.log_softmax(logits, dim=1)).sum(1).mean()
                loss = loss_v + loss_p
                opt.zero_grad()
                loss.backward()
                opt.step()

        net.eval()
        wr = eval_vs(net, baseline, eval_games, sims, rng) if baseline is not None else float("nan")

        # Keep every generation's checkpoint separately (for the rating curve),
        # and mirror the latest into the top-level dir for serving convenience.
        gen_num = start_gen + gen + 1
        gen_dir = os.path.join(out_dir, f"gen{gen_num:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        # Use null (not NaN) for an unmeasured winrate: NaN is invalid JSON and
        # the Rust engine's config reader rejects it.
        wr_json = None if wr != wr else wr
        gen_cfg = {"arch": arch_of(net), "generation": gen_num, "winrate_vs_random": wr_json}
        net.to("cpu")  # safetensors saves CPU tensors
        save_checkpoint(net, os.path.join(gen_dir, "checkpoint.safetensors"))
        with open(os.path.join(gen_dir, "config.json"), "w") as f:
            json.dump(gen_cfg, f, indent=2)
        save_checkpoint(net, ckpt)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(gen_cfg, f, indent=2)
        net.to(device)
        batch_note = f"  avg_batch {avg_batch:.0f}" if concurrent > 1 else ""
        print(
            f"gen {gen_num} (+{gen + 1}/{generations})  games {games_per_gen}  "
            f"examples {len(data)}  avg_sims/move {avg_sims:.0f}{batch_note}  "
            f"winrate_vs_random {wr:.2f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {"kind": "selfplay", "arch": arch_of(net), "generations": generations,
             "games_per_gen": games_per_gen, "sims": sims, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
            f,
            indent=2,
        )
    print(f"saved {ckpt}")
    return ckpt
