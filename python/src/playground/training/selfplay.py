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

from .connect4 import COLS, Board


class _Node:
    __slots__ = ("board", "P", "N", "W", "children", "expanded", "terminal", "term_v")

    def __init__(self, board: Board) -> None:
        self.board = board
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.children: dict[int, _Node] = {}
        self.expanded = False
        self.terminal = False
        self.term_v = 0.0


def _expand(node: _Node, net) -> float:
    """Expand `node`; return the value for the side to move."""
    term, tv = node.board.terminal()
    if term:
        node.terminal = True
        node.term_v = tv
        return tv
    planes = torch.from_numpy(node.board.planes()).unsqueeze(0)
    with torch.no_grad():
        v, probs = net.policy_value(planes)
    v = float(v[0])
    probs = probs[0].numpy()
    legal = node.board.legal()
    s = float(sum(probs[c] for c in legal)) or 1.0
    for c in legal:
        node.P[c] = float(probs[c]) / s
        node.N[c] = 0
        node.W[c] = 0.0
    node.expanded = True
    return v


def _select(node: _Node, c_puct: float) -> int:
    total = sum(node.N.values())
    sqrt_total = (total + 1) ** 0.5
    best, best_score = -1, -1e30
    for c in node.P:
        q = node.W[c] / node.N[c] if node.N[c] > 0 else 0.0
        u = c_puct * node.P[c] * sqrt_total / (1 + node.N[c])
        if q + u > best_score:
            best_score, best = q + u, c
    return best


def run_mcts(
    board: Board,
    net,
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_eps: float,
    rng,
    move_ms: float = 0.0,
) -> np.ndarray:
    """PUCT MCTS. Runs a fixed `sims` count, or — if `move_ms > 0` — uncapped
    rollouts until the per-move time budget (ms) is spent."""
    root = _Node(board)
    _expand(root, net)
    legal = list(root.P)
    if dir_eps > 0 and legal:
        noise = rng.dirichlet([dir_alpha] * len(legal))
        for i, c in enumerate(legal):
            root.P[c] = (1 - dir_eps) * root.P[c] + dir_eps * noise[i]

    def simulate() -> None:
        node, path = root, []
        while node.expanded and not node.terminal:
            c = _select(node, c_puct)
            path.append((node, c))
            if c not in node.children:
                node.children[c] = _Node(node.board.play(c))
            node = node.children[c]
        v = node.term_v if node.terminal else _expand(node, net)
        for n, c in reversed(path):
            v = -v
            n.N[c] += 1
            n.W[c] += v

    if move_ms > 0:
        deadline = time.perf_counter() + move_ms / 1000.0
        while time.perf_counter() < deadline:
            simulate()
    else:
        for _ in range(sims):
            simulate()

    counts = np.zeros(COLS, dtype=np.float64)
    for c, n in root.N.items():
        counts[c] = n
    return counts


# ── Parallel self-play workers ────────────────────────────────────────────────
# Game generation is embarrassingly parallel, so we fan it out over a process
# pool. Each worker loads the current net once (via an initializer) and plays
# whole games. Crucially each worker pins torch to a single thread: the MCTS does
# batch-1 forwards on a tiny 6×7 board, where torch's intra-op threading is pure
# overhead (same lesson as the in-process candle engine). N single-threaded
# workers therefore scale far better than one many-threaded process.

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
        _WORKER["net"], p["sims"], p["c_puct"], p["dir_alpha"], p["dir_eps"],
        p["temp_moves"], rng, p["move_ms"],
    )


def self_play_game(net, sims, c_puct, dir_alpha, dir_eps, temp_moves, rng, move_ms=0.0):
    board = Board()
    examples = []  # (planes, policy(7), to_move)
    total_sims = 0
    while not board.terminal()[0]:
        counts = run_mcts(board, net, sims, c_puct, dir_alpha, dir_eps, rng, move_ms)
        total_sims += int(counts.sum())
        policy = counts / counts.sum()
        examples.append((board.planes(), policy.astype(np.float32), board.to_move))
        if board.ply < temp_moves:
            col = int(rng.choice(COLS, p=policy))
        else:
            col = int(counts.argmax())
        board = board.play(col)

    w = board.winner()
    out = []
    for planes, policy, to_move in examples:
        z = 0.0 if w == 0 else (1.0 if w == to_move else -1.0)
        out.append((planes, policy, np.float32(z)))
    avg_sims = total_sims / max(1, len(examples))
    return out, avg_sims


def _greedy_policy_move(net, board: Board) -> int:
    planes = torch.from_numpy(board.planes()).unsqueeze(0)
    with torch.no_grad():
        _, probs = net.policy_value(planes)
    probs = probs[0].numpy()
    legal = board.legal()
    return max(legal, key=lambda c: probs[c])


def eval_vs(net, opponent, games: int, sims: int, rng) -> float:
    """Win rate of `net` (MCTS) vs `opponent` (greedy policy), random openings."""
    wins = draws = 0
    for g in range(games):
        board = Board()
        net_first = g % 2 == 0
        if board.legal():
            board = board.play(rng.choice(board.legal()))  # random opening for variety
        while not board.terminal()[0]:
            net_turn = (board.to_move == 1) == net_first
            if net_turn:
                counts = run_mcts(board, net, sims, 1.4, 0.0, 0.0, rng)
                col = int(counts.argmax())
            else:
                col = _greedy_policy_move(opponent, board)
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
) -> str:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
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
        if workers > 1:
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
            for _ in range(games_per_gen):
                ex, avg_sims = self_play_game(
                    net, sims, c_puct, dir_alpha, dir_eps, temp_moves, rng, move_ms
                )
                gen_examples += ex
                sims_acc += avg_sims
        avg_sims = sims_acc / max(1, games_per_gen)
        window.append(gen_examples)
        data = [ex for g in window for ex in g]

        planes = torch.from_numpy(np.stack([d[0] for d in data]))
        target_pi = torch.from_numpy(np.stack([d[1] for d in data]))
        target_z = torch.from_numpy(np.array([d[2] for d in data], dtype=np.float32))

        net.train()
        idx = np.arange(len(data))
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                b = torch.from_numpy(idx[i : i + batch_size])
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
        gen_cfg = {"arch": arch_of(net), "generation": gen_num, "winrate_vs_random": wr}
        save_checkpoint(net, os.path.join(gen_dir, "checkpoint.safetensors"))
        with open(os.path.join(gen_dir, "config.json"), "w") as f:
            json.dump(gen_cfg, f, indent=2)
        save_checkpoint(net, ckpt)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(gen_cfg, f, indent=2)
        print(
            f"gen {gen_num} (+{gen + 1}/{generations})  games {games_per_gen}  "
            f"examples {len(data)}  avg_sims/move {avg_sims:.0f}  "
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
