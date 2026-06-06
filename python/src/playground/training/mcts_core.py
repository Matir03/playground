"""Torch-free MCTS + self-play game loop.

Kept free of any torch import so worker processes (which run the search but not
inference) stay lightweight under the 'spawn' start method — they import only
numpy + the rules here, while a central server process owns the GPU net. The
search is decoupled from inference via an `evaluate(board) -> (value, probs)`
callable, so the same code drives a direct forward, a threaded batched
evaluator, or the cross-process inference server.
"""

from __future__ import annotations

import time

import numpy as np

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


def _expand(node: _Node, evaluate) -> float:
    """Expand `node`; return the value for the side to move.

    `evaluate(board) -> (value: float, probs: np.ndarray[COLS])`.
    """
    term, tv = node.board.terminal()
    if term:
        node.terminal = True
        node.term_v = tv
        return tv
    v, probs = evaluate(node.board)
    legal = node.board.legal()
    s = float(sum(probs[c] for c in legal)) or 1.0
    for c in legal:
        node.P[c] = float(probs[c]) / s
        node.N[c] = 0
        node.W[c] = 0.0
    node.expanded = True
    return float(v)


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


def run_mcts(board, evaluate, sims, c_puct, dir_alpha, dir_eps, rng, move_ms=0.0):
    """PUCT MCTS. Runs a fixed `sims` count, or — if `move_ms > 0` — uncapped
    rollouts until the per-move time budget (ms) is spent. Returns
    `(visit_counts[COLS], root_value)`, the root value being the search's value
    estimate for the side to move (visit-weighted mean of the children's Q)."""
    root = _Node(board)
    _expand(root, evaluate)
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
        v = node.term_v if node.terminal else _expand(node, evaluate)
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
    total_n = sum(root.N.values())
    root_value = (sum(root.W.values()) / total_n) if total_n > 0 else 0.0
    return counts, root_value


def self_play_game(evaluate, sims, c_puct, dir_alpha, dir_eps, temp_moves, rng, move_ms=0.0):
    """Play one self-play game; return (examples, avg_sims).

    Each example is (planes, policy_target[COLS], value_target). The policy
    target is the MCTS visit distribution; the value target is the MCTS root
    value at that position (the local search estimate for the side to move) —
    not the final game result.
    """
    board = Board()
    examples = []
    total_sims = 0
    while not board.terminal()[0]:
        counts, root_v = run_mcts(board, evaluate, sims, c_puct, dir_alpha, dir_eps, rng, move_ms)
        total_sims += int(counts.sum())
        policy = counts / counts.sum()
        examples.append((board.planes(), policy.astype(np.float32), np.float32(root_v)))
        if board.ply < temp_moves:
            col = int(rng.choice(COLS, p=policy))
        else:
            col = int(counts.argmax())
        board = board.play(col)

    avg_sims = total_sims / max(1, len(examples))
    return examples, avg_sims
