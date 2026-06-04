"""Play Connect 4 matches between two nets (each moving by its own MCTS).

Used to rate the self-play generations against each other and the baselines.
Reuses the in-process MCTS from training.selfplay (no Dirichlet noise, greedy
on visit counts); a short random opening per game gives variety.
"""

from __future__ import annotations

import json
import os

from playground.model_server.net import build_model, load_checkpoint
from playground.training.connect4 import Board
from playground.training.selfplay import run_mcts


def load_net(checkpoint: str):
    """Load a net, picking the architecture from a sibling config.json."""
    arch = None
    cfg = os.path.join(os.path.dirname(checkpoint), "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            arch = json.load(f).get("arch")
    net = build_model(arch)
    load_checkpoint(net, checkpoint)
    net.eval()
    return net


def play_game(net_a, net_b, sims: int, rng, a_first: bool, opening: int) -> float:
    """Score for net_a: 1 win, 0.5 draw, 0 loss."""
    board = Board()
    for _ in range(opening):
        if board.terminal()[0]:
            break
        board = board.play(int(rng.choice(board.legal())))
    while not board.terminal()[0]:
        a_turn = (board.to_move == 1) == a_first
        net = net_a if a_turn else net_b
        counts = run_mcts(board, net, sims, 1.4, 0.0, 0.0, rng)
        board = board.play(int(counts.argmax()))
    w = board.winner()
    if w == 0:
        return 0.5
    return 1.0 if (w == 1) == a_first else 0.0


def match(net_a, net_b, games: int, sims: int, rng) -> float:
    """Average score for net_a over `games` (alternating first move)."""
    total = 0.0
    for g in range(games):
        opening = int(rng.integers(1, 5))  # 1..4 random opening plies
        total += play_game(net_a, net_b, sims, rng, a_first=(g % 2 == 0), opening=opening)
    return total / games
