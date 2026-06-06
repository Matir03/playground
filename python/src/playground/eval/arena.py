"""Play Connect 4 matches between two "players" for rating.

A player is a callable `(board, rng) -> column`. Two kinds:
  - a net, moving by the in-process MCTS (training.selfplay.run_mcts), and
  - the perfect solver, moving via a running engine's RequestMove (SOLVER).

A short random opening per game gives variety (and can hand either side a
lost position, which keeps even the perfect solver's score below 100%).
"""

from __future__ import annotations

import json
import os

import numpy as np

from playground.model_server.net import build_model, load_checkpoint
from playground.training.connect4 import Board
from playground.training.selfplay import _net_evaluate, run_mcts


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


def net_player(net, sims: int):
    evaluate = _net_evaluate(net)

    def move(board: Board, rng) -> int:
        counts, _ = run_mcts(board, evaluate, sims, 1.4, 0.0, 0.0, rng)
        return int(counts.argmax())

    return move


def _board_payload(board: Board) -> bytes:
    # Matches crates/games/connect4/src/game.rs encode_state: 42 cells (row
    # major, row 0 bottom) + 2 little-endian ply bytes.
    return board.cells.astype(np.uint8).tobytes() + int(board.ply).to_bytes(2, "little")


def solver_player(engine_addr: str):
    """Perfect-solver player backed by a running engine (book + alpha-beta)."""
    import grpc

    from playground.gen.playground.engine import engine_pb2 as e
    from playground.gen.playground.engine import engine_pb2_grpc as eg
    from playground.gen.playground.game import game_pb2 as g

    stub = eg.EngineServiceStub(grpc.insecure_channel(engine_addr))

    def move(board: Board, rng) -> int:
        state = g.GameState(game_id=g.GAME_ID_CONNECT4, payload=_board_payload(board))
        handle = stub.NewGame(e.NewGameRequest(game_id=g.GAME_ID_CONNECT4, initial_state=state))
        for prog in stub.RequestMove(
            e.RequestMoveRequest(handle=handle, player=e.ENGINE_PLAYER_SOLVER)
        ):
            if prog.HasField("chosen"):
                return int(prog.chosen.payload[0])
        raise RuntimeError("solver returned no move")

    return move


def play_game(player_a, player_b, rng, a_first: bool, opening: int) -> float:
    """Score for player_a: 1 win, 0.5 draw, 0 loss."""
    board = Board()
    for _ in range(opening):
        if board.terminal()[0]:
            break
        board = board.play(int(rng.choice(board.legal())))
    while not board.terminal()[0]:
        a_turn = (board.to_move == 1) == a_first
        col = (player_a if a_turn else player_b)(board, rng)
        board = board.play(col)
    w = board.winner()
    if w == 0:
        return 0.5
    return 1.0 if (w == 1) == a_first else 0.0


def match(player_a, player_b, games: int, rng) -> float:
    """Average score for player_a over `games` (alternating first move)."""
    total = 0.0
    for gm in range(games):
        opening = int(rng.integers(1, 5))
        total += play_game(player_a, player_b, rng, a_first=(gm % 2 == 0), opening=opening)
    return total / games
