"""Generate behavioral-cloning data by annotating random-net self-play.

Games are played by the **random net against itself** (with randomized openings
for diversity). Every position is then annotated by the **perfect bot** — its
best move (policy target) and its evaluation of the position (value target).
We train on the perfect bot's annotation, never on the realized game scores, so
the value target is the solver's graded assessment rather than "who happened to
win this random game".

Requires a running engine (default :7100) that has the random NN model plus the
solver, and the model server(s) it points at.
"""

from __future__ import annotations

import random

import grpc
import numpy as np

from playground.gen.playground.engine import engine_pb2 as e
from playground.gen.playground.engine import engine_pb2_grpc as eg
from playground.gen.playground.game import game_pb2 as g
from playground.model_server.net import connect4_legal_columns

CONNECT4 = g.GAME_ID_CONNECT4


def _solver_eval(stub, handle) -> tuple[int, float]:
    """Perfect bot's (best move, evaluation for the side to move)."""
    col, val = None, 0.0
    for prog in stub.RequestMove(e.RequestMoveRequest(handle=handle, player=e.ENGINE_PLAYER_SOLVER)):
        if prog.HasField("stats"):
            val = prog.stats.best_value
        elif prog.HasField("chosen"):
            col = prog.chosen.payload[0]
    if col is None:
        raise RuntimeError("solver returned no move")
    return int(col), float(val)


def _nn_move(stub, handle, model: str, visits: int) -> int:
    col = None
    for prog in stub.RequestMove(
        e.RequestMoveRequest(
            handle=handle,
            player=e.ENGINE_PLAYER_NN,
            model=model,
            budget=e.SearchBudget(max_visits=visits),
        )
    ):
        if prog.HasField("chosen"):
            col = prog.chosen.payload[0]
    if col is None:
        raise RuntimeError("NN returned no move")
    return int(col)


def _apply(stub, handle, col):
    return stub.ApplyMove(
        e.ApplyMoveRequest(handle=handle, move=g.Move(game_id=CONNECT4, payload=bytes([col])))
    )


def generate(engine_addr: str, games: int, max_opening: int, nn_visits: int, nn_model: str, seed: int):
    rng = random.Random(seed)
    stub = eg.EngineServiceStub(grpc.insecure_channel(engine_addr))

    states: list[np.ndarray] = []
    cols: list[int] = []
    values: list[float] = []

    for gi in range(games):
        handle = stub.NewGame(e.NewGameRequest(game_id=CONNECT4))
        state = stub.GetState(handle)

        # Randomized opening so the deterministic self-play games differ.
        for _ in range(rng.randint(0, max_opening)):
            if state.outcome != g.OUTCOME_IN_PROGRESS:
                break
            legal = connect4_legal_columns(state.payload)
            if not legal:
                break
            state = _apply(stub, handle, rng.choice(legal))

        # Play out as random-net self-play, annotating each position with the
        # perfect bot's move + evaluation before the random net moves.
        while state.outcome == g.OUTCOME_IN_PROGRESS:
            sol_col, sol_val = _solver_eval(stub, handle)
            states.append(np.frombuffer(bytes(state.payload), dtype=np.uint8))
            cols.append(sol_col)
            values.append(sol_val)
            nn_col = _nn_move(stub, handle, nn_model, nn_visits)
            state = _apply(stub, handle, nn_col)

        if (gi + 1) % 25 == 0:
            print(f"  {gi + 1}/{games} games, {len(cols)} annotated positions")

    if not states:
        return np.zeros((0, 44), np.uint8), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return (
        np.stack(states).astype(np.uint8),
        np.array(cols, np.int64),
        np.array(values, np.float32),
    )
