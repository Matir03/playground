"""Generate behavioral-cloning data from perfect-bot vs random-net games.

Drives a running Connect 4 engine (and the model server it points at) over
gRPC. Each game: the perfect solver plays one side, the random NN the other,
from a randomized opening for diversity. We record **only the perfect bot's**
moves — (board, chosen column, eventual game result) — as the training targets.

Requires the engine (default :7100) and a model server (its --model-uri) to be
running.
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


def _chosen_col(stub, req) -> int:
    col = None
    for prog in stub.RequestMove(req):
        if prog.HasField("chosen"):
            col = prog.chosen.payload[0]
    if col is None:
        raise RuntimeError("RequestMove returned no chosen move")
    return int(col)


def _apply(stub, handle, col):
    return stub.ApplyMove(
        e.ApplyMoveRequest(handle=handle, move=g.Move(game_id=CONNECT4, payload=bytes([col])))
    )


def generate(engine_addr: str, games: int, max_opening: int, nn_visits: int, seed: int):
    rng = random.Random(seed)
    stub = eg.EngineServiceStub(grpc.insecure_channel(engine_addr))

    states: list[np.ndarray] = []
    cols: list[int] = []
    values: list[float] = []

    for gi in range(games):
        handle = stub.NewGame(e.NewGameRequest(game_id=CONNECT4))
        state = stub.GetState(handle)

        # Randomized opening for position diversity (both engines are otherwise
        # deterministic, so without this every game would be identical).
        for _ in range(rng.randint(0, max_opening)):
            if state.outcome != g.OUTCOME_IN_PROGRESS:
                break
            legal = connect4_legal_columns(state.payload)
            if not legal:
                break
            state = _apply(stub, handle, rng.choice(legal))

        perfect_side = rng.choice([g.PLAYER_FIRST, g.PLAYER_SECOND])
        game_records: list[tuple[bytes, int]] = []

        while state.outcome == g.OUTCOME_IN_PROGRESS:
            if state.to_move == perfect_side:
                col = _chosen_col(
                    stub, e.RequestMoveRequest(handle=handle, player=e.ENGINE_PLAYER_SOLVER)
                )
                game_records.append((bytes(state.payload), col))  # perfect-bot data only
            else:
                col = _chosen_col(
                    stub,
                    e.RequestMoveRequest(
                        handle=handle,
                        player=e.ENGINE_PLAYER_NN,
                        budget=e.SearchBudget(max_visits=nn_visits),
                    ),
                )
            state = _apply(stub, handle, col)

        # Game result, from the perfect bot's perspective (= side to move at
        # every recorded position).
        oc = state.outcome
        if oc == g.OUTCOME_DRAW:
            v = 0.0
        elif (oc == g.OUTCOME_FIRST_WINS and perfect_side == g.PLAYER_FIRST) or (
            oc == g.OUTCOME_SECOND_WINS and perfect_side == g.PLAYER_SECOND
        ):
            v = 1.0
        else:
            v = -1.0

        for payload, col in game_records:
            states.append(np.frombuffer(payload, dtype=np.uint8))
            cols.append(col)
            values.append(v)

        if (gi + 1) % 25 == 0:
            print(f"  {gi + 1}/{games} games, {len(cols)} positions")

    if not states:
        return np.zeros((0, 44), np.uint8), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return (
        np.stack(states).astype(np.uint8),
        np.array(cols, np.int64),
        np.array(values, np.float32),
    )
