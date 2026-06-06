"""Self-play worker process (torch-free).

Runs the MCTS for whole games and offloads every leaf evaluation to the central
inference server (the main process, which owns the GPU net) through shared
memory: the worker writes its leaf planes into its slot, raises a request flag
the server polls, and blocks on a per-slot Event until the server writes back
the value/policy. Keeping this module import-light (numpy only) means 'spawn'ed
workers don't each load torch.
"""

from __future__ import annotations

import numpy as np

from .mcts_core import self_play_game

PLANES = 2 * 6 * 7
COLS = 7


def worker(slot, n_games, base_seed, counter, finished, req_planes, req_ready,
           val_arr, prob_arr, resp_event, result_q, n_slots, params) -> None:
    req_np = np.frombuffer(req_planes, dtype=np.float32).reshape(n_slots, 2, 6, 7)
    ready_np = np.frombuffer(req_ready, dtype=np.int8)
    val_np = np.frombuffer(val_arr, dtype=np.float32)
    prob_np = np.frombuffer(prob_arr, dtype=np.float32).reshape(n_slots, COLS)

    def evaluate(board):
        req_np[slot] = board.planes()           # write planes first ...
        resp_event.clear()
        ready_np[slot] = 1                       # ... then raise the request flag
        resp_event.wait()                        # server sets this after the forward
        return float(val_np[slot]), prob_np[slot].copy()

    results = []
    while True:
        with counter.get_lock():
            idx = counter.value
            if idx >= n_games:
                break
            counter.value = idx + 1
        rng = np.random.default_rng(base_seed + idx)
        ex, avg = self_play_game(
            evaluate, params["sims"], params["c_puct"], params["dir_alpha"],
            params["dir_eps"], params["temp_moves"], rng, params["move_ms"],
        )
        results.append((ex, avg))
        with finished.get_lock():
            finished.value += 1

    result_q.put(results)
