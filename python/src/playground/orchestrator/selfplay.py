"""Self-play loop: spawns/supervises engine workers in batched self-play mode.

Stub — see docs/training.md for the intended shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfPlayConfig:
    game: str
    run_id: str
    model_version: str
    parallel_games: int = 64
    games_per_engine_worker: int = 1024
    engine_workers: int = 4
    per_move_visits: int = 200
    records_dir: str = "data/runs"


def run_loop(_config: SelfPlayConfig) -> None:
    """Run the self-play loop until interrupted. Stub."""
    raise NotImplementedError("run_loop: implement at M2")
