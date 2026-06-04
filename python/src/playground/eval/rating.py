"""Bradley-Terry / Elo fit from round-robin results.

Given total scores and game counts between participants, fit Elo ratings by
gradient ascent on the Bradley-Terry log-likelihood, anchored so the named
reference (default "random") sits at 0.
"""

from __future__ import annotations

import numpy as np

_SCALE = 400.0  # Elo scale
_LN10 = np.log(10.0)


def fit_elo(names, score, games, anchor: str = "random", iters: int = 5000, lr: float = 4.0):
    """`score[i][j]` = points i scored vs j; `games[i][j]` = games played.

    Returns {name: elo}, anchored so `anchor` (if present) is 0.
    """
    n = len(names)
    r = np.zeros(n)
    score = np.asarray(score, dtype=float)
    games = np.asarray(games, dtype=float)
    anchor_idx = names.index(anchor) if anchor in names else None

    for _ in range(iters):
        grad = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j or games[i][j] == 0:
                    continue
                p = 1.0 / (1.0 + 10.0 ** ((r[j] - r[i]) / _SCALE))
                grad[i] += (score[i][j] - games[i][j] * p) * (_LN10 / _SCALE)
        r += lr * grad
        if anchor_idx is not None:
            r -= r[anchor_idx]

    return {names[i]: float(r[i]) for i in range(n)}
