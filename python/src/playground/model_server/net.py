"""Random-weight neural network for the Connect 4 ModelService.

This is the reification-based MuZero variant's four heads (see
docs/algorithm.md) wired up with *random* initial weights — enough to serve
Encode/Eval/Policy/Reify end-to-end. It is not trained, so its play is weak;
the point is to exercise the full engine ↔ model-server path.

GS (game-state latent) is a `GS_DIM`-vector of float32, tanh-bounded. The
encoder reads a game's concrete state; for Connect 4 that is two 6×7 planes
(side-to-move's stones, opponent's stones).
"""

from __future__ import annotations

import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

GS_DIM = 64
ACTION_DIM = 16

# Connect 4 board (must match crates/games/connect4/src/game.rs encoding).
C4_ROWS = 6
C4_COLS = 7
C4_CELLS = C4_ROWS * C4_COLS


def connect4_planes(payload: bytes) -> torch.Tensor:
    """Decode a Connect 4 GameState.payload into 2×6×7 side-to-move planes.

    payload = 42 cell bytes (row-major, row 0 = bottom; 0 empty, 1 first,
    2 second) followed by 2 little-endian ply bytes.
    """
    if len(payload) < C4_CELLS + 2:
        raise ValueError(f"connect4 payload too short: {len(payload)}")
    ply = struct.unpack_from("<H", payload, C4_CELLS)[0]
    mover = 1 if ply % 2 == 0 else 2
    own = torch.zeros(C4_ROWS, C4_COLS)
    opp = torch.zeros(C4_ROWS, C4_COLS)
    for r in range(C4_ROWS):
        for c in range(C4_COLS):
            v = payload[r * C4_COLS + c]
            if v == mover:
                own[r, c] = 1.0
            elif v != 0:
                opp[r, c] = 1.0
    return torch.stack([own, opp], dim=0)  # (2, 6, 7)


def connect4_legal_columns(payload: bytes) -> list[int]:
    """Columns that are not full — playable from this Connect 4 state."""
    top = C4_ROWS - 1
    return [c for c in range(C4_COLS) if payload[top * C4_COLS + c] == 0]


class ConnectFourModel(nn.Module):
    """Encoder + eval + policy + reifier over the GS latent."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder: 2×6×7 -> GS_DIM
        self.enc_conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.enc_fc = nn.Linear(32 * C4_CELLS, GS_DIM)

        # Eval: GS -> scalar in [-1, 1]
        self.val_fc1 = nn.Linear(GS_DIM, 64)
        self.val_fc2 = nn.Linear(64, 1)

        # Policy: GS -> B priors, action vectors, and predicted next-GS.
        self.pol_prior = nn.Linear(GS_DIM, C4_COLS)
        self.pol_action = nn.Linear(GS_DIM, C4_COLS * ACTION_DIM)
        self.pol_next = nn.Linear(GS_DIM, C4_COLS * GS_DIM)

        # Reifier: (from_gs, to_gs) -> score per column (masked to legal).
        self.reify_fc = nn.Linear(2 * GS_DIM, C4_COLS)

    # — heads —
    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_conv1(planes.unsqueeze(0)))
        x = F.relu(self.enc_conv2(x))
        x = x.flatten(1)
        return torch.tanh(self.enc_fc(x)).squeeze(0)  # (GS_DIM,)

    def value(self, gs: torch.Tensor) -> float:
        x = F.relu(self.val_fc1(gs))
        return float(torch.tanh(self.val_fc2(x)).item())

    def policy(self, gs: torch.Tensor, top_b: int):
        priors = F.softmax(self.pol_prior(gs), dim=-1)
        actions = self.pol_action(gs).view(C4_COLS, ACTION_DIM)
        nexts = torch.tanh(self.pol_next(gs)).view(C4_COLS, GS_DIM)
        order = torch.argsort(priors, descending=True)
        out = []
        for i in order[: max(0, top_b)]:
            out.append((float(priors[i]), actions[i], nexts[i]))
        return out

    def reify_column(self, from_gs: torch.Tensor, to_gs: torch.Tensor, legal: list[int]) -> int:
        scores = self.reify_fc(torch.cat([from_gs, to_gs]))
        if not legal:
            return 0
        # Pick the highest-scoring legal column.
        best = max(legal, key=lambda c: float(scores[c]))
        return best


def gs_to_bytes(gs: torch.Tensor) -> bytes:
    return gs.detach().to(torch.float32).contiguous().numpy().tobytes()


def gs_from_bytes(payload: bytes) -> torch.Tensor:
    import numpy as np

    arr = np.frombuffer(payload, dtype="<f4").copy()
    return torch.from_numpy(arr)
