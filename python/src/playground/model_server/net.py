"""Connect 4 network for the ModelService.

The reification-based MuZero variant's heads (see docs/algorithm.md) over a
`GS_DIM` tanh latent (the GS / LatentGameState):

  encoder : 2×6×7 planes (side-to-move's stones, opponent's) -> GS
  eval    : GS -> scalar value in [-1, 1] (outcome for the side to move)
  policy  : GS -> per-column prior + predicted next-GS
  reifier : decode a policy edge back to a concrete move

Connect 4's action space is just the 7 columns, so policy candidate *i*
corresponds to column *i* and reification is a structural decode of the column
the policy selected (no learned reifier needed for a discrete action space —
that machinery is for Amazons/Arimaa). The trainable, play-relevant heads are
therefore `eval` and `pol_prior`; `train.py` fits them by cloning the perfect
solver.
"""

from __future__ import annotations

import numpy as np
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
    cells = np.frombuffer(bytes(payload[:C4_CELLS]), dtype=np.uint8).reshape(C4_ROWS, C4_COLS)
    ply = int(payload[C4_CELLS]) | (int(payload[C4_CELLS + 1]) << 8)
    mover = 1 if ply % 2 == 0 else 2
    own = (cells == mover).astype(np.float32)
    opp = ((cells != 0) & (cells != mover)).astype(np.float32)
    return torch.from_numpy(np.stack([own, opp], axis=0))  # (2, 6, 7)


def connect4_legal_columns(payload: bytes) -> list[int]:
    """Columns that are not full — playable from this Connect 4 state."""
    top = C4_ROWS - 1
    return [c for c in range(C4_COLS) if payload[top * C4_COLS + c] == 0]


def action_vector_for_column(col: int) -> torch.Tensor:
    """Action-head seed that encodes the chosen column (reifier decodes it)."""
    v = torch.zeros(ACTION_DIM)
    v[0] = float(col)
    return v


def decode_column(action_seed: bytes) -> int | None:
    if not action_seed:
        return None
    arr = np.frombuffer(bytes(action_seed), dtype="<f4")
    return int(round(float(arr[0]))) if arr.size else None


class _C4Base(nn.Module):
    """Shared heads + inference over a `_trunk(planes) -> (B, trunk_features)`.

    Subclasses pick the trunk (a small 2-conv net or a ResNet); the encoder fc,
    eval head, and policy heads (prior over columns + predicted next-GS) are
    common, so checkpoints differ only in trunk weights.
    """

    def __init__(self, trunk_features: int) -> None:
        super().__init__()
        self.enc_fc = nn.Linear(trunk_features, GS_DIM)
        self.val_fc1 = nn.Linear(GS_DIM, 64)
        self.val_fc2 = nn.Linear(64, 1)
        self.pol_prior = nn.Linear(GS_DIM, C4_COLS)
        self.pol_next = nn.Linear(GS_DIM, C4_COLS * GS_DIM)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # — batched —
    def encode_batch(self, planes: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc_fc(self._trunk(planes)))

    def value_batch(self, gs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.val_fc1(gs))
        return torch.tanh(self.val_fc2(x)).squeeze(-1)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,2,6,7) -> (value (B,), policy logits (B,7)). Used by training."""
        gs = self.encode_batch(planes)
        return self.value_batch(gs), self.pol_prior(gs)

    def policy_value(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,2,6,7) -> (value (B,), policy probs (B,7)). Used by self-play MCTS."""
        gs = self.encode_batch(planes)
        return self.value_batch(gs), F.softmax(self.pol_prior(gs), dim=-1)

    # — single-example (model server inference) —
    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        return self.encode_batch(planes.unsqueeze(0)).squeeze(0)

    def value(self, gs: torch.Tensor) -> float:
        return float(self.value_batch(gs.unsqueeze(0)).item())

    def policy(self, gs: torch.Tensor, top_b: int):
        priors = F.softmax(self.pol_prior(gs), dim=-1)
        nexts = torch.tanh(self.pol_next(gs)).view(C4_COLS, GS_DIM)
        order = torch.argsort(priors, descending=True)
        out = []
        for i in order[: max(0, top_b)]:
            col = int(i)
            out.append((float(priors[col]), action_vector_for_column(col), nexts[col]))
        return out

    def reify_column(self, action_seed: bytes, legal: list[int]) -> int:
        col = decode_column(action_seed)
        if col is not None and col in legal:
            return col
        return legal[0] if legal else 0


class ConnectFourModel(_C4Base):
    """Small net (2 conv layers). Used by the random + bc-perfect checkpoints."""

    def __init__(self) -> None:
        super().__init__(32 * C4_CELLS)
        self.enc_conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 32, 3, padding=1)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_conv1(planes))
        x = F.relu(self.enc_conv2(x))
        return x.flatten(1)


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ConnectFourResNet(_C4Base):
    """Larger residual net for self-play training (AlphaZero-style trunk)."""

    def __init__(self, channels: int = 64, blocks: int = 5) -> None:
        super().__init__(channels * C4_CELLS)
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(2, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return x.flatten(1)


def build_model(arch: dict | None) -> _C4Base:
    """Construct a net from an architecture descriptor (see arch_of)."""
    if arch and arch.get("kind") == "resnet":
        return ConnectFourResNet(channels=int(arch.get("channels", 64)), blocks=int(arch.get("blocks", 5)))
    return ConnectFourModel()


def arch_of(model: _C4Base) -> dict:
    if isinstance(model, ConnectFourResNet):
        return {"kind": "resnet", "channels": model.channels, "blocks": model.blocks_n}
    return {"kind": "small"}


def save_checkpoint(model: _C4Base, path: str) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), path)


def load_checkpoint(model: _C4Base, path: str) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(path))


def gs_to_bytes(gs: torch.Tensor) -> bytes:
    return gs.detach().to(torch.float32).contiguous().numpy().tobytes()


def gs_from_bytes(payload: bytes) -> torch.Tensor:
    arr = np.frombuffer(bytes(payload), dtype="<f4").copy()
    return torch.from_numpy(arr)
