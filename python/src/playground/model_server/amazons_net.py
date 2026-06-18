"""10×10 Amazons AlphaZero net — pure ResNet trunk + a value head and TWO spatial
policy heads (queen-move + arrow).

A move is compound (queen from→to, then shoot an arrow), so a flat policy over whole
turns is ~10⁶. Instead the policy factors into two heads, each `N*N*72` logits
(per square, 72 move-types = 8 directions × N-1 distances). The per-move prior used
by the search is `softmax_over_legal(queen_logit[from,qmt] + arrow_logit[to,amt])`.

Each head is its 72 conv channels reshaped `(B,72,N,N) -> permute (B,N,N,72) ->
(B,7200)`, so the flat index is `(r*N+c)*72 + movetype` — this MUST match the Rust
`moves::queen_index`/`arrow_index` and `net.rs` head reshape. Input is 4 canonical
planes (my queens, opp queens, arrows, bias=1). Checkpoints are the exact
`.safetensors` the Rust engine loads; parameter names must stay in sync with
`crates/games/amazons/src/net.rs`.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 10
CELLS = N * N          # 100
IN_PLANES = 4
MOVE_TYPES = 72        # 8 dirs × 9 dists
PER_HEAD = CELLS * MOVE_TYPES  # 7200


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


class AmazonsAZNet(nn.Module):
    """ResNet trunk + value head + queen/arrow policy heads (no latent, no reifier)."""

    def __init__(self, channels: int = 96, blocks: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        # Two spatial policy heads: 72 move-type channels per square (raw logits).
        self.q_conv = nn.Conv2d(channels, MOVE_TYPES, 3, padding=1)
        self.a_conv = nn.Conv2d(channels, MOVE_TYPES, 3, padding=1)
        # Value head.
        self.val_conv = nn.Conv2d(channels, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(CELLS, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, C, N, N)

    def _head(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        # (B,72,N,N) -> (B,N,N,72) -> (B,7200): idx = (r*N+c)*72 + movetype.
        b = x.shape[0]
        return conv(x).permute(0, 2, 3, 1).reshape(b, PER_HEAD)

    def forward(
        self, planes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B,4,N,N) -> (value (B,), queen_logits (B,7200), arrow_logits (B,7200))."""
        x = self._trunk(planes)
        q = self._head(self.q_conv, x)
        a = self._head(self.a_conv, x)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)
        return v, q, a


def arch_of(model: AmazonsAZNet) -> dict:
    return {"kind": "amazons-az", "channels": model.channels, "blocks": model.blocks_n}


def build_model(arch: dict | None) -> AmazonsAZNet:
    if arch:
        return AmazonsAZNet(int(arch.get("channels", 96)), int(arch.get("blocks", 10)))
    return AmazonsAZNet()


def save_checkpoint(model: AmazonsAZNet, path: str) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), path)


def load_checkpoint(model: AmazonsAZNet, path: str) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(path))


def save_random(out_dir: str, channels: int, blocks: int, seed: int = 0) -> str:
    torch.manual_seed(seed)
    model = AmazonsAZNet(channels=channels, blocks=blocks)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")
    save_checkpoint(model, ckpt)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"arch": arch_of(model), "generation": 0}, f, indent=2)
    return ckpt
