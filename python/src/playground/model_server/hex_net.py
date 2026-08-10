"""13×13 Hex AlphaZero net — a ResNet trunk, one spatial policy head (169
logits, one per cell), a scalar value head, and KataGo's ownership head. No latent state, no reifier,
no encoding space.

Input is 2 planes in the **canonical frame**: plane 0 the mover's stones, plane 1
the opponent's, with the board transposed when White is to move so the net always
sees itself as the player connecting top↔bottom. Hex's transpose+colour-swap
symmetry means the two colours are literally the same problem, so folding them
together halves what the net must learn — and removes the need for a
side-to-move plane. The policy head indexes that same frame; the Rust side maps
back with `canon_index` (see crates/games/hex/src/game.rs).

Checkpoints are the exact `.safetensors` the Rust engine loads; parameter names
must stay in sync with `crates/games/hex/src/net.rs`.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 13
CELLS = N * N  # 169
IN_PLANES = 2
POLICY_DIM = CELLS


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


class HexAZNet(nn.Module):
    """ResNet trunk + value head + one per-cell policy head."""

    def __init__(self, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        # Policy: one logit per cell, index r*13+c (raw — the search does the
        # softmax over legal cells).
        self.pol_conv = nn.Conv2d(channels, 1, 3, padding=1)
        # Ownership (KataGo's auxiliary target): per cell, who holds it when the
        # game ends, in [-1, 1] from the mover's point of view. 169 supervised
        # numbers per position instead of one ±1 outcome — the dense signal a
        # value head needs while every opening still looks like a coin flip.
        # Unused at inference; the Rust engine simply doesn't load it.
        self.own_conv = nn.Conv2d(channels, 1, 3, padding=1)
        # Value.
        self.val_conv = nn.Conv2d(channels, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(CELLS, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, C, N, N)

    def forward(
        self, planes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B,2,13,13) -> (value (B,), policy_logits (B,169), ownership (B,169))."""
        x = self._trunk(planes)
        b = planes.shape[0]
        p = self.pol_conv(x).reshape(b, POLICY_DIM)
        o = torch.tanh(self.own_conv(x)).reshape(b, POLICY_DIM)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)
        return v, p, o


def arch_of(model: HexAZNet) -> dict:
    return {"kind": "hex-az", "channels": model.channels, "blocks": model.blocks_n}


def build_model(arch: dict | None) -> HexAZNet:
    if arch:
        return HexAZNet(int(arch.get("channels", 64)), int(arch.get("blocks", 6)))
    return HexAZNet()


def save_checkpoint(model: HexAZNet, path: str) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), path)


def load_checkpoint(model: HexAZNet, path: str) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(path))


def save_random(out_dir: str, channels: int, blocks: int, seed: int = 0) -> str:
    """Write a random gen-0 checkpoint + config.json so the actor can start."""
    torch.manual_seed(seed)
    model = HexAZNet(channels=channels, blocks=blocks)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")
    save_checkpoint(model, ckpt)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"arch": arch_of(model), "generation": 0}, f, indent=2)
    return ckpt
