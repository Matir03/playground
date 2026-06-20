"""9×9 Go network — the reification MuZero-variant heads over a tanh latent (LGS).

Mirrors the Connect 4 net (`net.py` `_C4Base`/`ConnectFourResNet`) but for Go 9×9:
  * input is 3 planes (own stones, opponent stones, side-to-move constant), 9×9,
    matching `crates/games/go9/src/game.rs` `Go9::to_features`;
  * the action space is 82 — board points 0..80 (a = r*9 + c) plus pass (81);
  * heads: encoder -> LGS (LGS_DIM), value, policy prior over the 82 actions, and
    the reifier `pol_next` predicting the next-LGS per action (82 * LGS_DIM).

Self-play training uses `forward`/`policy_value` (value + prior); the reifier
`pol_next` is trained by a consistency loss (see rust_go9_selfplay_train.py) and
used by the engine's reification MCTS at play time. Checkpoints are the exact
`.safetensors` the Rust engine loads (parameter names must stay in sync with
`crates/games/go9/src/net.rs`).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LGS_DIM = 128
N = 9
CELLS = N * N            # 81 board points
ACTIONS = CELLS + 1      # 82: points + pass
PASS = CELLS             # action index of pass


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


class Go9ResNet(nn.Module):
    """AlphaZero-style ResNet trunk + reification heads over a LGS_DIM latent."""

    def __init__(self, channels: int = 96, blocks: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        self.enc_fc = nn.Linear(channels * CELLS, LGS_DIM)
        self.val_fc1 = nn.Linear(LGS_DIM, 64)
        self.val_fc2 = nn.Linear(64, 1)
        self.pol_prior = nn.Linear(LGS_DIM, ACTIONS)
        self.pol_next = nn.Linear(LGS_DIM, ACTIONS * LGS_DIM)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return x.flatten(1)

    def encode_batch(self, planes: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc_fc(self._trunk(planes)))

    def value_batch(self, lgs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.val_fc1(lgs))
        return torch.tanh(self.val_fc2(x)).squeeze(-1)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,3,9,9) -> (value (B,), policy logits (B,82)). Used by training."""
        lgs = self.encode_batch(planes)
        return self.value_batch(lgs), self.pol_prior(lgs)

    def policy_value(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,3,9,9) -> (value (B,), policy probs (B,82)). Used by self-play MCTS."""
        lgs = self.encode_batch(planes)
        return self.value_batch(lgs), F.softmax(self.pol_prior(lgs), dim=-1)


class Go9AZNet(nn.Module):
    """Traditional AlphaZero net (benchmark baseline): a ResNet trunk with standard
    conv policy + value heads that read the **spatial** trunk directly — no latent
    (LGS) bottleneck and no reifier. Policy is a direct softmax over the 82 actions.
    Contrast with `Go9ResNet`, which funnels the trunk through a LGS_DIM latent (for
    the reification MCTS) before its heads. Same input planes + checkpoint format;
    parameter names must stay in sync with `crates/games/go9/src/net.rs` AzNet.
    """

    def __init__(self, channels: int = 96, blocks: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        # Policy head: 1×1 conv -> 2 planes -> FC to 82 logits (AlphaZero-standard).
        self.pol_conv = nn.Conv2d(channels, 2, 1)
        self.pol_bn = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2 * CELLS, ACTIONS)
        # Value head: 1×1 conv -> 1 plane -> FC -> FC -> tanh.
        self.val_conv = nn.Conv2d(channels, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(CELLS, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, channels, 9, 9) — spatial structure preserved for the heads

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,3,9,9) -> (value (B,), policy logits (B,82))."""
        x = self._trunk(planes)
        p = F.relu(self.pol_bn(self.pol_conv(x))).flatten(1)
        logits = self.pol_fc(p)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)
        return v, logits

    def policy_value(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, logits = self.forward(planes)
        return v, F.softmax(logits, dim=-1)


def arch_of(model: nn.Module) -> dict:
    if isinstance(model, Go9AZNet):
        return {"kind": "go9-az", "channels": model.channels, "blocks": model.blocks_n}
    return {"kind": "go9-resnet", "channels": model.channels, "blocks": model.blocks_n}


def build_model(arch: dict | None) -> nn.Module:
    if arch and arch.get("kind") == "go9-az":
        return Go9AZNet(channels=int(arch.get("channels", 96)), blocks=int(arch.get("blocks", 10)))
    if arch and arch.get("kind") == "go9-resnet":
        return Go9ResNet(channels=int(arch.get("channels", 96)), blocks=int(arch.get("blocks", 10)))
    return Go9ResNet()


def save_checkpoint(model: Go9ResNet, path: str) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), path)


def load_checkpoint(model: Go9ResNet, path: str) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(path))


def save_random(out_dir: str, channels: int, blocks: int, seed: int = 0,
                kind: str = "go9-resnet") -> str:
    """Write a random-init checkpoint + sibling config.json (arch descriptor).
    `kind` selects go9-resnet (reification) or go9-az (vanilla AlphaZero)."""
    torch.manual_seed(seed)
    model = Go9AZNet(channels=channels, blocks=blocks) if kind == "go9-az" \
        else Go9ResNet(channels=channels, blocks=blocks)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")
    save_checkpoint(model, ckpt)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"arch": arch_of(model), "generation": 0, "winrate": None}, f, indent=2)
    return ckpt


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Go9 net utilities")
    ap.add_argument("--out", default="data/models/go9/random")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = save_random(args.out, args.channels, args.blocks, args.seed)
    print(f"saved random Go9 net to {path}")
