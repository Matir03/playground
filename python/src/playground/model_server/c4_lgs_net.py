"""Connect 4 network for the reification-based latent-space algorithm.

See docs/algorithm.md. This is a *separate* net from net.py (the discrete-AZ
baseline); the two coexist so the latent search can be regressed against
direct-action AZ. Heads over a tanh LGS latent (dim ``LGS_DIM``):

  encoder : 2x6x7 planes -> LGS            (ResNet trunk + global pooling)
  eval    : LGS -> value in [-1, 1] (side to move)
  policy  : LGS -> MDN over next-LGS       (K diagonal Gaussians)
  reifier : [LGS || next-LGS] -> 7 column logits
  aux     : opponent-reply 7-way + short-term value (training regularizers)

Layer names are chosen so the candle port (crates/games/connect4/src/lgs_net.rs)
can load the exact safetensors keys. BatchNorm runs in eval mode at inference,
matching candle's `forward_t(.., train=false)`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import C4_CELLS, C4_COLS, _ResBlock, connect4_legal_columns, connect4_planes

LGS_DIM = 64
K_COMPONENTS = 8
SIGMA_MIN = 0.05  # std floor (added to softplus); also the KR-PUCT scale floor


class GlobalPoolBias(nn.Module):
    """KataGo-style global pooling: pool channel features across the board
    (mean + max) and broadcast them back as per-channel biases, so the trunk can
    route board-global information (who's ahead, threats) into every cell."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)  # (B,2C)
        return x + self.fc(pooled)[:, :, None, None]


class C4LGSNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 5, k: int = K_COMPONENTS) -> None:
        super().__init__()
        self.channels, self.blocks_n, self.k = channels, blocks, k
        self.stem = nn.Conv2d(2, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        self.gpool = GlobalPoolBias(channels)
        feat = channels * C4_CELLS
        self.enc_fc = nn.Linear(feat, LGS_DIM)
        # eval
        self.val_fc1 = nn.Linear(LGS_DIM, 64)
        self.val_fc2 = nn.Linear(64, 1)
        # MDN policy over next-LGS
        self.mix_logits = nn.Linear(LGS_DIM, k)
        self.means = nn.Linear(LGS_DIM, k * LGS_DIM)
        self.log_stds = nn.Linear(LGS_DIM, k * LGS_DIM)
        # reifier: (LGS, next-LGS) -> column logits
        self.reif1 = nn.Linear(2 * LGS_DIM, 128)
        self.reif2 = nn.Linear(128, C4_COLS)
        # auxiliary heads
        self.aux_oppreply = nn.Linear(LGS_DIM, C4_COLS)
        self.aux_stv = nn.Linear(LGS_DIM, 1)

    # — trunk / encoder —
    def trunk(self, planes: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        x = self.gpool(x)
        return x.flatten(1)

    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc_fc(self.trunk(planes)))  # (B, d)

    # — heads —
    def value(self, lgs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.val_fc2(F.relu(self.val_fc1(lgs)))).squeeze(-1)  # (B,)

    def policy_mdn(self, lgs: torch.Tensor):
        """LGS (B,d) -> mixture (weights (B,k), means (B,k,d), stds (B,k,d))."""
        w = F.softmax(self.mix_logits(lgs), dim=-1)
        mu = torch.tanh(self.means(lgs)).view(-1, self.k, LGS_DIM)
        std = F.softplus(self.log_stds(lgs)).view(-1, self.k, LGS_DIM) + SIGMA_MIN
        return w, mu, std

    def reify_logits(self, lgs: torch.Tensor, next_lgs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.reif1(torch.cat([lgs, next_lgs], dim=-1)))
        return self.reif2(h)  # (B, 7)

    def aux(self, lgs: torch.Tensor):
        return self.aux_oppreply(lgs), torch.tanh(self.aux_stv(lgs)).squeeze(-1)


def mdn_log_prob(x: torch.Tensor, w: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """log p(x) under a diagonal Gaussian mixture, batched.

    x:(B,d)  w:(B,k)  mu:(B,k,d)  std:(B,k,d)  ->  (B,)
    """
    d = x.shape[-1]
    xe = x.unsqueeze(1)  # (B,1,d)
    z = (xe - mu) / std  # (B,k,d)
    log_comp = (
        -0.5 * (z * z).sum(-1)
        - std.log().sum(-1)
        - 0.5 * d * math.log(2.0 * math.pi)
    )  # (B,k)
    return torch.logsumexp(w.clamp_min(1e-9).log() + log_comp, dim=-1)  # (B,)


def mmd2_mixtures(
    wa, mua, sda, wb, mub, sdb, h: float
) -> torch.Tensor:
    """Squared MMD between two diagonal-Gaussian mixtures under the RBF kernel
    k(x,y)=exp(-||x-y||^2 / 2h^2). Closed form: each cross term is a
    Gaussian-Gaussian convolution integral. Used by the consistency loss.

    Batched over the leading dim; a/b are (B,k,*) with weights (B,k).
    """
    def cross(w1, mu1, sd1, w2, mu2, sd2):
        # sum_{i,j} w1_i w2_j prod_dim N-overlap(mu1_i, mu2_j ; sd1_i^2+sd2_j^2+h^2)
        mu1 = mu1.unsqueeze(2)   # (B,k1,1,d)
        mu2 = mu2.unsqueeze(1)   # (B,1,k2,d)
        var = (sd1 * sd1).unsqueeze(2) + (sd2 * sd2).unsqueeze(1) + (h * h)  # (B,k1,k2,d)
        diff2 = (mu1 - mu2) ** 2
        # RBF kernel integral against two Gaussians (per dim), product over dims:
        log_k = (-0.5 * (diff2 / var) - 0.5 * torch.log(var / (h * h))).sum(-1)  # (B,k1,k2)
        wij = w1.unsqueeze(2) * w2.unsqueeze(1)  # (B,k1,k2)
        return (wij * log_k.exp()).sum(dim=(1, 2))  # (B,)

    kaa = cross(wa, mua, sda, wa, mua, sda)
    kbb = cross(wb, mub, sdb, wb, mub, sdb)
    kab = cross(wa, mua, sda, wb, mub, sdb)
    return (kaa + kbb - 2.0 * kab).clamp_min(0.0)


# — checkpoint / arch helpers (mirror net.py) —
def build_lgs_model(arch: dict | None) -> C4LGSNet:
    arch = arch or {}
    return C4LGSNet(
        channels=int(arch.get("channels", 64)),
        blocks=int(arch.get("blocks", 5)),
        k=int(arch.get("k", K_COMPONENTS)),
    )


def arch_of(model: C4LGSNet) -> dict:
    return {"kind": "lgs", "channels": model.channels, "blocks": model.blocks_n, "k": model.k}


def save_checkpoint(model: C4LGSNet, path: str) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), path)


def load_checkpoint(model: C4LGSNet, path: str) -> None:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(path))


__all__ = [
    "LGS_DIM",
    "K_COMPONENTS",
    "SIGMA_MIN",
    "C4LGSNet",
    "mdn_log_prob",
    "mmd2_mixtures",
    "build_lgs_model",
    "arch_of",
    "save_checkpoint",
    "load_checkpoint",
    "connect4_planes",
    "connect4_legal_columns",
]
