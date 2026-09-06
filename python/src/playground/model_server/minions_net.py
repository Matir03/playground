"""Simplified-Minions network — the source of truth for the architecture.

The candle mirror lives in ``crates/games/minions/src/net.rs`` and loads the
*same* ``.safetensors`` file, so parameter names here are a wire format: renaming
one silently breaks the Rust loader. A parity test pins them.

Four heads on one trunk, matching the plan:

``value``
    Scalar for the side to move. A per-cell ``ownership`` auxiliary used to sit
    here — graveyard control at game end, the Hex ownership trick — but it was
    dropped as redundant: ``hex_control`` already covers the same ground as a
    time-averaged occupancy map over every cell.

``diffusion``
    The sampler. Predicts, for every cell, the three declaration fields of a
    :class:`TurnSpec` — 37 move offsets, 3 target classes, 6 spawn classes — given
    the state *and a partially committed declaration*. Trained BERT-style: mask a
    random fraction of the live variables and predict them from the rest. Sampling
    runs it ``T`` times, committing the most confident fraction each round.

    ``T`` is a dial rather than a constant. ``T=1`` degenerates to a mean-field
    factored policy; ``T=n`` commits one variable at a time in a random order,
    which is exact autoregressive sampling whose product of conditionals is a
    proper normalized density — that mode is how the actor measures effective
    sample size honestly, using the very same weights it runs fast in self-play.

``energy``
    Scores a *whole* declaration, ``f(s, a)``. The policy of record is
    ``pi(a|s) ∝ exp f(s, a)``. It sees the joint declaration, so it can score
    coordination — three units converging *and* the skeleton dying — that no
    per-cell head can express. Priors are only ever needed relative to a
    candidate set, so ``softmax`` over the sampled candidates is all that is
    required; no normalizing constant, and no importance weights on the hot path.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Board / feature geometry. Must match crates/games/minions/src/{map,game,turn}.rs.
N = 7
CELLS = N * N
IN_PLANES = 26  # game::PLANES
N_OFFSETS = 37  # turn::N_OFFSETS — hexes within distance 3
N_TARGETS = 3  # stay / kill / unsummon
N_SPAWNS = 6  # nothing + 5 buyable units
DECL_LOGITS = N_OFFSETS + N_TARGETS + N_SPAWNS  # 46 per cell

# Planes describing a partially committed declaration, fed back into the
# diffusion head each round. Move destinations go in as (dx, dy) rather than a
# 37-way one-hot: same information, 34 fewer planes.
#   0    move dx / 3
#   1    move dy / 3
#   2    move committed
#   3-5  target one-hot (stay, kill, unsummon)
#   6-11 spawn one-hot (none + 5 units)
#
# Only moves carry a commit flag. Target and spawn are one-hot over class sets
# that already include "nothing", so all-zeros means uncommitted; their old
# commit planes were exactly the sum of their one-hots. A move committed to
# offset 0 ("stay") writes (0, 0), indistinguishable from uncommitted without
# the flag — so that one stays.
DECL_PLANES = 12

# The energy head only ever scores *complete* declarations, so the move-commit
# flag degenerates to a liveness mask there — information `h` already carries.
# It gets its own 11-plane encoding without it:
#   0,1  move dx/3, dy/3 · 2-4 target one-hot · 5-10 spawn one-hot
ENERGY_PLANES = DECL_PLANES - 1


def head_width(channels: int) -> int:
    """Width of the shared head neck. Mirrored in crates/games/minions/src/net.rs."""
    return max(32, channels // 2)

# Auxiliary "rest of the game" target geometry — must match
# crates/games/minions/src/selfplay.rs.
NECRO_HORIZON = 10  # side-turns of look-ahead for the survival hazard curve
N_BUYABLE = 5  # zombie, skeleton, serpent, warg, ghost


class _ResBlock(nn.Module):
    """Same shape and parameter names as every other net in the repo, so the
    candle loaders stay interchangeable."""

    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class GlobalPoolBias(nn.Module):
    """KataGo-style global pooling: mean+max over the board, broadcast back.

    Minions leans on this harder than a placement game does — gold, graveyard
    count and material balance are global quantities that stacked 3x3 convs
    aggregate poorly, and the value head and the adjudication rule both turn on
    exactly those.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(2 * channels, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        flat = x.view(b, c, -1)
        pooled = torch.cat([flat.mean(dim=2), flat.max(dim=2).values], dim=1)
        return x + self.fc(pooled).view(b, c, 1, 1)


class MinionsNet(nn.Module):
    def __init__(self, channels: int = 96, blocks: int = 8, energy_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.energy_blocks_n = energy_blocks

        # ── shared trunk ──────────────────────────────────────────────────────
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))
        self.gpool = GlobalPoolBias(channels)

        # ── per-head necks ────────────────────────────────────────────────────
        #
        # Every head used to hang straight off the last trunk layer, and value
        # collapsed 96 channels to 1 in a single 1x1. That layer was serving
        # value, hex control and necromancer survival at once, *and* feeding the diffusion and energy heads — so
        # each head's gradient fought the others for the same features. A small
        # private neck per head gives each one somewhere to specialise before
        # the bottleneck, at ~1% of trunk cost (96->32 at 7x7).
        neck = max(16, channels // 3)
        self.neck_w = neck

        def conv_neck() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, neck, 3, padding=1),
                nn.BatchNorm2d(neck),
                nn.ReLU(inplace=True),
                nn.Conv2d(neck, neck, 3, padding=1),
                nn.BatchNorm2d(neck),
                nn.ReLU(inplace=True),
            )

        # ── value ─────────────────────────────────────────────────────────────
        self.val_neck = conv_neck()
        self.val_conv = nn.Conv2d(neck, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(CELLS, 64)
        self.val_fc2 = nn.Linear(64, 1)


        # ── auxiliary "rest of the game" heads ────────────────────────────────
        #
        # All in the canonical frame: channel/slot 0 is the side to move. They
        # share the trunk, so their gradient shapes the representation — which is
        # the point of an auxiliary target, not the predictions themselves.
        #
        # Hex control is a soft occupancy map (2 channels of logits over the
        # board). Necromancer survival is a per-side hazard curve over
        # NECRO_HORIZON turns — monotone by construction in the target, so the
        # head is forced to learn a coherent curve rather than 10 loose bits.
        # Unit selection is a small non-negative purchase rate, so it is
        # regressed through softplus rather than squashed.
        self.hex_neck = conv_neck()
        self.hex_conv = nn.Conv2d(neck, 2, 3, padding=1)
        # The pooled heads get a hidden layer of their own for the same reason.
        self.necro_fc0 = nn.Linear(2 * channels, 128)
        self.necro_fc1 = nn.Linear(128, 64)
        self.necro_fc2 = nn.Linear(64, 2 * NECRO_HORIZON)


        # ── shared head neck ──────────────────────────────────────────────────
        #
        # Depth between the trunk and the two declaration heads, computed *once*
        # per position and reused by all T diffusion rounds and the energy pass.
        # Putting layers here instead of inside the heads makes them free per
        # round rather than paid T+1 times.
        #
        # It also narrows 96 -> 48, which matters more than the arithmetic: these
        # heads are bound by data movement, and the neck output is what gets
        # repeated across the K candidate chains. An earlier attempt to
        # precompute only the first conv's `h` half was measured *slower* because
        # it doubled that transfer; halving the width moves it the other way.
        head_ch = head_width(channels)
        self.head_ch = head_ch
        self.head_neck = nn.Sequential(
            nn.Conv2d(channels, head_ch, 3, padding=1),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.head_block = _ResBlock(head_ch)

        # ── log-partition head, for trajectory balance ────────────────────────
        #
        # `Z(s) = sum_a exp f(s,a)` over every legal turn from s — intractable to
        # compute (~1e15 turns), so it is learned. State-conditional because the
        # reward `f(s,a)` is state-conditional: one global scalar would have to
        # average over positions with wildly different numbers of legal turns.
        #
        # Training-only: never exported to ONNX, and the sampler never needs it.
        self.logz_fc1 = nn.Linear(2 * channels, 64)
        self.logz_fc2 = nn.Linear(64, 1)

        # ── diffusion (sampler) head ──────────────────────────────────────────
        # Conditions on the trunk plus the partially committed declaration.
        #
        # Kept as one convolution over the concatenated input. Splitting it so
        # the `h` half could be precomputed is algebraically exact and removes
        # ~28% of the head's arithmetic — but measured *slower* end to end: these
        # heads are bound by data movement, not FLOPs, and the split doubled the
        # trunk's output transfer for a 4% gain on the head.
        self.dif_in = nn.Conv2d(head_ch + DECL_PLANES, channels, 3, padding=1)
        self.dif_bn = nn.BatchNorm2d(channels)
        self.dif_block = _ResBlock(channels)
        self.dif_gpool = GlobalPoolBias(channels)
        self.dif_out = nn.Conv2d(channels, DECL_LOGITS, 1)

        # ── energy head ───────────────────────────────────────────────────────
        self.eng_in = nn.Conv2d(head_ch + ENERGY_PLANES, channels, 3, padding=1)
        self.eng_bn = nn.BatchNorm2d(channels)
        self.eng_blocks = nn.ModuleList(_ResBlock(channels) for _ in range(energy_blocks))
        self.eng_gpool = GlobalPoolBias(channels)
        self.eng_fc1 = nn.Linear(2 * channels, 64)
        self.eng_fc2 = nn.Linear(64, 1)

    # ── forward pieces ────────────────────────────────────────────────────────

    def trunk(self, planes):
        """(B, IN_PLANES, N, N) -> (B, C, N, N)."""
        x = F.relu(self.stem_bn(self.stem(planes)))
        for blk in self.blocks:
            x = blk(x)
        return self.gpool(x)

    def value_head(self, h):
        """-> value (B,).

        No ownership output: it scored graveyard control at the final state,
        which `hex_control` already covers as a time-averaged occupancy map over
        every cell. Two heads pulling the trunk toward the same signal.
        """
        v = F.relu(self.val_bn(self.val_conv(self.val_neck(h)))).flatten(1)
        return torch.tanh(self.val_fc2(F.relu(self.val_fc1(v)))).squeeze(-1)

    def aux_head(self, h):
        """-> hex-control logits (B, 2, CELLS), necromancer-hazard logits
        (B, 2, NECRO_HORIZON), unit-purchase rates (B, 2, N_BUYABLE).

        Logits, not probabilities: the losses are BCE-with-logits, which is
        numerically better behaved and keeps the ONNX graph free of a sigmoid the
        Rust side would have to mirror exactly.
        """
        hexc = self.hex_conv(self.hex_neck(h)).flatten(2)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        necro = self.necro_fc2(F.relu(self.necro_fc1(F.relu(self.necro_fc0(pooled)))))
        return hexc, necro.view(-1, 2, NECRO_HORIZON)

    def log_z(self, h):
        """-> (B,) estimate of log Z(s), the log-partition of `exp f(s, .)`."""
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.logz_fc2(F.relu(self.logz_fc1(pooled))).squeeze(-1)

    def head_features(self, h):
        """Trunk -> the (B, head_ch, N, N) both declaration heads condition on.

        Run once per position; the sampler caches it across the T diffusion
        rounds and the energy pass.
        """
        return self.head_block(self.head_neck(h))

    def diffusion_head(self, hh, decl):
        """Logits for every cell's three fields.

        ``hh`` is `head_features(h)`. ``decl`` is (B, DECL_PLANES, N, N)
        describing what is committed so far.
        Returns (moves, targets, spawns) with shapes
        (B, CELLS, 37), (B, CELLS, 3), (B, CELLS, 6).
        """
        x = F.relu(self.dif_bn(self.dif_in(torch.cat([hh, decl], dim=1))))
        x = self.dif_gpool(self.dif_block(x))
        # (B, 46, N, N) -> (B, CELLS, 46) so each cell's logits are contiguous.
        out = self.dif_out(x).permute(0, 2, 3, 1).reshape(-1, CELLS, DECL_LOGITS)
        return torch.split(out, [N_OFFSETS, N_TARGETS, N_SPAWNS], dim=2)

    def energy_head(self, hh, decl):
        """Scalar score for a whole declaration -> (B,).

        ``decl`` here is ENERGY_PLANES wide, not DECL_PLANES: complete
        declarations need no commit flag.
        """
        x = F.relu(self.eng_bn(self.eng_in(torch.cat([hh, decl], dim=1))))
        for blk in self.eng_blocks:
            x = blk(x)
        x = self.eng_gpool(x)
        b, c, _, _ = x.shape
        flat = x.view(b, c, -1)
        pooled = torch.cat([flat.mean(dim=2), flat.max(dim=2).values], dim=1)
        return self.eng_fc2(F.relu(self.eng_fc1(pooled))).squeeze(-1)

    def forward(self, planes, decl=None, edecl=None):
        """Value, and — when a declaration is supplied — its logits.

        ``edecl`` is the ENERGY_PLANES-wide encoding of the same declaration;
        defaults to dropping the move-commit plane from ``decl``.
        """
        h = self.trunk(planes)
        v = self.value_head(h)
        if decl is None:
            return v
        hh = self.head_features(h)
        if edecl is None:
            edecl = torch.cat([decl[:, :2], decl[:, 3:]], dim=1)
        return v, *self.diffusion_head(hh, decl), self.energy_head(hh, edecl)


# ── losses ────────────────────────────────────────────────────────────────────


def masked_declaration_loss(logits, targets, mask):
    """Cross-entropy over masked positions only (the BERT/D3PM objective).

    ``logits`` (B, CELLS, K), ``targets`` (B, CELLS) int64, ``mask`` (B, CELLS)
    bool — True where the variable was hidden and must be predicted. Positions
    that are not live in this position (an empty cell has no move to declare) are
    simply never masked, so they never contribute.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    flat = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    tgt = targets.reshape(-1)[mask.reshape(-1)]
    return F.cross_entropy(flat, tgt)


def masked_declaration_loss_soft(logits, target_probs, mask):
    """Soft-target version of :func:`masked_declaration_loss`.

    ``target_probs`` (B, CELLS, K) is a distribution per position rather than a
    class index, which is what lets a geometric prior be written into the target
    (see ``smoothed_move_target`` in the learner). With a one-hot
    ``target_probs`` this is identical to the hard-label loss.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    flat = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    tgt = target_probs.reshape(-1, target_probs.shape[-1])[mask.reshape(-1)]
    return -(tgt * F.log_softmax(flat, dim=-1)).sum(dim=-1).mean()


def energy_loss(scores, visits):
    """Cross-entropy of ``softmax f`` over a candidate set against pruned visits.

    ``scores`` (B, K) energies for one node's candidates, ``visits`` (B, K) the
    *pruned* visit distribution — forced playouts and injected random candidates
    already subtracted out, so the target reflects what search chose on merit.
    """
    tgt = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return -(tgt * F.log_softmax(scores, dim=1)).sum(dim=1).mean()


# ── ONNX export ───────────────────────────────────────────────────────────────
#
# Three graphs, not one. The sampler computes the trunk *once* per position and
# then runs the diffusion head T times and the energy head once against it; a
# single fused graph would recompute the trunk on every one of those calls and
# throw away the only thing that makes T-step sampling affordable.


class _TrunkGraph(nn.Module):
    """planes -> (head features, value). Run once per position.

    Emits `head_features(h)`, not the raw trunk: the declaration heads are the
    only consumers on the Rust side, and the neck is the part worth caching
    across the T diffusion rounds.
    """

    def __init__(self, net: "MinionsNet"):
        super().__init__()
        self.net = net

    def forward(self, planes):
        h = self.net.trunk(planes)
        return self.net.head_features(h), self.net.value_head(h)


class _DiffusionGraph(nn.Module):
    """(h, decl) -> (B, CELLS, 46). Run once per diffusion round, batched over
    the K candidate chains."""

    def __init__(self, net: "MinionsNet"):
        super().__init__()
        self.net = net

    def forward(self, h, decl):
        mv, tg, sp = self.net.diffusion_head(h, decl)
        return torch.cat([mv, tg, sp], dim=2)


class _AuxGraph(nn.Module):
    """planes -> (hex-control logits, necromancer-hazard logits, log Z).

    Analysis only. The auxiliary heads exist to shape the trunk during
    training, and self-play never reads them, so they are exported as their own
    graph rather than added to `trunk.onnx`: an extra output on the hot path
    would cost every expansion in every self-play worker to serve a view that
    runs once when a human asks for it.

    That independence is the point — this graph can appear, change or go missing
    without the training path noticing. It recomputes the trunk internally,
    which is a wasted forward pass in exchange for not touching the graph the
    actor depends on.
    """

    def __init__(self, net: "MinionsNet"):
        super().__init__()
        self.net = net

    def forward(self, planes):
        h = self.net.trunk(planes)
        hexc, necro = self.net.aux_head(h)
        # `log_z` pools the *raw* trunk (2 * channels), not the head features.
        return hexc, necro, self.net.log_z(h)


class _EnergyGraph(nn.Module):
    """(h, decl) -> (B,). Run once per expansion, batched over candidates."""

    def __init__(self, net: "MinionsNet"):
        super().__init__()
        self.net = net

    def forward(self, h, decl):
        return self.net.energy_head(h, decl)


def export_onnx_atomic(net: MinionsNet, out_dir: str) -> None:
    """Write ``trunk.onnx``, ``diffusion.onnx``, ``energy.onnx`` and ``aux.onnx``.

    Every axis that varies at run time is dynamic: batch everywhere, and the
    trunk width, so a checkpoint at different ``channels`` still loads. Written
    to ``.tmp`` and renamed, because the actor may pick these up mid-run.
    """
    was_training = net.training
    net.eval()
    dev = next(net.parameters()).device
    # The declaration graphs consume the neck output, not the raw trunk.
    c = net.head_ch
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        (
            "trunk.onnx",
            _TrunkGraph(net),
            (torch.zeros(1, IN_PLANES, N, N, device=dev),),
            ["planes"],
            ["h", "value"],
        ),
        (
            "diffusion.onnx",
            _DiffusionGraph(net),
            (torch.zeros(1, c, N, N, device=dev), torch.zeros(1, DECL_PLANES, N, N, device=dev)),
            ["h", "decl"],
            ["logits"],
        ),
        (
            "energy.onnx",
            _EnergyGraph(net),
            (torch.zeros(1, c, N, N, device=dev), torch.zeros(1, ENERGY_PLANES, N, N, device=dev)),
            ["h", "decl"],
            ["energy"],
        ),
        (
            "aux.onnx",
            _AuxGraph(net),
            (torch.zeros(1, IN_PLANES, N, N, device=dev),),
            ["planes"],
            ["hex", "necro", "logz"],
        ),
    ]
    with torch.no_grad():
        for name, mod, args, ins, outs in specs:
            path = os.path.join(out_dir, name)
            tmp = path + ".tmp"
            torch.onnx.export(
                mod, args, tmp,
                input_names=ins, output_names=outs,
                dynamic_axes={k: {0: "b"} for k in (*ins, *outs)},
                opset_version=17, dynamo=False,
            )
            os.replace(tmp, path)
    if was_training:
        net.train()


# ── checkpoint I/O ────────────────────────────────────────────────────────────


def arch_of(net: MinionsNet) -> dict:
    return {
        "kind": "minions",
        "channels": net.channels,
        "blocks": net.blocks_n,
        "energy_blocks": net.energy_blocks_n,
    }


def build_model(arch: dict) -> MinionsNet:
    return MinionsNet(
        channels=int(arch.get("channels", 96)),
        blocks=int(arch.get("blocks", 8)),
        energy_blocks=int(arch.get("energy_blocks", 2)),
    )


def save_random(out_dir: str, channels: int = 96, blocks: int = 8, seed: int = 0) -> str:
    """Seed generation 0, the way hex_net.py does, so the actor has weights to start from."""
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    net = MinionsNet(channels=channels, blocks=blocks)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, ckpt)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"arch": arch_of(net), "generation": 0}, fh, indent=2)
    return ckpt


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="write a random Minions checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = save_random(args.out, args.channels, args.blocks, args.seed)
    print(f"wrote {path}")
