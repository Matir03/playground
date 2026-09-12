"""Minions learner: reads the actor's replay chunks, fits the four heads, and
publishes weights the actor hot-reloads.

Shaped after ``hex_az_learner.py`` — the only learner here with real mileage:
an in-memory cache synced to the chunk files on disk, uniform sampling with
replacement from the merged window, and an atomic ``latest.safetensors`` write
every N steps that the actor picks up by mtime.

Four loss terms::

    L = value MSE
      + w_diff * masked declaration CE       (trains the sampler)
      + w_energy * candidate-set CE          (trains the policy of record)

The declaration CE is the BERT/D3PM objective: take a declaration the search
liked, hide a random fraction of the live variables, and predict them from the
rest. The candidate-set CE fits ``softmax f`` over one node's candidates to its
**pruned** visit distribution — forced playouts and injected random candidates
were already subtracted by the actor, so exploration is never taught as if the
search had chosen it.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from playground.model_server.minions_net import (
    CELLS, DECL_PLANES, ENERGY_PLANES, IN_PLANES, N, N_OFFSETS, N_SPAWNS, N_TARGETS,
    MinionsNet, arch_of, build_model, energy_loss, export_onnx_atomic,
    masked_declaration_loss, masked_declaration_loss_soft, SCORE_SCALE,
)

# ── record layout — must match crates/games/minions/src/selfplay.rs ────────────
MAGIC = 0x4D494E33
VERSION = 9  # must match selfplay::VERSION in Rust
# v4 records are identical except that they predate the trailing "turns left"
# feature plane, so they are one plane short. They are still usable: the plane is
# appended last, and the stem weights that read it start at zero (see
# python/minions_migrate_planes.py), so a v4 example padded with zeros is exactly
# what the net saw when it produced that example. They age out of the sliding
# window within ~10 generations.
# v5 and earlier predate `is_full` and the turns-left plane; they are not
# read back, since the stride differs in two places now.
# v8 only. v6 and v7 have the same *layout* as v8 but not the same *meaning*:
# both were recorded under a 20-turn cap, so their turns-left plane tops out at
# 20 and their score's speed term is anchored to 20 rather than 30. Reading them
# would train the net on two contradictory games at once, which no amount of
# masking fixes -- unlike the v6 score field, which was merely absent.
#
# The buffer holds a single generation, so this costs one cycle of data and
# heals itself. `read_chunk` skips unknown versions rather than raising.
COMPAT_VERSIONS = {VERSION: IN_PLANES}
SCORELESS_VERSIONS = set()
# Every child of a full-search root, zero targets included; must match
# selfplay::TOP_K. v8 stored only the top 8 with non-zero targets, so v8 chunks
# are not in COMPAT_VERSIONS -- same stride would not even be the question, the
# candidate set means something different.
TOP_K = 16
SPEC_BYTES = 3 * CELLS
PLANE_BYTES = IN_PLANES * CELLS
CAND_BYTES = SPEC_BYTES + 4
# Auxiliary "rest of the game" targets, all in the canonical frame (slot 0 is the
# side to move). Hex control is a byte-quantised fraction, necromancer survival a
# 0/1 hazard curve per side, unit selection a small non-negative rate.
NECRO_HORIZON = 10
N_BUYABLE = 5
HEX_BYTES = 2 * CELLS
NECRO_BYTES = 2 * NECRO_HORIZON
SCORE_BYTES = 4
EXAMPLE_BYTES = (
    PLANE_BYTES + 1 + 4 + CELLS + 1 + TOP_K * CAND_BYTES
    + HEX_BYTES + NECRO_BYTES + 1 + SCORE_BYTES
)

O_PLANES = 0
O_TOMOVE = PLANE_BYTES
O_VALUE = O_TOMOVE + 1
O_LIVE = O_VALUE + 4
O_NCAND = O_LIVE + CELLS
O_CANDS = O_NCAND + 1
O_HEX = O_CANDS + TOP_K * CAND_BYTES
O_NECRO = O_HEX + HEX_BYTES
O_ISFULL = O_NECRO + NECRO_BYTES
O_SCORE = O_ISFULL + 1

NO_MOVE = 255
C_MOVE, C_TARGET, C_SPAWN = 1, 2, 4


def _offsets() -> np.ndarray:
    """The 37 (dx, dy) move offsets, in the same order as ``turn::OFFSETS``."""
    out = [(0, 0)]
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            if max(abs(dx), abs(dy), abs(dx + dy)) <= 3:
                out.append((dx, dy))
    assert len(out) == N_OFFSETS, len(out)
    return np.array(out, dtype=np.int8)


OFFSETS = _offsets()


def _offset_distance() -> np.ndarray:
    """``(N_OFFSETS, N_OFFSETS)`` hex distance between two offsets' destinations.

    Independent of the origin cell: the destinations differ by ``o_j - o_k``, and
    hex distance is a function of that difference alone.
    """
    d = OFFSETS[:, None, :].astype(np.int32) - OFFSETS[None, :, :].astype(np.int32)
    ax, ay = d[:, :, 0], d[:, :, 1]
    return np.maximum(np.maximum(np.abs(ax), np.abs(ay)), np.abs(ax + ay))


def _move_dest_valid() -> np.ndarray:
    """``(CELLS, N_OFFSETS)`` bool — whether the offset lands on the board.

    Mirrors ``turn::build_move_dest``: same bounds test, same ``x = c % N``,
    ``y = c // N`` indexing. Off-board offsets must get zero smoothing weight or
    the target would actively teach illegal declarations.
    """
    c = np.arange(CELLS)
    x = (c % N)[:, None] + OFFSETS[None, :, 0]
    y = (c // N)[:, None] + OFFSETS[None, :, 1]
    return (x >= 0) & (x < N) & (y >= 0) & (y < N)


OFFSET_DIST = _offset_distance()
MOVE_DEST_VALID = _move_dest_valid()


def smoothed_move_target(moves: np.ndarray, sigma: float) -> np.ndarray:
    """Geometric label smoothing for the 37-way move head.

    ``moves`` is ``(B, CELLS)`` offset indices; returns ``(B, CELLS, N_OFFSETS)``
    probabilities. The hard label at offset ``k`` is spread over nearby offsets
    with weight ``exp(-dist^2 / 2 sigma^2)``, restricted to offsets that stay on
    the board.

    Why this and not a one-hot: measured on the trained net, the correlation
    between move-offset similarity and the learned prior is **-0.108**, with only
    3.7% of offset pairs positive — the head has learned no spatial structure at
    all, because nothing in a one-hot target says that stepping one hex west and
    two hexes west are nearly the same declaration. This puts that geometry in
    the target directly.

    ``sigma <= 0`` returns the one-hot target, so it is an exact identity.
    """
    b = moves.shape[0]
    idx = np.clip(moves, 0, N_OFFSETS - 1).astype(np.int64)
    if sigma <= 0.0:
        out = np.zeros((b, CELLS, N_OFFSETS), dtype=np.float32)
        np.put_along_axis(out, idx[:, :, None], 1.0, axis=2)
        return out
    w = np.exp(-(OFFSET_DIST.astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    out = w[idx]                                     # (B, CELLS, N_OFFSETS)
    out = out * MOVE_DEST_VALID[None, :, :]          # zero off-board offsets
    # The label itself is always legal, so the row can never be all-zero; the
    # guard is against a future offset table where that stops holding.
    total = out.sum(axis=2, keepdims=True)
    return (out / np.maximum(total, 1e-12)).astype(np.float32)



def read_chunk(path: str) -> dict | None:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 16:
        return None
    magic, version = np.frombuffer(raw[:8].tobytes(), dtype="<u4")
    count = int(np.frombuffer(raw[8:16].tobytes(), dtype="<u8")[0])
    if magic != MAGIC or int(version) not in COMPAT_VERSIONS:
        # Skip rather than raise. A record-format bump leaves older chunks in the
        # sliding window, and an unattended run must not die on them — the actor
        # replaces them within a generation anyway. Raising here once killed a
        # cycle mid-run, which is exactly the failure mode this avoids.
        print(
            f"skipping {os.path.basename(path)}: format {magic:#x} v{version}, "
            f"expected {MAGIC:#x} v{VERSION}",
            file=sys.stderr,
        )
        return None
    body = raw[16:]
    # Older versions differ in the plane count, which shifts every offset, and
    # in whether the trailing score field is present.
    n_planes = COMPAT_VERSIONS[int(version)]
    scoreless = int(version) in SCORELESS_VERSIONS
    plane_bytes = n_planes * CELLS
    stride = EXAMPLE_BYTES - (PLANE_BYTES - plane_bytes) - (SCORE_BYTES if scoreless else 0)
    if body.size != count * stride:
        raise ValueError(
            f"{path}: expected {count * stride} body bytes for v{version}, got {body.size}"
        )
    rec = body.reshape(count, stride)
    if scoreless:
        # Append the missing score field so every downstream offset is the
        # current layout's. `has_score` is what keeps these rows out of the
        # score loss; the zeros here are never trained against.
        rec = np.concatenate(
            [rec, np.zeros((count, SCORE_BYTES), dtype=np.uint8)], axis=1
        )
    if plane_bytes != PLANE_BYTES:
        # Pad the missing trailing plane(s) with zeros and re-align the rest.
        pad = np.zeros((count, PLANE_BYTES - plane_bytes), dtype=np.uint8)
        rec = np.concatenate([rec[:, :plane_bytes], pad, rec[:, plane_bytes:]], axis=1)
    cands = rec[:, O_CANDS:O_HEX].reshape(count, TOP_K, CAND_BYTES)
    return {
        "planes": rec[:, O_PLANES:O_TOMOVE].copy(),
        "value": rec[:, O_VALUE:O_VALUE + 4].copy().view("<f4").reshape(count),
        "live": rec[:, O_LIVE:O_LIVE + CELLS].copy(),
        "n_cands": rec[:, O_NCAND].copy(),
        "spec": cands[:, :, :SPEC_BYTES].copy(),
        "prob": cands[:, :, SPEC_BYTES:].copy().reshape(-1, 4).view("<f4").reshape(count, TOP_K),
        # Fractions stored as bytes; rescaled once here rather than per step.
        "hex_control": (
            rec[:, O_HEX:O_HEX + HEX_BYTES].astype(np.float32) / 255.0
        ).reshape(count, 2, CELLS),
        "necro_safety": rec[:, O_NECRO:O_NECRO + NECRO_BYTES]
        .astype(np.float32)
        .reshape(count, 2, NECRO_HORIZON),
        # Playout-cap randomization: only full searches carry a policy target.
        "is_full": rec[:, O_ISFULL].copy(),
        # Terminal score in raw points, from the mover's seat.
        "score": rec[:, O_SCORE:O_SCORE + SCORE_BYTES].copy().view("<f4").reshape(count),
        "has_score": np.full(count, 0 if scoreless else 1, dtype=np.uint8),
    }


class Buffer:
    """Chunk cache kept in sync with the actor's sliding window."""

    def __init__(self, buffer_dir: str):
        self.dir = buffer_dir
        self.cache: dict[str, dict] = {}

    def refresh(self) -> int:
        on_disk = set(glob.glob(os.path.join(self.dir, "chunk_*.bin")))
        for gone in set(self.cache) - on_disk:
            del self.cache[gone]
        for path in sorted(on_disk - set(self.cache)):
            try:
                d = read_chunk(path)
            except (ValueError, OSError) as exc:
                print(f"skipping {path}: {exc}")
                continue
            if d is not None:
                self.cache[path] = d
        return self.size()

    def size(self) -> int:
        return sum(len(d["value"]) for d in self.cache.values())

    def merged(self) -> dict:
        keys = ["planes", "value", "live", "n_cands", "spec", "prob",
                "hex_control", "necro_safety", "is_full", "score", "has_score"]
        parts = [self.cache[k] for k in sorted(self.cache)]
        return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def decl_planes(spec: np.ndarray, committed: np.ndarray) -> np.ndarray:
    """Build ``(B, DECL_PLANES, N, N)`` — must match ``Partial::write_planes``.

    ``spec`` is (B, 300) raw bytes, ``committed`` is (B, CELLS, 3) bool for the
    move/target/spawn fields.
    """
    b = spec.shape[0]
    out = np.zeros((b, DECL_PLANES, CELLS), dtype=np.float32)
    moves = spec[:, :CELLS].astype(np.int64)
    targets = spec[:, CELLS:2 * CELLS].astype(np.int64)
    spawns = spec[:, 2 * CELLS:].astype(np.int64)

    cm = committed[:, :, 0]
    valid = cm & (moves < N_OFFSETS)
    idx = np.clip(moves, 0, N_OFFSETS - 1)
    out[:, 0, :] = np.where(valid, OFFSETS[idx, 0] / 3.0, 0.0)
    out[:, 1, :] = np.where(valid, OFFSETS[idx, 1] / 3.0, 0.0)
    out[:, 2, :] = cm.astype(np.float32)

    ct = committed[:, :, 1]
    t = np.clip(targets, 0, N_TARGETS - 1)
    for k in range(N_TARGETS):
        out[:, 3 + k, :] = ((t == k) & ct).astype(np.float32)

    cs = committed[:, :, 2]
    sp = np.clip(spawns, 0, N_SPAWNS - 1)
    for k in range(N_SPAWNS):
        out[:, 6 + k, :] = ((sp == k) & cs).astype(np.float32)
    return out.reshape(b, DECL_PLANES, N, N)


def energy_planes(spec: np.ndarray, committed: np.ndarray) -> np.ndarray:
    """``ENERGY_PLANES``-wide encoding: `decl_planes` without the move-commit
    plane. The energy head only scores complete declarations, where that flag is
    just a liveness mask the trunk features already carry."""
    d = decl_planes(spec, committed)
    return np.concatenate([d[:, :2], d[:, 3:]], axis=1)


def sample_tb_specs(batch: dict, idx: np.ndarray, k: int):
    """Up to `k` stored candidates per position, for trajectory balance.

    TB replays *recorded* turns, and until now it replayed exactly one per
    position — so the sampler was being fit to `exp f` on a single turn that
    search had already favoured. The record already holds up to `TOP_K`
    realized candidates per position at no extra cost, and realized is what
    matters: the energy head is trained and evaluated on realized specs
    (`sampler.rs` realizes before scoring), so freshly sampled declarations
    would be out of distribution for the reward.

    Returns `(specs, rows)` where `rows` indexes back into `idx`, so callers can
    line up planes and live masks with the expanded candidate set.
    """
    n = batch["n_cands"][idx].astype(np.int64)
    specs, rows = [], []
    for r, cnt in enumerate(n):
        take = int(min(max(cnt, 1), k))
        for c in range(take):
            specs.append(batch["spec"][idx[r], c])
            rows.append(r)
    return np.stack(specs), np.asarray(rows, dtype=np.int64)


def sample_target_spec(batch: dict, idx: np.ndarray, rng: np.random.Generator):
    """Pick one candidate per example, proportional to its pruned visit share."""
    probs = batch["prob"][idx].astype(np.float64)
    # int64, not the record's uint8: fast rows have `n_cands == 0`, and `n - 1`
    # on an unsigned type wraps to 255, which silently defeats the clamp below.
    n = batch["n_cands"][idx].astype(np.int64)
    mask = np.arange(TOP_K)[None, :] < n[:, None]
    probs = np.where(mask, np.maximum(probs, 0.0), 0.0)
    total = probs.sum(axis=1, keepdims=True)
    # A row with no mass falls back to uniform over its populated slots.
    flat = np.where(total > 0, probs / np.maximum(total, 1e-12), mask / np.maximum(n[:, None], 1))
    draw = (flat.cumsum(axis=1) < rng.random((len(idx), 1))).sum(axis=1)
    draw = np.clip(draw, 0, np.maximum(n - 1, 0))
    return batch["spec"][idx, draw]


def committed_after(t: int, total_steps: int, n: int) -> int:
    """Mirror of ``MinionsModel::committed_after`` — the cosine commit schedule."""
    if t >= total_steps:
        return n
    frac = 1.0 - math.cos(math.pi / 2 * t / total_steps)
    return int(math.ceil(n * frac))


def trajectory_balance_loss(net, h, hh, spec, live_f, reward, t_steps, rng, device):
    """GFlowNet trajectory balance for the masked-diffusion sampler.

    The sampler is already a constructive process over a DAG: each round commits
    a subset of the still-masked declaration variables, and a complete turn is a
    terminal state. Trajectory balance therefore applies directly, and we can
    keep generation *identical* while changing only what the head is fit to.

        L = ( log Z(s) + sum_t log P_F(s_t -> s_t+1) - f(s, a) )^2

    Two things make this tractable here:

    * The commit order is now uniformly random and value-independent, so
      ``P_F`` factorises into a plain product of per-variable conditionals. Under
      the old confidence ordering the order depended on the sampled values and
      ``log P_F`` would have required summing over every order that could have
      produced the sample.
    * The combinatorial "which subset commits this round" factor is the same for
      every trajectory from a given state and does not depend on the parameters,
      so it is absorbed into the state-conditional ``log Z(s)`` rather than
      computed.

    ``reward`` is ``f(s, a)`` from the **frozen** energy net, so log R needs no
    exponentiation and Z(s) is a fixed constant while the sampler is fit.

    Why this instead of masked cross-entropy: MLE fits the sampler to pi_hat,
    which only ever contains turns the sampler already proposed — so it can
    quietly re-certify its own support. This fits ``exp f`` instead, which is
    defined over turns the sampler has never drawn.
    """
    b = spec.shape[0]
    n_live = live_f.sum(axis=(1, 2))  # variables per example

    # One uniformly random commit order per example, as a permutation of the
    # (cell, field) slots. Field-major flattening matches `live_f`'s layout.
    flat_live = live_f.reshape(b, -1)
    order = np.argsort(rng.random(flat_live.shape) - flat_live.astype(np.float64), axis=1)

    logpf = torch.zeros(b, device=device)
    committed = np.zeros_like(flat_live)
    prev = np.zeros(b, dtype=np.int64)
    for t in range(1, t_steps + 1):
        want = np.array([committed_after(t, t_steps, int(n)) for n in n_live])
        # Context = everything committed in earlier rounds.
        decl = torch.from_numpy(
            decl_planes(spec, committed.reshape(live_f.shape))
        ).to(device)
        mv_l, tg_l, sp_l = net.diffusion_head(hh, decl)
        logits = torch.cat([mv_l, tg_l, sp_l], dim=2)  # (B, CELLS, 46)

        # Slots committed *this* round, per example.
        step_mask = np.zeros_like(flat_live)
        for i in range(b):
            lo, hi = int(prev[i]), int(min(want[i], n_live[i]))
            if hi > lo:
                step_mask[i, order[i, lo:hi]] = 1
        prev = np.minimum(want, n_live)

        if step_mask.sum() == 0:
            committed = np.maximum(committed, step_mask)
            continue

        # log p of the *actual* recorded value for each newly committed slot.
        lp = _slot_log_probs(logits, spec, step_mask, device)
        logpf = logpf + lp
        committed = np.maximum(committed, step_mask)

    log_z = net.log_z(h)
    return ((log_z + logpf - reward) ** 2).mean()


def _slot_log_probs(logits, spec, step_mask, device):
    """Sum of log-softmax at the recorded class, over the slots in ``step_mask``.

    Each declaration field owns a contiguous block of the 46 logits, and each is
    normalised over *its own* classes — the fields are separate categoricals, not
    one 46-way choice.
    """
    b = spec.shape[0]
    sm = torch.from_numpy(step_mask.reshape(b, 3, CELLS)).to(device)
    out = torch.zeros(b, device=device)
    blocks = (
        (0, N_OFFSETS, 0),                                   # moves
        (N_OFFSETS, N_TARGETS, 1),                           # targets
        (N_OFFSETS + N_TARGETS, N_SPAWNS, 2),                # spawns
    )
    for off, width, field in blocks:
        tgt = np.clip(
            spec[:, field * CELLS:(field + 1) * CELLS].astype(np.int64), 0, width - 1
        )
        tgt_t = torch.from_numpy(tgt).to(device)
        lsm = F.log_softmax(logits[:, :, off:off + width], dim=2)
        picked = lsm.gather(2, tgt_t.unsqueeze(-1)).squeeze(-1)  # (B, CELLS)
        out = out + (picked * sm[:, field].float()).sum(dim=1)
    return out


def train_block(net, opt, batch, steps, bs, device, rng, w_diff, w_energy, w_aux=0.15,
                tb_weight=0.0, t_steps=4, move_smoothing=0.0, tb_candidates=1,
                w_score=0.15):
    """One generation of updates.

    With ``tb_weight > 0`` this runs the two-phase schedule: every step first
    fits value / aux / energy / declaration as usual, then — with the energy head
    **frozen** — fits the sampler by trajectory balance against it.

    Freezing matters. Trajectory balance learns ``log Z(s)``, the log-partition
    of ``exp f(s, .)``; if ``f`` moved in the same step, ``log Z`` would be
    chasing a target that shifts under it and the residual could not settle. With
    ``f`` held fixed for the sampler update, Z(s) is a constant per state and the
    regression is stationary within the generation.
    """
    n = len(batch["value"])
    tally = np.zeros(8)
    tb_tally = 0.0
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, n, size=bs)
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).to(device)
        planes = planes.view(bs, IN_PLANES, N, N)
        h = net.trunk(planes)
        hh = net.head_features(h)

        v_pred = net.value_head(h)
        loss_v = F.mse_loss(v_pred, torch.from_numpy(batch["value"][idx]).to(device))

        # ── policy targets: full searches only ────────────────────────────────
        #
        # Playout-cap randomization. Every row trains value and the auxiliaries,
        # because those targets come from the game's outcome, not the search. But
        # a 50-simulation visit count is a poor policy target, so the declaration
        # and energy losses see only the 200-simulation rows.
        full_rows = batch["is_full"][idx].astype(bool)
        full_t = torch.from_numpy(full_rows).to(device)
        spec = sample_target_spec(batch, idx, rng)
        live = batch["live"][idx]
        live_f = np.stack(
            [(live & C_MOVE) > 0, (live & C_TARGET) > 0, (live & C_SPAWN) > 0], axis=2
        )
        # One mask ratio per example, uniform in (0, 1].
        ratio = rng.random((bs, 1, 1))
        hidden = live_f & (rng.random(live_f.shape) < ratio) & full_rows[:, None, None]
        committed = live_f & ~hidden
        decl = torch.from_numpy(decl_planes(spec, committed)).to(device)
        mv_l, tg_l, sp_l = net.diffusion_head(hh, decl)
        moves_t = torch.from_numpy(np.clip(spec[:, :CELLS], 0, N_OFFSETS - 1).astype(np.int64))
        tg_t = torch.from_numpy(np.clip(spec[:, CELLS:2 * CELLS], 0, N_TARGETS - 1).astype(np.int64))
        sp_t = torch.from_numpy(np.clip(spec[:, 2 * CELLS:], 0, N_SPAWNS - 1).astype(np.int64))
        hm = torch.from_numpy(hidden[:, :, 0]).to(device)
        ht = torch.from_numpy(hidden[:, :, 1]).to(device)
        hs = torch.from_numpy(hidden[:, :, 2]).to(device)
        # Only the move field gets geometric smoothing: its 37 classes are hexes
        # with a real metric between them. Target classes (stay/kill/unsummon)
        # and spawn classes (which unit) are unordered — there is no sense in
        # which "kill" is near "unsummon", so smoothing them would just blur the
        # label.
        mv_target_np = smoothed_move_target(spec[:, :CELLS], move_smoothing)
        mv_target = torch.from_numpy(mv_target_np).to(device)
        # Smoothing puts irreducible entropy into the target: the best possible
        # cross-entropy is H(target), not 0. Without logging it, turning
        # smoothing on looks like `loss_declaration` regressing by ~1.25 nats at
        # sigma 0.45 when nothing has got worse — the same entropy-floor
        # confusion that made `loss_energy` look flat. The comparable quantity
        # is `decl_kl = loss_declaration - decl_entropy`.
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(mv_target_np * np.log(np.maximum(mv_target_np, 1e-12))).sum(axis=2)
        hm_np = hidden[:, :, 0]
        h_decl = float(ent[hm_np].mean()) if hm_np.any() else 0.0
        loss_d = (
            masked_declaration_loss_soft(mv_l, mv_target, hm)
            + masked_declaration_loss(tg_l, tg_t.to(device), ht)
            + masked_declaration_loss(sp_l, sp_t.to(device), hs)
        )

        # ── candidate-set CE for the energy head ──────────────────────────────
        n_c = batch["n_cands"][idx]
        specs = batch["spec"][idx].reshape(bs * TOP_K, SPEC_BYTES)
        comm = np.repeat(live_f, TOP_K, axis=0)
        decl_all = torch.from_numpy(energy_planes(specs, comm)).to(device)
        h_rep = hh.repeat_interleave(TOP_K, dim=0)
        scores = net.energy_head(h_rep, decl_all).view(bs, TOP_K)
        slot_mask = torch.from_numpy(
            (np.arange(TOP_K)[None, :] < n_c[:, None])
        ).to(device)
        scores = scores.masked_fill(~slot_mask, -1e9)
        visits = torch.from_numpy(
            np.where(np.arange(TOP_K)[None, :] < n_c[:, None], batch["prob"][idx], 0.0)
        ).to(device)
        usable = (visits.sum(dim=1) > 0) & full_t
        loss_e = energy_loss(scores[usable], visits[usable]) if usable.any() else scores.sum() * 0
        # `energy_loss` is the cross-entropy H(pi_hat, softmax f) = H(pi_hat) +
        # KL(pi_hat || softmax f). The entropy term is a property of the search
        # target, not of the net, and it is the floor the loss cannot go below —
        # so logging it turns `loss_energy` into a readable divergence:
        #   KL = loss_energy - target_entropy.
        # Without it a "flat" energy loss is ambiguous between "not learning" and
        # "already at the floor".
        with torch.no_grad():
            if usable.any():
                v = visits[usable]
                tgt = v / v.sum(dim=1, keepdim=True).clamp_min(1e-9)
                h_t = -(tgt * torch.log(tgt.clamp_min(1e-9))).sum(dim=1).mean()
            else:
                h_t = torch.zeros((), device=device)

        # ── auxiliary "rest of the game" heads ────────────────────────────────
        #
        # Their value is the gradient they push into the shared trunk, not the
        # predictions: they force it to represent territory, danger and economy
        # over a horizon the scalar value target only summarises. Hex control and
        # necromancer survival are probabilities (BCE, soft targets in the first
        # case); unit selection is a non-negative rate, so plain MSE against a
        # softplus output.
        hex_l, necro_l = net.aux_head(h)
        hex_t = torch.from_numpy(batch["hex_control"][idx]).to(device)
        necro_t = torch.from_numpy(batch["necro_safety"][idx]).to(device)
        loss_aux = (
            F.binary_cross_entropy_with_logits(hex_l, hex_t)
            + F.binary_cross_entropy_with_logits(necro_l, necro_t)
        )

        # ── terminal-score head ───────────────────────────────────────────────
        #
        # Masked to rows whose chunk actually carried a score: a record written
        # before the field existed has a placeholder 0, which is a wrong label
        # rather than a neutral one, and training on it would teach the head
        # that every old position was a dead draw.
        score_t = torch.from_numpy(
            batch["score"][idx] / SCORE_SCALE
        ).to(device)
        has_s = torch.from_numpy(
            batch["has_score"][idx].astype(bool)
        ).to(device)
        score_p = net.score_head(h)
        if has_s.any():
            loss_score = F.mse_loss(score_p[has_s], score_t[has_s])
        else:
            loss_score = score_p.sum() * 0

        loss = (
            loss_v + w_diff * loss_d + w_energy * loss_e
            + w_aux * loss_aux + w_score * loss_score
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ── phase 2: sampler by trajectory balance, energy frozen ─────────────
        if tb_weight > 0.0:
            with torch.no_grad():
                # The reward for the recorded turn, from the just-updated energy
                # head. `no_grad` is what freezes it: TB shapes the sampler
                # toward f, never f toward the sampler.
                h2 = net.trunk(planes)
                hh2 = net.head_features(h2)
                comm_all = live_f.astype(bool)
                edecl = torch.from_numpy(energy_planes(spec, comm_all)).to(device)
                reward = net.energy_head(hh2, edecl)
            h2 = net.trunk(planes)
            hh2 = net.head_features(h2)
            # Same restriction: TB replays a recorded turn, so it needs a real
            # policy target behind it.
            tb_live = live_f * full_rows[:, None, None]
            if tb_candidates > 1:
                # Expand to several stored candidates per position. Each is a
                # realized turn the search actually looked at, so the frozen
                # energy head scores them in-distribution; more of them per
                # position simply gives TB more of the support it is meant to
                # cover than the single best-visited turn does.
                tb_spec, tb_rows = sample_tb_specs(batch, idx, tb_candidates)
                rows_t = torch.from_numpy(tb_rows).to(device)
                h2e, hh2e = h2[rows_t], hh2[rows_t]
                reward_e = reward[rows_t] if reward.dim() == 1 else reward[rows_t].squeeze(-1)
                loss_tb = trajectory_balance_loss(
                    net, h2e, hh2e, tb_spec, tb_live[tb_rows], reward_e,
                    t_steps, rng, device
                )
            else:
                loss_tb = trajectory_balance_loss(
                    net, h2, hh2, spec, tb_live, reward, t_steps, rng, device
                )
            opt.zero_grad(set_to_none=True)
            (tb_weight * loss_tb).backward()
            opt.step()
            tb_tally += float(loss_tb.detach())
        tally += [
            loss_v.item(),
            float(h_t),
            float(loss_d.detach()),
            float(loss_e.detach()),
            float(loss_aux.detach()),
            h_decl,
            float(loss_score.detach()),
            # Mean absolute error in *points*, which is the number worth
            # reading: an MSE on a /100 target is not interpretable next to the
            # score values the eval harness reports.
            float(
                (score_p[has_s] - score_t[has_s]).abs().mean().detach() * SCORE_SCALE
            ) if bool(has_s.any()) else 0.0,
        ]
    return np.append(tally / max(steps, 1), tb_tally / max(steps, 1))


def save_atomic(net, path):
    tmp = path + ".tmp"
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Minions sampled-AZ learner")
    ap.add_argument("--live", required=True)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-diff", type=float, default=1.0)
    ap.add_argument("--w-energy", type=float, default=1.0)
    # Auxiliary rest-of-game heads (hex control, necromancer survival, unit
    # selection). Modest weight: these are there to shape the trunk,
    # not to compete with the value and policy objectives.
    ap.add_argument("--aux-weight", type=float, default=0.15)
    ap.add_argument("--move-smoothing", type=float, default=0.0,
                    help="hex bandwidth for geometric label smoothing on the 37-way "
                         "move head. 0 = one-hot targets, exactly today's behaviour. "
                         "Mass kept on the true label at a central cell: 0.99 at "
                         "sigma 0.25, 0.79 at 0.40, 0.55 at 0.50, 0.16 at 1.00 - so "
                         "0.4-0.5 is the useful band and 1.0 is far too flat.")
    ap.add_argument("--score-weight", type=float, default=0.15,
                    help="weight on the terminal-score head. Same rationale as "
                         "--aux-weight: it is there to shape the trunk with a "
                         "signal the saturating value head loses, not to compete "
                         "with the value and policy objectives.")
    ap.add_argument("--tb-candidates", type=int, default=1,
                    help="stored candidates per position to run trajectory balance "
                         "over. 1 replays only the best-visited turn, which is what "
                         "TB was meant to avoid; up to TOP_K are available free.")
    ap.add_argument("--tb-weight", type=float, default=0.0,
                    help="weight on the GFlowNet trajectory-balance loss for the "
                         "sampler; 0 keeps the masked-CE-only behaviour")
    ap.add_argument("--t-steps", type=int, default=4,
                    help="commit rounds to replay in trajectory balance; must "
                         "match the actor's --t-steps")
    ap.add_argument("--min-samples", type=int, default=2000)
    ap.add_argument("--gen-examples", type=int, default=0,
                    help="wait for this many fresh examples before training")
    ap.add_argument("--exit-after-gens", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    live = args.live
    os.makedirs(os.path.join(live, "buffer"), exist_ok=True)
    os.makedirs(os.path.join(live, "snapshots"), exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    cfg_path = os.path.join(live, "config.json")
    latest = os.path.join(live, "latest.safetensors")
    if os.path.exists(cfg_path) and os.path.exists(latest):
        cfg = json.load(open(cfg_path))
        net = build_model(cfg["arch"]).to(device)
        # Tolerate *new* heads, refuse anything else.
        #
        # Adding an auxiliary head mid-run leaves the checkpoint without its
        # parameters, and a strict load then dies — which killed a cycle at
        # generation 5. Blanket `strict=False` would be worse: it also swallows
        # renamed or dropped trunk tensors, and those parameter names are a wire
        # format shared with the candle loader, so a silent mismatch there is
        # exactly the failure the parity test exists to catch. So: allow missing
        # keys, report them, and fail loudly on any unexpected one.
        missing, unexpected = net.load_state_dict(load_file(latest), strict=False)
        if unexpected:
            raise RuntimeError(
                f"{latest} has parameters this architecture does not: {sorted(unexpected)[:8]}"
            )
        if missing:
            print(f"initialising {len(missing)} new parameters from scratch "
                  f"(e.g. {sorted(missing)[:4]})", file=sys.stderr)
        generation = int(cfg.get("generation", 0))
        print(f"resumed generation {generation}")
    else:
        torch.manual_seed(args.seed)
        net = MinionsNet(args.channels, args.blocks).to(device)
        generation = 0
        save_atomic(net, latest)
        export_onnx_atomic(net, live)
        json.dump({"arch": arch_of(net), "generation": 0}, open(cfg_path, "w"), indent=2)
        print("seeded generation 0")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_path = os.path.join(live, "opt_state.pt")
    if os.path.exists(opt_path):
        # Adam's moments are keyed by parameter index, so they are meaningless
        # once the parameter set changes (a new head shifts every later index).
        # Losing them costs a few hundred steps of re-warming; loading mismatched
        # ones would corrupt the update silently.
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"optimiser state does not match this architecture ({exc}); "
                  f"starting fresh moments", file=sys.stderr)

    buf = Buffer(os.path.join(live, "buffer"))
    gens_done = 0
    seen = buf.refresh()

    while True:
        size = buf.refresh()
        if size < args.min_samples:
            print(f"buffer {size} < {args.min_samples}; waiting")
            time.sleep(20)
            continue
        if args.gen_examples and size - seen < args.gen_examples and gens_done:
            time.sleep(20)
            continue
        seen = size

        batch = buf.merged()
        t0 = time.time()
        lv, hent, ld, le, la, hdecl, lsc, smae, ltb = train_block(
            net, opt, batch, args.steps, args.batch, device, rng,
            args.w_diff, args.w_energy, args.aux_weight,
            tb_weight=args.tb_weight, t_steps=args.t_steps,
            move_smoothing=args.move_smoothing,
            tb_candidates=args.tb_candidates,
            w_score=args.score_weight,
        )
        generation += 1
        gens_done += 1

        save_atomic(net, latest)
        # The ORT/TensorRT actor reads these; the candle actor reads the
        # safetensors. Both are refreshed every generation so a run can switch
        # backends without a stale graph.
        export_onnx_atomic(net, live)
        json.dump({"arch": arch_of(net), "generation": generation},
                  open(cfg_path, "w"), indent=2)
        snap = os.path.join(live, "snapshots", f"gen{generation:03d}")
        os.makedirs(snap, exist_ok=True)
        save_atomic(net, os.path.join(snap, "checkpoint.safetensors"))
        json.dump({"arch": arch_of(net), "generation": generation},
                  open(os.path.join(snap, "config.json"), "w"), indent=2)
        metrics = {
            "generation": generation, "examples": size,
            "loss_value": lv,
            "loss_declaration": ld, "loss_energy": le, "loss_aux": la,
            # KL(pi_hat || softmax f): what the energy head is actually missing,
            # with the target's own entropy (its floor) taken out.
            "energy_entropy": hent, "energy_kl": le - hent,
            # With --move-smoothing 0 this is 0 and decl_kl == loss_declaration,
            # so the numbers stay comparable with every generation trained so far.
            "decl_entropy": hdecl, "decl_kl": ld - hdecl,
            "loss_tb": ltb,
            # MSE on the /SCORE_SCALE target, plus the same error in points,
            # which is the readable one.
            "loss_score": lsc, "score_mae": round(smae, 2),
            "seconds": round(time.time() - t0, 1),
        }
        json.dump(metrics, open(os.path.join(live, "metrics.json"), "w"), indent=2)
        json.dump(metrics, open(os.path.join(snap, "metrics.json"), "w"), indent=2)
        torch.save(opt.state_dict(), opt_path)
        print(f"gen {generation}: {metrics}")

        if args.exit_after_gens and gens_done >= args.exit_after_gens:
            return


if __name__ == "__main__":
    main()
