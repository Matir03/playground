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
VERSION = 10  # must match selfplay::VERSION in Rust
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
#
# v10 stores each record's children at their count, after every record's fixed
# part (see `read_chunk`); v9 held 16 fixed-stride slots inside each record.
# Same planes, same meaning -- v9 is the same search's children cut to the top
# 16 -- so v9 is read and re-laid-out as v10.
COMPAT_VERSIONS = {VERSION: IN_PLANES, 9: IN_PLANES}
# Most children a record holds: the stable children of a full-search root,
# strongest first, zero targets included; must match selfplay::TOP_K.
TOP_K = 160
# The fixed-stride slots each v9 record carried.
V9_SLOTS = 16
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
# The fixed-stride part of a record; its children follow every fixed part.
EXAMPLE_BYTES = (
    PLANE_BYTES + 1 + 4 + CELLS + 1
    + HEX_BYTES + NECRO_BYTES + 1 + SCORE_BYTES
)

O_PLANES = 0
O_TOMOVE = PLANE_BYTES
O_VALUE = O_TOMOVE + 1
O_LIVE = O_VALUE + 4
O_NCAND = O_LIVE + CELLS
O_HEX = O_NCAND + 1
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
    if int(version) == 9:
        # Sixteen slots inside each record, between the count and the hex
        # targets. Cut them out, and the rest is the v10 fixed part.
        o_slots, slot_bytes = O_NCAND + 1, V9_SLOTS * CAND_BYTES
        stride = EXAMPLE_BYTES + slot_bytes
        if body.size != count * stride:
            raise ValueError(
                f"{path}: expected {count * stride} body bytes for v9, got {body.size}"
            )
        rec9 = body.reshape(count, stride)
        slots = rec9[:, o_slots:o_slots + slot_bytes].reshape(count, V9_SLOTS, CAND_BYTES)
        rec = np.concatenate([rec9[:, :o_slots], rec9[:, o_slots + slot_bytes:]], axis=1)
        n = rec[:, O_NCAND].astype(np.int64)
        # Row-major, so each row's populated slots stay together and in order.
        cands = slots[np.arange(V9_SLOTS)[None, :] < n[:, None]]
    else:
        # Every fixed part at one stride, then every row's children back to
        # back, as many as its count says.
        fixed = count * EXAMPLE_BYTES
        if body.size < fixed:
            raise ValueError(f"{path}: {body.size} body bytes, {fixed} of fixed parts alone")
        rec = body[:fixed].reshape(count, EXAMPLE_BYTES)
        n = rec[:, O_NCAND].astype(np.int64)
        want = fixed + int(n.sum()) * CAND_BYTES
        if body.size != want:
            raise ValueError(
                f"{path}: expected {want} body bytes for v{version}, got {body.size}"
            )
        cands = body[fixed:].reshape(-1, CAND_BYTES)
    return {
        "planes": rec[:, O_PLANES:O_TOMOVE].copy(),
        "value": rec[:, O_VALUE:O_VALUE + 4].copy().view("<f4").reshape(count),
        "live": rec[:, O_LIVE:O_LIVE + CELLS].copy(),
        "n_cands": rec[:, O_NCAND].copy(),
        # Every row's children back to back, in row order; `Buffer.merged`
        # turns the counts into offsets.
        "cand_spec": cands[:, :SPEC_BYTES].copy(),
        "cand_prob": cands[:, SPEC_BYTES:].copy().view("<f4").reshape(-1),
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
        "has_score": np.ones(count, dtype=np.uint8),
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
        keys = ["planes", "value", "live", "n_cands", "cand_spec", "cand_prob",
                "hex_control", "necro_safety", "is_full", "score", "has_score"]
        parts = [self.cache[k] for k in sorted(self.cache)]
        out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        # Row i's children are cand_*[cand_off[i]:cand_off[i + 1]]: the chunks'
        # candidate blocks are concatenated in the same order as their rows.
        out["cand_off"] = np.concatenate(
            [[0], np.cumsum(out["n_cands"], dtype=np.int64)]
        ).astype(np.int64)
        return out


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


def tb_trajectories(batch: dict, idx: np.ndarray, k: int, sampled: np.ndarray,
                    full_rows: np.ndarray):
    """The recorded turns trajectory balance replays, as ``(specs, rows)``.

    Full-search rows only. A fast row records no children (``n_cands == 0``),
    so it has no turn to replay; it used to be replayed anyway -- as slot 0,
    which is all zeros -- with its log P_F masked to 0, which left the residual
    ``log Z(s) - f(s, empty)`` and trained the log Z head on a meaningless
    target for most of every batch.

    ``k == 1`` replays the one turn ``sample_target_spec`` drew per row, in
    proportion to the pruned visits. Larger ``k`` replays up to ``k`` of each
    row's children in stored order (most visited first), so ``k >= TOP_K`` is
    every child. The children are realized turns the search expanded, zero-visit
    ones included, so the frozen energy head scores them in distribution -- a
    freshly sampled declaration would not be -- and fitting several per state is
    what pins P_F's *relative* mass across them to ``exp f``, not only its level
    on the favourite.

    ``rows`` indexes into ``idx``, to line up planes and live masks.
    """
    n = batch["n_cands"][idx].astype(np.int64)
    use = np.flatnonzero(full_rows & (n > 0))
    if k <= 1:
        return sampled[use], use
    rows, flat, slot = candidate_rows(batch, idx, use)
    keep = slot < k
    return batch["cand_spec"][flat[keep]], rows[keep]


def apply_no_decay(net, opt, live: str, wd: float) -> None:
    """Exempt the parameters named in ``<live>/no_decay.txt`` from weight decay.

    One regex per line (``re.match`` on the parameter name), ``#`` comments.
    No file, or no match, leaves Adam's own coupled L2 exactly as it was.

    Why it exists: under Adam, L2 is added to the gradient and then normalised,
    so a weight whose loss gradient is ~0 moves toward zero by a full ``lr``
    every step whatever ``wd`` is. A residual block whose last BatchNorm gamma
    is 0 gets exactly zero gradient in its convolutions, so they are erased in
    ~30 steps and the block can never switch on. That is how blocks grown by
    ``minions_grow.py`` died within one generation (2026-09-24), and it is the
    likely way blocks 3, 4 and 6 of the 96x8 run had been dead since before
    gen300.

    The exemption is done by moving decay into gradient hooks rather than a
    second param group, so the saved optimiser state keeps loading: for every
    other parameter ``grad + wd * p`` is what Adam computed internally.
    """
    import re
    path = os.path.join(live, "no_decay.txt")
    if wd == 0 or not os.path.exists(path):
        return
    pats = [re.compile(l.strip()) for l in open(path)
            if l.strip() and not l.strip().startswith("#")]
    exempt = [n for n, _ in net.named_parameters() if any(p.match(n) for p in pats)]
    if not exempt:
        return
    for g in opt.param_groups:
        g["weight_decay"] = 0.0
    keep = set(exempt)
    for n, p in net.named_parameters():
        if n not in keep:
            p.register_hook(lambda grad, p=p: grad + wd * p.detach())
    print(f"weight decay off for {len(exempt)} parameters (no_decay.txt), "
          f"e.g. {exempt[:3]}")


def candidate_rows(batch: dict, idx: np.ndarray, rows: np.ndarray | None = None):
    """The children of ``idx[rows]`` (default: all of ``idx``), flattened.

    Returns ``(row, flat, slot)``, one entry per child: its position in ``idx``,
    its index into ``cand_spec``/``cand_prob``, and its rank within its row
    (0 = strongest). Rows follow ``rows``' order, each row's children together.
    """
    if rows is None:
        rows = np.arange(len(idx))
    # int64, not the record's uint8: fast rows have `n_cands == 0`, and any
    # arithmetic on an unsigned count wraps silently.
    lens = batch["n_cands"][idx[rows]].astype(np.int64)
    starts = batch["cand_off"][idx[rows]]
    run = np.repeat(np.cumsum(lens) - lens, lens)   # where each row's run begins
    slot = np.arange(int(lens.sum()), dtype=np.int64) - run
    return np.repeat(rows, lens), np.repeat(starts, lens) + slot, slot


def sample_target_spec(batch: dict, idx: np.ndarray, rng: np.random.Generator):
    """Pick one candidate per example, proportional to its pruned visit share.

    A row with no mass falls back to uniform over its children. A row with no
    children -- a fast row -- gets the empty turn, all zeros, as it always has;
    nothing is masked for it, so it is never trained against.
    """
    bs = len(idx)
    out = np.zeros((bs, SPEC_BYTES), dtype=np.uint8)
    row, flat, _ = candidate_rows(batch, idx)
    if len(flat) == 0:
        return out
    p = np.maximum(batch["cand_prob"][flat].astype(np.float64), 0.0)
    lens = batch["n_cands"][idx].astype(np.int64)
    total = np.bincount(row, weights=p, minlength=bs)
    u = rng.random(bs)
    has = np.flatnonzero(lens > 0)
    first = (np.cumsum(lens) - lens)[has]   # each row's first child in `flat`
    last = first + lens[has] - 1
    # Inverse CDF within each row: the first child whose running mass passes
    # u * total, clipped to the row against rounding at its edges.
    cum = np.cumsum(p)
    target = cum[first] - p[first] + u[has] * total[has]
    pick = np.clip(np.searchsorted(cum, target, side="right"), first, last)
    uniform = first + np.minimum((u[has] * lens[has]).astype(np.int64), lens[has] - 1)
    pick = np.where(total[has] > 0, pick, uniform)
    out[has] = batch["cand_spec"][flat[pick]]
    return out


def committed_after(t: int, total_steps: int, n: int) -> int:
    """Mirror of ``MinionsModel::committed_after`` — the cosine commit schedule."""
    if t >= total_steps:
        return n
    frac = 1.0 - math.cos(math.pi / 2 * t / total_steps)
    return int(math.ceil(n * frac))


def tb_residuals(net, h, hh, spec, live_f, reward, t_steps, rng, device):
    """GFlowNet trajectory balance for the masked-diffusion sampler: the
    per-trajectory residual ``log Z(s) + log P_F(a) - f(s, a)``.

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
    # (cell, field) slots. `live_f` is (B, CELLS, 3), so the flattening is
    # cell-major with the field innermost, and `_slot_log_probs` must read the
    # mask back in that same layout.
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

    return net.log_z(h) + logpf - reward


def trajectory_balance_loss(net, h, hh, spec, live_f, reward, t_steps, rng, device):
    """Mean squared trajectory-balance residual; see ``tb_residuals``."""
    return (tb_residuals(net, h, hh, spec, live_f, reward, t_steps, rng, device) ** 2).mean()


def _slot_log_probs(logits, spec, step_mask, device):
    """Sum of log-softmax at the recorded class, over the slots in ``step_mask``.

    Each declaration field owns a contiguous block of the 46 logits, and each is
    normalised over *its own* classes — the fields are separate categoricals, not
    one 46-way choice.

    ``step_mask`` is ``(B, CELLS * 3)`` flattened from ``(B, CELLS, 3)``. It was
    read back as ``(B, 3, CELLS)``, which summed log-probs at scrambled slots --
    dead cells included, at their clipped sentinel values -- so the "log P_F"
    trajectory balance fit was not the probability of the recorded turn at all.
    """
    b = spec.shape[0]
    sm = torch.from_numpy(step_mask.reshape(b, CELLS, 3)).to(device)
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
        out = out + (picked * sm[:, :, field].float()).sum(dim=1)
    return out


@torch.no_grad()
def calibrate_log_z(net, batch, rng, device, t_steps, tb_candidates, batches=8, bs=512):
    """Shift the log Z head's output bias so the mean TB residual starts at 0.

    Needed once after the slot-layout fix in ``_slot_log_probs``: log Z had
    been fit against a scrambled log P_F and sat ~20 nats below the true
    log-partition. Left to the optimiser at lr 1e-3 the output would take
    thousands of steps to climb that far, and until it did every TB step would
    be a large, uniform "raise every replayed turn" push on the sampler and the
    shared trunk -- a transient that has nothing to do with matching exp f. A
    constant offset is exactly what a bias absorbs, so it is moved there.

    Returns the shift.
    """
    n = len(batch["value"])
    res = []
    for _ in range(batches):
        idx = rng.integers(0, n, size=bs)
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).to(device)
        planes = planes.view(bs, IN_PLANES, N, N)
        live = batch["live"][idx]
        live_f = np.stack(
            [(live & C_MOVE) > 0, (live & C_TARGET) > 0, (live & C_SPAWN) > 0], axis=2
        )
        full_rows = batch["is_full"][idx].astype(bool)
        spec = sample_target_spec(batch, idx, rng)
        tb_spec, rows = tb_trajectories(batch, idx, tb_candidates, spec, full_rows)
        if not len(rows):
            continue
        rows_t = torch.from_numpy(rows).to(device)
        h = net.trunk(planes)
        hh = net.head_features(h)
        tb_live = live_f[rows]
        edecl = torch.from_numpy(energy_planes(tb_spec, tb_live)).to(device)
        reward = net.energy_head(hh[rows_t], edecl).reshape(-1)
        res.append(tb_residuals(net, h[rows_t], hh[rows_t], tb_spec, tb_live, reward,
                                t_steps, rng, device))
    if not res:
        return 0.0
    r = torch.cat(res)
    shift = -float(r.mean())
    net.logz_fc2.bias += shift
    print(f"calibrated log Z: shifted by {shift:+.2f} over {len(r)} trajectories; "
          f"TB loss {float((r ** 2).mean()):.1f} -> {float(((r + shift) ** 2).mean()):.1f}",
          flush=True)
    return shift


def train_block(net, opt, batch, steps, bs, device, rng, w_diff, w_energy, w_aux=0.15,
                tb_weight=0.0, t_steps=4, move_smoothing=0.0, tb_candidates=1,
                w_score=0.15, energy_children=8192, w_unmask=0.0):
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
    tally = np.zeros(12)
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

        # ── unmasking consistency ─────────────────────────────────────────────
        #
        # The masked CE above trains *hidden* positions only. A committed
        # position's logits are completely unconstrained, so today they carry no
        # usable signal. This term makes the output at a committed field equal
        # the distribution the model would give if that one field were
        # uncommitted — which is what lets a single forward pass report "here is
        # what I would have proposed instead" for every committed field at once,
        # instead of one forward pass per field.
        #
        # **Exactly one field moves.** Pass B commits `committed | {j}`, so "B
        # with j uncommitted" is precisely pass A's input, and A's distribution
        # at j is the exact target. Move two fields and the two contexts differ
        # by more than j, and the target quietly stops being the thing it claims
        # to be — the loss would still fall, and it would be measuring nothing.
        #
        # Training this toward the ground truth instead would collapse to the
        # identity: the committed value is *in* the input, so echoing it is a
        # zero-loss solution that says nothing about alternatives. The target has
        # to be the model's own belief, which is why this is self-distillation.
        loss_u = mv_l.sum() * 0.0
        # Diagnostics for the term above, 0 when it is off. These are what make
        # `loss_unmask` readable at all -- see where they are filled below.
        h_unmask = p_com_b = p_com_a = 0.0
        if w_unmask > 0.0:
            flat_hidden = hidden.reshape(bs, -1)
            has_hidden = flat_hidden.any(axis=1)
            if has_hidden.any():
                # A uniformly random hidden slot per row: hidden slots land in
                # [-1, 0) and the rest in [0, 1), so argmin picks among hidden.
                jsel = np.argmin(
                    rng.random(flat_hidden.shape) - flat_hidden.astype(np.float64), axis=1
                )
                rows_h = np.flatnonzero(has_hidden)
                committed_b = committed.reshape(bs, -1).copy()
                # Only touch rows that *have* a hidden slot; for the rest `jsel`
                # is meaningless and writing it would un-commit a real field.
                committed_b[rows_h, jsel[rows_h]] = True
                decl_b = torch.from_numpy(
                    decl_planes(spec, committed_b.reshape(live_f.shape))
                ).to(device)
                mv_b, tg_b, sp_b = net.diffusion_head(hh, decl_b)
                # live_f is (B, CELLS, 3) flattened cell-major: slot = cell*3 + field.
                cell, field = jsel // 3, jsel % 3
                ces, ents, p_b, p_a = [], [], [], []
                for fi, (la_, lb_) in enumerate(((mv_l, mv_b), (tg_l, tg_b), (sp_l, sp_b))):
                    sel = np.flatnonzero(has_hidden & (field == fi))
                    if not len(sel):
                        continue
                    st = torch.from_numpy(sel).to(device)
                    ct = torch.from_numpy(cell[sel]).to(device)
                    # detach: the target is pass A's belief, held fixed. Without
                    # it both sides drift toward each other and the pair can
                    # satisfy the objective by agreeing on anything at all.
                    tgt = F.softmax(la_[st, ct].detach(), dim=-1)
                    ces.append(-(tgt * F.log_softmax(lb_[st, ct], dim=-1)).sum(dim=-1))
                    # CE >= H(A), so what this objective actually learns is
                    # `loss_unmask - unmask_entropy`. Logging the CE alone hides
                    # a head that is merely sharpening *both* passes: trained
                    # standalone, the KL reached ~0 by step 600 while the CE went
                    # on falling for another 5400 steps.
                    ents.append(-(tgt * torch.log(tgt.clamp_min(1e-12))).sum(dim=-1))
                    # Mass each pass puts on the value actually committed at the
                    # supervised field. Pass B has that value in its input, so
                    # near 1.0 means it is echoing the input and the "what would
                    # I have proposed instead" signal is vacuous. Pass A, which
                    # has the field hidden, is the reference to read it against.
                    tv = (moves_t, tg_t, sp_t)[fi].to(device)[st, ct]
                    p_b.append(
                        F.softmax(lb_[st, ct].detach(), dim=-1)
                        .gather(-1, tv[:, None])
                        .squeeze(-1)
                    )
                    p_a.append(tgt.gather(-1, tv[:, None]).squeeze(-1))
                if ces:
                    # One mean over every supervised position, not a sum over
                    # field groups: which fields appear varies per batch, and
                    # summing would make the term's scale depend on that.
                    loss_u = torch.cat(ces).mean()
                    h_unmask = float(torch.cat(ents).mean())
                    p_com_b = float(torch.cat(p_b).mean())
                    p_com_a = float(torch.cat(p_a).mean())

        # ── candidate-set CE for the energy head ──────────────────────────────
        # Over every recorded child of each usable row -- all of the root's
        # stable children, up to TOP_K -- each scored once, then laid out one
        # row per position with the empty slots masked, so `energy_loss` sees
        # the same padded shape as ever. Fast rows have no children and cost
        # nothing here.
        n_c = batch["n_cands"][idx].astype(np.int64)
        row, flat, _ = candidate_rows(batch, idx)
        mass = np.bincount(row, weights=batch["cand_prob"][flat], minlength=bs)
        use = np.flatnonzero(full_rows & (n_c > 0) & (mass > 0))
        # Cap the children scored per step. The energy head's activations grow
        # with them, and a batch of v10 rows (~130 children each) passes 15k.
        # That is past what the 8 GB card holds, so the Windows driver spills
        # into shared system memory and the step runs ~20x slower: 14 GB
        # reserved, p90 step 6.9 s against a 335 ms median. The cap drops whole
        # rows, never children within one, so every scored row still sees all
        # its visits. `idx` is a random draw, so a prefix of `use` is a random
        # subset. 8192 is what v9's 512 x 16 slots cost, which always fitted.
        if energy_children:
            use = use[np.cumsum(n_c[use]) <= energy_children]
        if len(use):
            row, flat, slot = candidate_rows(batch, idx, use)
            dense = torch.from_numpy(np.searchsorted(use, row)).to(device)
            slot_t = torch.from_numpy(slot).to(device)
            decl_all = torch.from_numpy(
                energy_planes(batch["cand_spec"][flat], live_f[row])
            ).to(device)
            flat_scores = net.energy_head(hh[torch.from_numpy(row).to(device)], decl_all).view(-1)
            scores = flat_scores.new_full((len(use), int(n_c[use].max())), -1e9)
            scores = scores.index_put((dense, slot_t), flat_scores)
            visits = torch.zeros(scores.shape, device=device)
            visits[dense, slot_t] = torch.from_numpy(
                batch["cand_prob"][flat].astype(np.float32)
            ).to(device)
            loss_e = energy_loss(scores, visits)
        else:
            loss_e = hh.sum() * 0
        # `energy_loss` is the cross-entropy H(pi_hat, softmax f) = H(pi_hat) +
        # KL(pi_hat || softmax f). The entropy term is a property of the search
        # target, not of the net, and it is the floor the loss cannot go below —
        # so logging it turns `loss_energy` into a readable divergence:
        #   KL = loss_energy - target_entropy.
        # Without it a "flat" energy loss is ambiguous between "not learning" and
        # "already at the floor".
        with torch.no_grad():
            if len(use):
                tgt = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-9)
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
            + w_unmask * loss_u
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ── phase 2: sampler by trajectory balance, energy frozen ─────────────
        tb_rows = np.zeros(0, dtype=np.int64)
        if tb_weight > 0.0:
            tb_spec, tb_rows = tb_trajectories(batch, idx, tb_candidates, spec, full_rows)
        if len(tb_rows):
            rows_t = torch.from_numpy(tb_rows).to(device)
            tb_live = live_f[tb_rows]
            with torch.no_grad():
                # Each replayed turn's own reward f(s, a), from the just-updated
                # energy head. `no_grad` is what freezes it: TB shapes the
                # sampler toward f, never f toward the sampler. (Scoring one turn
                # per row and handing that score to every child of the row was
                # the old `--tb-candidates` bug.)
                h2 = net.trunk(planes)
                hh2 = net.head_features(h2)
                edecl = torch.from_numpy(energy_planes(tb_spec, tb_live)).to(device)
                reward = net.energy_head(hh2[rows_t], edecl).reshape(-1)
            h2 = net.trunk(planes)
            hh2 = net.head_features(h2)
            loss_tb = trajectory_balance_loss(
                net, h2[rows_t], hh2[rows_t], tb_spec, tb_live, reward, t_steps, rng, device
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
            float(loss_u.detach()),
            h_unmask,
            p_com_b,
            p_com_a,
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
    ap.add_argument("--tb-candidates", type=int, default=TOP_K,
                    help="stored children per full-search position to run trajectory "
                         "balance over, each against its own energy score. The "
                         f"default, {TOP_K} (= TOP_K), is every child; 1 replays one "
                         "turn drawn in proportion to the pruned visits.")
    ap.add_argument("--w-unmask", type=float, default=0.0,
                    help="weight on the unmasking-consistency loss: a committed "
                         "field's logits are trained toward the distribution the "
                         "model gives with that one field uncommitted (its own "
                         "belief, stop-gradient). 0 = exactly today's behaviour. "
                         "Costs one extra diffusion-head forward per step, and "
                         "supervises one committed field per row per step.")
    ap.add_argument("--tb-weight", type=float, default=0.0,
                    help="weight on the GFlowNet trajectory-balance loss for the "
                         "sampler; 0 keeps the masked-CE-only behaviour")
    ap.add_argument("--calibrate-log-z", action="store_true",
                    help="before the first generation, shift the log Z head's output "
                         "bias so the mean TB residual is 0. Use once on a net trained "
                         "before the TB slot-layout fix, whose log Z sits ~20 nats low.")
    ap.add_argument("--t-steps", type=int, default=4,
                    help="commit rounds to replay in trajectory balance; must "
                         "match the actor's --t-steps")
    ap.add_argument("--min-samples", type=int, default=2000)
    ap.add_argument("--gen-examples", type=int, default=0,
                    help="wait for this many fresh examples before training")
    ap.add_argument("--exit-after-gens", type=int, default=0)
    ap.add_argument("--energy-children", type=int, default=8192,
                    help="most children the energy loss scores per step, "
                         "by whole rows; 0 = no cap")
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
    # The saved state carries its group's weight_decay, which would override the
    # flag -- and is 0 whenever an exemption below was in force.
    for g in opt.param_groups:
        g["weight_decay"] = args.weight_decay
    apply_no_decay(net, opt, live, args.weight_decay)

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
        if args.calibrate_log_z and gens_done == 0 and args.tb_weight > 0:
            calibrate_log_z(net, batch, rng, device, args.t_steps, args.tb_candidates)
        t0 = time.time()
        lv, hent, ld, le, la, hdecl, lsc, smae, lu, hu, pcb, pca, ltb = train_block(
            net, opt, batch, args.steps, args.batch, device, rng,
            args.w_diff, args.w_energy, args.aux_weight,
            tb_weight=args.tb_weight, t_steps=args.t_steps,
            move_smoothing=args.move_smoothing,
            tb_candidates=args.tb_candidates,
            w_score=args.score_weight,
            energy_children=args.energy_children,
            w_unmask=args.w_unmask,
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
            # Self-distillation CE at one committed field per row: how far the
            # committed-position output is from what the model would say with
            # that field uncommitted. 0 when --w-unmask is 0.
            "loss_unmask": lu,
            # CE >= H(A), so the consistency actually learned is the KL. A
            # falling loss_unmask alongside falling entropy is the head
            # sharpening both passes, not agreeing better -- read this first.
            "unmask_entropy": hu, "unmask_kl": lu - hu,
            # Mass on the committed value: pass B has it in its input, so ~1.0
            # means echoing and a vacuous reroll signal; pass A is the reference.
            "unmask_p_committed": pcb, "unmask_p_own": pca,
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
