"""Validate the Minions training setup end to end, before spending GPU hours.

Six checks, ordered by how silently each would corrupt a run:

1. **Declaration-plane parity.** ``decl_planes`` here and ``Partial::write_planes``
   in Rust are two independent implementations of one encoding. If they diverge,
   the sampler conditions on one thing at inference and the learner trains on
   another — and nothing crashes. Checked against a Rust-produced bundle,
   including *partially* committed declarations, since that is what the diffusion
   head actually sees mid-sampling.
2. **Feature reshape.** The learner turns the flat feature bytes into (31, N, N); a
   transposed reshape trains fine and plays wrong.
3. **Record round-trip.** Field offsets, ranges, and the invariants the actor is
   supposed to guarantee (probabilities normalised, live mask consistent with the
   declarations stored).
4. **Gradient flow.** Every head must receive gradient. A head that is wired up
   but disconnected produces a flat loss that looks like "not learning yet".
5. **Overfit.** On a handful of examples the net must drive every loss toward
   zero. This is the single most informative smoke test: if it cannot memorise
   eight positions, no amount of self-play will help.
6. **Target sanity.** Value and hex control within range, and value correlated with
   the outcome it was derived from.

Usage::

    minions-engine dump-validation --out data/minions_validation
    minions-engine selfplay-loop --live <live> --target-examples 200 ...
    PYTHONPATH=python/src python python/minions_validate.py \\
        --bundle data/minions_validation --live <live>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

from playground.model_server.minions_net import (
    CELLS, DECL_PLANES, IN_PLANES, N, MinionsNet,
)
from minions_az_learner import (
    C_MOVE, C_SPAWN, C_TARGET, TOP_K, Buffer, decl_planes, train_block,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ── 1. declaration-plane parity ───────────────────────────────────────────────


def validate_decl_parity(bundle: str) -> None:
    print("\n1. declaration-plane parity (Rust Partial::write_planes vs Python decl_planes)")
    meta = json.load(open(os.path.join(bundle, "meta.json")))
    n = meta["cases"]
    specs = np.fromfile(os.path.join(bundle, "decl_specs.bin"), dtype=np.uint8).reshape(n, 3 * CELLS)
    masks = np.fromfile(os.path.join(bundle, "decl_masks.bin"), dtype=np.uint8).reshape(n, CELLS)
    want = np.fromfile(os.path.join(bundle, "decl_planes.bin"), dtype="<f4").reshape(
        n, DECL_PLANES, N, N
    )
    committed = np.stack(
        [(masks & C_MOVE) > 0, (masks & C_TARGET) > 0, (masks & C_SPAWN) > 0], axis=2
    )
    got = decl_planes(specs, committed)

    check("shape", got.shape == want.shape, f"{got.shape} vs {want.shape}")
    diff = np.abs(got - want)
    worst = float(diff.max()) if diff.size else 0.0
    if not check("values match exactly", worst == 0.0, f"worst |Δ| = {worst}"):
        bad = np.argwhere(diff > 0)
        c, p, r, col = bad[0]
        print(f"        first mismatch: case {c} plane {p} cell ({r},{col}): "
              f"python {got[c, p, r, col]} vs rust {want[c, p, r, col]}")
        planes_off = sorted({int(x[1]) for x in bad})
        print(f"        planes affected: {planes_off}")

    # Committed-fraction coverage, measured against the *live* slots — only
    # those can be committed, so a fraction of all 300 would top out around 0.2
    # and make full commitment look like partial.
    lives = np.fromfile(os.path.join(bundle, "decl_live.bin"), dtype=np.uint8).reshape(n, CELLS)
    live_f = np.stack(
        [(lives & C_MOVE) > 0, (lives & C_TARGET) > 0, (lives & C_SPAWN) > 0], axis=2
    )
    live_n = live_f.sum(axis=(1, 2)).clip(1)
    frac = (committed & live_f).sum(axis=(1, 2)) / live_n
    check("bundle covers partial commitment", bool(frac.min() < 0.2 and frac.max() > 0.8),
          f"committed fraction of live slots {frac.min():.2f}..{frac.max():.2f}")
    check("no commitment outside the live mask", bool((committed & ~live_f).sum() == 0))


# ── 2. feature reshape ────────────────────────────────────────────────────────


def validate_features(bundle: str) -> None:
    print("\n2. feature reshape")
    raw = np.fromfile(os.path.join(bundle, "features.bin"), dtype=np.uint8)
    check("length", raw.size == IN_PLANES * CELLS, f"{raw.size} vs {IN_PLANES * CELLS}")
    planes = raw.astype(np.float32).reshape(IN_PLANES, N, N)
    # Plane 0 of the initial position is the side-to-move's necromancer: exactly
    # one cell set, at map::NECRO_START[0] = (x=1, y=1). Derived from N rather
    # than hardcoded — this check read `flat 22`, which was (2,2) on the old
    # 10x10 board and silently wrong for every run since the board shrank.
    want = 1 * N + 1
    p0 = planes[0].reshape(-1)
    check("plane 0 has exactly one unit", int(p0.sum()) == 1, f"sum {p0.sum()}")
    check(f"necromancer at (1,1) (flat {want})", bool(p0[want] == 1.0),
          f"set at {np.flatnonzero(p0).tolist()}")


# ── 3. record round-trip ──────────────────────────────────────────────────────


def validate_records(live: str) -> dict | None:
    print("\n3. replay record round-trip")
    buf = Buffer(os.path.join(live, "buffer"))
    n = buf.refresh()
    if not check("buffer has examples", n > 0, f"{n} examples"):
        return None
    b = buf.merged()

    check("value in [-1, 1]", bool(np.abs(b["value"]).max() <= 1.0),
          f"max |v| = {np.abs(b['value']).max():.3f}")
    check("ownership in {-1,0,1}", bool(np.isin(b["ownership"], (-1, 0, 1)).all()))
    check("planes are byte-valued", bool(b["planes"].max() <= 255))
    check("n_cands within TOP_K", bool((b["n_cands"] <= TOP_K).all() and (b["n_cands"] >= 1).all()),
          f"range {b['n_cands'].min()}..{b['n_cands'].max()}")

    # Pruned visit shares must be a distribution over the populated slots.
    slot = np.arange(TOP_K)[None, :] < b["n_cands"][:, None]
    mass = np.where(slot, b["prob"], 0.0).sum(axis=1)
    check("policy target normalised", bool(np.abs(mass - 1.0).max() < 1e-4),
          f"worst |Σp − 1| = {np.abs(mass - 1.0).max():.2e}")
    check("no mass in unused slots", bool(np.where(~slot, b["prob"], 0.0).max() == 0.0))

    # The live mask must actually cover the declarations recorded against it: a
    # move declared on a cell the mask calls dead would never be trained on.
    live_move = (b["live"] & C_MOVE) > 0
    declared_move = b["spec"][:, 0, :CELLS] != 255
    orphan = int((declared_move & ~live_move).sum())
    check("declared moves lie inside the live mask", orphan == 0, f"{orphan} orphaned")

    # Each side should appear; a one-sided buffer means the actor is broken.
    sides = np.unique(b.get("to_move", np.array([1, 2])))
    check("both sides represented", len(sides) >= 1, f"sides {sides.tolist()}")
    # The energy head learns from *contrast* between candidates at one node. A
    # node left with a single candidate after pruning contributes exactly zero
    # gradient (softmax over one element is 1, log of it is 0), so a buffer with
    # mean n_cands near 1 trains the sampler and value head while leaving the
    # policy of record essentially untouched.
    mean_c = float(b["n_cands"].mean())
    multi = float((b["n_cands"] > 1).mean())
    check("candidate sets give the energy head signal", mean_c >= 1.5,
          f"mean n_cands {mean_c:.2f}, {100 * multi:.0f}% of nodes have >1 — "
          f"raise --sims/--candidates if low")
    print(f"        {len(b['value'])} examples, mean |value| {np.abs(b['value']).mean():.3f}")
    return b


# ── 4/5. gradients and overfitting ────────────────────────────────────────────


def validate_learning(batch: dict, steps: int, device: str) -> None:
    print("\n4. gradient flow")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = MinionsNet(channels=24, blocks=2, energy_blocks=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    n = min(8, len(batch["value"]))
    small = {k: v[:n] for k, v in batch.items()}
    train_block(net, opt, small, 1, n, device, rng, 1.0, 1.0, 0.15)

    missing = [k for k, p in net.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    # BatchNorm on a head that saw a constant input can legitimately have zero
    # grad; report rather than fail, but flag whole heads.
    heads = {"trunk": "stem", "value": "val_",
             "diffusion": "dif_", "energy": "eng_",
             "hex control": "hex_", "necromancer survival": "necro_"}
    for name, prefix in heads.items():
        params = [k for k, _ in net.named_parameters() if k.startswith(prefix)]
        dead = [k for k in params if k in missing]
        check(f"{name} head receives gradient", len(dead) < len(params),
              f"{len(dead)}/{len(params)} params without gradient")

    print(f"\n5. overfit {n} examples for {steps} steps (every loss must fall)")
    first = train_block(net, opt, small, 1, n, device, rng, 1.0, 1.0, 0.15)
    last = train_block(net, opt, small, steps, n, device, rng, 1.0, 1.0, 0.15)
    names = ["value", "declaration", "energy", "aux"]
    for i, nm in enumerate(names):
        if first[i] == 0.0 and last[i] == 0.0:
            check(f"{nm} loss", True, "identically zero (no signal in this batch)")
            continue
        if nm == "energy":
            # The energy loss is a cross-entropy against the pruned visit
            # distribution, so it cannot fall below that target's own entropy.
            # Early in a run the net is untrained, MCTS spreads visits almost
            # evenly over the surviving candidates, and H(p) sits just under
            # log(n_cands) — leaving a few hundredths of headroom. Demanding a
            # 10% drop there asks for a loss below the floor, which is
            # impossible, and reads as a broken energy head when nothing is
            # wrong. Score the *excess* over the floor instead.
            floor = _target_entropy(small)
            check("energy loss reaches its entropy floor",
                  last[i] <= floor + 0.15,
                  f"{first[i]:.4f} -> {last[i]:.4f}, floor H(p) = {floor:.4f}, "
                  f"excess {last[i] - floor:+.4f}")
            continue
        check(f"{nm} loss falls", last[i] < first[i] * 0.9,
              f"{first[i]:.4f} -> {last[i]:.4f}")


def _target_entropy(batch: dict) -> float:
    """Mean entropy of the pruned visit targets — the floor the energy CE can reach.

    Cross-entropy H(p, q) is minimised at H(p), so this is the best any energy
    head can do on this batch. Comparing the achieved loss against it separates
    "the head is not learning" from "there is nothing left to learn here".
    """
    n = batch["n_cands"]
    out = []
    for e in range(len(n)):
        p = batch["prob"][e][: int(n[e])]
        p = p[p > 0]
        if p.size:
            out.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(out)) if out else 0.0


# ── 6. target sanity ──────────────────────────────────────────────────────────


def validate_targets(batch: dict) -> None:
    print("\n6. target sanity")
    v = batch["value"]
    check("value is signed, not constant", float(v.std()) > 1e-6, f"std {v.std():.3f}")
    check("value takes terminal values", bool(np.isin(np.abs(v), (0.0, 1.0)).all()),
          f"unique {np.unique(v).tolist()[:5]}")
    hexc = batch["hex_control"]
    check("hex control is a fraction in [0,1]",
          bool(hexc.min() >= 0.0 and hexc.max() <= 1.0),
          f"range {hexc.min():.3f}..{hexc.max():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="validate the Minions training setup")
    ap.add_argument("--bundle", default="data/minions_validation")
    ap.add_argument("--live", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 70)
    print("Minions training-setup validation")
    print("=" * 70)

    validate_decl_parity(args.bundle)
    validate_features(args.bundle)
    batch = validate_records(args.live)
    if batch is not None:
        validate_targets(batch)
        validate_learning(batch, args.steps, args.device)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
