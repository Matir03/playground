"""Widen a checkpoint's stem for newly appended input planes, zero-initialised.

Adding a feature plane changes `stem.weight` from ``[C, old, 3, 3]`` to
``[C, new, 3, 3]``. Loading the old checkpoint into the new net would fail on the
shape; retraining from scratch would throw away the run. Instead, append zero
weights for the new channels: the net computes *exactly* what it did before, so
no training progress is lost, and gradients start flowing into the new plane on
the next training step.

    python python/minions_migrate_planes.py --live data/models/minions/live
"""

import argparse
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from playground.model_server.minions_net import IN_PLANES  # noqa: E402

STEM = "stem.weight"


def migrate(path: str, want: int) -> str:
    if not os.path.exists(path):
        return f"{os.path.basename(path)}: absent, skipped"
    t = load_file(path)
    if STEM not in t:
        return f"{os.path.basename(path)}: no {STEM}, skipped"
    w = t[STEM]
    have = w.shape[1]
    if have == want:
        return f"{os.path.basename(path)}: already {want} planes"
    if have > want:
        return f"{os.path.basename(path)}: has {have} > {want}, refusing to truncate"
    pad = torch.zeros(w.shape[0], want - have, *w.shape[2:], dtype=w.dtype)
    t[STEM] = torch.cat([w, pad], dim=1)
    save_file(t, path)
    return f"{os.path.basename(path)}: {have} -> {want} planes ({want - have} zero-initialised)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", required=True)
    ap.add_argument("--planes", type=int, default=IN_PLANES)
    args = ap.parse_args()

    targets = [
        os.path.join(args.live, "latest.safetensors"),
        os.path.join(args.live, "best.safetensors"),
    ]
    snaps = os.path.join(args.live, "snapshots")
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            targets.append(os.path.join(snaps, d, "checkpoint.safetensors"))
    for p in targets:
        print(migrate(p, args.planes))
    # The optimiser state carries a moment per parameter, so it is stale for the
    # widened tensor. Dropping it costs a little momentum for one generation and
    # is far safer than pairing new weights with old moments.
    opt = os.path.join(args.live, "opt_state.pt")
    if os.path.exists(opt):
        os.remove(opt)
        print("opt_state.pt: removed (moments no longer match the widened stem)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
