"""Train a best-move head on a frozen checkpoint, from the replay buffer.

The head replaces the diffusion sampler (``MinionsNet(proposer="best")``): one
forward pass predicts the search's favourite turn -- the most-visited child of
every full-search root in the buffer -- as a regression, squared error on every
class of every live declaration variable against that turn's one-hot. Dead
cells declare nothing, so they carry no loss; the search never reads them.

Everything else -- trunk, value, energy, aux, score -- is copied from ``--init``
and frozen, BatchNorm statistics included, so a match against ``--init``
isolates the proposer: both seats evaluate and score candidates identically and
differ only in where the candidates come from.

    python/minions_best_head.py --init data/models/minions/live/snapshots/gen343 \\
        --buffer data/models/minions/live/buffer --out data/models/minions/best-head

The last ``--val-chunks`` chunks (the newest) are held out. The head kept is the
one with the lowest held-out squared error across the periodic evaluations.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minions_az_learner import C_MOVE, C_SPAWN, C_TARGET, read_chunk  # noqa: E402
from playground.model_server.minions_net import (  # noqa: E402
    CELLS, IN_PLANES, N, N_OFFSETS, N_SPAWNS, N_TARGETS, arch_of, build_model,
    export_onnx_atomic,
)

# (spec field, classes, live bit, offset into the 46 outputs per cell)
FIELDS = (
    (0, N_OFFSETS, C_MOVE, 0),
    (1, N_TARGETS, C_TARGET, N_OFFSETS),
    (2, N_SPAWNS, C_SPAWN, N_OFFSETS + N_TARGETS),
)
FIELD_NAMES = ("move", "target", "spawn")


def load_rows(paths):
    """Full-search rows that have a most-visited child: (planes, live, its spec).

    Children are stored with their pruned visit share; the target is the
    largest. A row whose shares are all zero (every visit forced) has no
    favourite and is skipped, as are fast rows, which record no children.
    """
    planes, live, spec = [], [], []
    for p in paths:
        d = read_chunk(p)
        if d is None:
            continue
        n = d["n_cands"].astype(np.int64)
        if n.sum() == 0:
            continue
        # Children are stored strongest first and back to back, so a row's
        # first child is its favourite.
        first = np.cumsum(n) - n
        keep = np.flatnonzero((d["is_full"] > 0) & (n > 0))
        keep = keep[d["cand_prob"][first[keep]] > 0]
        planes.append(d["planes"][keep])
        live.append(d["live"][keep])
        spec.append(d["cand_spec"][first[keep]])
    return np.concatenate(planes), np.concatenate(live), np.concatenate(spec)


def targets(spec, live):
    """One-hot targets and the live-variable mask, both ``(B, CELLS, 46)``."""
    tgt, mask = [], []
    for field, width, bit, _ in FIELDS:
        v = spec[:, field * CELLS:(field + 1) * CELLS].long().clamp(0, width - 1)
        tgt.append(F.one_hot(v, width).float())
        on = ((live & bit) > 0).float().unsqueeze(-1)
        mask.append(on.expand(-1, -1, width))
    return torch.cat(tgt, dim=2), torch.cat(mask, dim=2)


def forward(net, planes, live, spec):
    b = len(live)
    with torch.no_grad():
        hh = net.head_features(net.trunk(planes.float().view(b, IN_PLANES, N, N)))
    tgt, mask = targets(spec, live)
    return net.best_head(hh), tgt, mask


def set_head_mode(net, train: bool):
    """Train mode for the head only: the frozen trunk keeps its BN statistics."""
    for name, mod in net.named_children():
        if name.startswith("best_"):
            mod.train(train)


@torch.no_grad()
def evaluate(net, data, bs=2048):
    """Held-out squared error, per-field argmax accuracy on live variables, and
    whole-turn exact match. Argmax here is over *all* classes; the engine
    decodes over legal ones only, so its accuracy can only be higher.
    ``base_*`` is the share of live variables whose answer is class 0 (stay,
    no attack, spawn nothing) -- what a head that never acts would score."""
    planes, live, spec = data
    sq = cnt = 0.0
    hit, tot, base = np.zeros(3), np.zeros(3), np.zeros(3)
    exact = 0
    for i in range(0, len(live), bs):
        lv, sp = live[i:i + bs], spec[i:i + bs]
        out, tgt, mask = forward(net, planes[i:i + bs], lv, sp)
        sq += float(((out - tgt) ** 2 * mask).sum())
        cnt += float(mask.sum())
        whole = torch.ones(len(lv), dtype=torch.bool, device=out.device)
        for j, (field, width, bit, off) in enumerate(FIELDS):
            on = (lv & bit) > 0
            want = sp[:, field * CELLS:(field + 1) * CELLS].long().clamp(0, width - 1)
            got = out[:, :, off:off + width].argmax(dim=2)
            hit[j] += float(((got == want) & on).sum())
            tot[j] += float(on.sum())
            base[j] += float(((want == 0) & on).sum())
            whole &= ((got == want) | ~on).all(dim=1)
        exact += int(whole.sum())
    m = {"mse": sq / max(cnt, 1.0), "turn_exact": exact / max(len(live), 1)}
    for j, name in enumerate(FIELD_NAMES):
        m[f"acc_{name}"] = hit[j] / max(tot[j], 1.0)
        m[f"base_{name}"] = base[j] / max(tot[j], 1.0)
    return m


def main():
    ap = argparse.ArgumentParser(description="best-move head on a frozen Minions net")
    ap.add_argument("--init", required=True, help="snapshot dir: checkpoint.safetensors + config.json")
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-chunks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    cfg = json.load(open(os.path.join(args.init, "config.json")))
    arch = {**cfg["arch"], "proposer": "best"}
    net = build_model(arch).to(dev)
    missing, unexpected = net.load_state_dict(
        load_file(os.path.join(args.init, "checkpoint.safetensors")), strict=False
    )
    bad = [k for k in unexpected if not k.startswith("dif_")]
    bad += [k for k in missing if not k.startswith("best_")]
    if bad:
        raise RuntimeError(f"checkpoint and architecture disagree beyond the proposer: {bad[:8]}")
    print(f"from {args.init}: dropped {len(unexpected)} diffusion tensors, "
          f"{len(missing)} best-head tensors to train", flush=True)
    head = []
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("best_")
        if p.requires_grad:
            head.append(p)
    print(f"trainable: {sum(p.numel() for p in head):,} of "
          f"{sum(p.numel() for p in net.parameters()):,} parameters", flush=True)

    paths = sorted(glob.glob(os.path.join(args.buffer, "chunk_*.bin")))
    train_paths, val_paths = paths[:-args.val_chunks], paths[-args.val_chunks:]
    t0 = time.time()
    to_dev = lambda rows: tuple(torch.from_numpy(a).to(dev) for a in rows)  # noqa: E731
    train = to_dev(load_rows(train_paths))
    val = to_dev(load_rows(val_paths))
    print(f"{len(train[1]):,} training rows from {len(train_paths)} chunks, "
          f"{len(val[1]):,} held out from {len(val_paths)} ({time.time() - t0:.0f}s)", flush=True)

    opt = torch.optim.Adam(head, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )
    net.eval()
    set_head_mode(net, False)
    m0 = evaluate(net, val)
    print(json.dumps({"step": 0, **{k: round(v, 4) for k, v in m0.items()}}), flush=True)

    n = len(train[1])
    log, best_m, best_state = [], None, None
    running, t0 = 0.0, time.time()
    set_head_mode(net, True)
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, n, size=args.batch)).to(dev)
        out, tgt, mask = forward(net, train[0][idx], train[1][idx], train[2][idx])
        loss = ((out - tgt) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss.detach())
        if step % args.eval_every == 0 or step == args.steps:
            set_head_mode(net, False)
            m = evaluate(net, val)
            set_head_mode(net, True)
            m = {"step": step, "train_mse": running / args.eval_every, **m,
                 "seconds": round(time.time() - t0, 1)}
            running = 0.0
            log.append(m)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in m.items()}), flush=True)
            if best_m is None or m["mse"] < best_m["mse"]:
                best_m = m
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()
                              if k.startswith("best_")}

    net.load_state_dict(best_state, strict=False)
    net.eval()
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
              os.path.join(args.out, "checkpoint.safetensors"))
    json.dump({"arch": arch_of(net), "generation": int(cfg.get("generation", 0)),
               "init": os.path.abspath(args.init)},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump({"kept": best_m, "untrained": m0, "log": log,
               "train_rows": len(train[1]), "val_rows": len(val[1])},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    export_onnx_atomic(net, args.out)
    print(f"kept step {best_m['step']} (held-out mse {best_m['mse']:.4f}); wrote {args.out}")


if __name__ == "__main__":
    main()
