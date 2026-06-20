"""Trainer for the Connect 4 reification-based latent-space net (docs/algorithm.md).

Reads v2 self-play dumps from `connect4-engine lgs-gen` and fits the LGS net:

  L = value MSE + lpi·MDN-policy-NLL + lrho·reifier-CE + lc·consistency + reg

All targets live in the *current* encoder's latent space: parents and reified
children are re-encoded each step, so the MDN policy target (the search's visit
distribution over next-LGS) and the reifier target track the moving encoder.
Writes checkpoint.safetensors + config.json (arch) that the candle self-play
net (crates/games/connect4/src/lgs_net.rs) loads directly.

Usage:
  # seed a random checkpoint to bootstrap self-play
  python c4_lgs_train.py --init data/models/connect4_lgs/live --channels 32 --blocks 3
  # train one or more dumps -> new checkpoint
  python c4_lgs_train.py --buffer run/buf --in live/checkpoint.safetensors \
      --out live/checkpoint.safetensors --epochs 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server import c4_lgs_net as M

PLANES = 2 * 6 * 7  # 84
MAX_CHILDREN = 7
REC_F32 = PLANES + 1 + MAX_CHILDREN * (2 + PLANES)  # 687, must match lgs_selfplay.rs


def read_dump(path: str):
    """Read a v2 dump into arrays.

    Returns (parent (N,2,6,7), z (N,), cols (N,7) int, visits (N,7),
             child (N,7,2,6,7))."""
    raw = np.fromfile(path, dtype=np.uint8)
    count = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    flat = np.frombuffer(raw[8 : 8 + count * REC_F32 * 4], dtype="<f4").reshape(count, REC_F32)
    parent = flat[:, :PLANES].reshape(count, 2, 6, 7)
    z = flat[:, PLANES].copy()
    block = flat[:, PLANES + 1 :].reshape(count, MAX_CHILDREN, 2 + PLANES)
    cols = block[:, :, 0].astype(np.int64)
    visits = block[:, :, 1].astype(np.float32)
    child = block[:, :, 2:].reshape(count, MAX_CHILDREN, 2, 6, 7)
    return parent, z, cols, visits, child


def load_buffer(path: str):
    files = (
        sorted(glob.glob(os.path.join(path, "*.bin")))
        if os.path.isdir(path)
        else [path]
    )
    if not files:
        raise SystemExit(f"no dump files under {path!r}")
    parts = [read_dump(f) for f in files]
    return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(5))


def do_init(args) -> None:
    os.makedirs(args.init, exist_ok=True)
    net = M.build_lgs_model({"channels": args.channels, "blocks": args.blocks, "k": args.k})
    M.save_checkpoint(net, os.path.join(args.init, "checkpoint.safetensors"))
    json.dump({"arch": M.arch_of(net)}, open(os.path.join(args.init, "config.json"), "w"))
    print(f"init: random LGS net -> {args.init} (arch {M.arch_of(net)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="write a random checkpoint to this dir and exit")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--k", type=int, default=M.K_COMPONENTS)
    ap.add_argument("--buffer", help="dump file or dir of *.bin")
    ap.add_argument("--in", dest="inp", help="checkpoint to resume from")
    ap.add_argument("--out", help="checkpoint to write")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lpi", type=float, default=1.0, help="MDN policy weight")
    ap.add_argument("--lrho", type=float, default=1.0, help="reifier weight")
    # Consistency defaults OFF: the exact-match form (α·d_lat ≈ d_func) collapses
    # the latent while the value head is still ≈constant (d_func≈0 ⇒ d_lat→0),
    # which kills value/reifier learning (docs/algorithm.md "central tension").
    # Re-enable later as a ramped anti-collapse (Lipschitz) term.
    ap.add_argument("--lc", type=float, default=0.0, help="consistency weight (0 = off)")
    ap.add_argument("--ent", type=float, default=1e-3, help="mixture-entropy bonus")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.init:
        do_init(args)
        return
    if not args.buffer or not args.out:
        raise SystemExit("training needs --buffer and --out (or use --init)")

    dev = torch.device(args.device)
    if args.inp and os.path.exists(args.inp):
        cfg = json.load(open(os.path.join(os.path.dirname(args.inp), "config.json")))["arch"]
        net = M.build_lgs_model(cfg)
        M.load_checkpoint(net, args.inp)
    else:
        net = M.build_lgs_model({"channels": args.channels, "blocks": args.blocks, "k": args.k})
    net.to(dev).train()

    alpha = torch.nn.Parameter(torch.tensor(1.0, device=dev))  # consistency metric scale
    opt = torch.optim.Adam(list(net.parameters()) + [alpha], lr=args.lr, weight_decay=args.wd)

    parent, z, cols, visits, child = load_buffer(args.buffer)
    n = parent.shape[0]
    parent_t = torch.from_numpy(parent).to(dev)
    z_t = torch.from_numpy(z).to(dev)
    cols_t = torch.from_numpy(cols).to(dev)
    visits_t = torch.from_numpy(visits).to(dev)
    child_t = torch.from_numpy(child).to(dev)
    print(f"train: {n} examples, {args.epochs} epochs, batch {args.batch}, device {dev}")

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = {"v": 0.0, "pi": 0.0, "rho": 0.0, "c": 0.0, "n": 0}
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            bp, bz, bcol, bvis, bch = parent_t[b], z_t[b], cols_t[b], visits_t[b], child_t[b]
            B = bp.shape[0]

            e_parent = net.encode(bp)  # (B,d)
            # value
            lv = F.mse_loss(net.value(e_parent), bz)

            # children encodings + visit-weighted normalization
            e_child = net.encode(bch.reshape(B * MAX_CHILDREN, 2, 6, 7)).reshape(B, MAX_CHILDREN, -1)
            wsum = bvis.sum(dim=1, keepdim=True).clamp_min(1.0)
            pi = bvis / wsum  # (B,7), zero for absent children

            w, mu, std = net.policy_mdn(e_parent)
            lpi = torch.zeros((), device=dev)
            lrho = torch.zeros((), device=dev)
            for c in range(MAX_CHILDREN):
                m = pi[:, c]  # (B,)
                # Per-dimension NLL: a d-dim Gaussian log-density is ~O(d), which
                # otherwise swamps the value/reifier gradients on the shared
                # encoder. Normalizing by d puts all loss terms on an O(1) scale.
                logp = M.mdn_log_prob(e_child[:, c], w, mu, std) / M.LGS_DIM  # (B,)
                lpi = lpi - (m * logp).sum()
                rlog = net.reify_logits(e_parent, e_child[:, c])  # (B,7)
                ce = F.cross_entropy(rlog, bcol[:, c], reduction="none")
                lrho = lrho + (m * ce).sum()
            lpi = lpi / B
            lrho = lrho / B

            # consistency: random within-batch pairs, value-only d_func (start)
            j = torch.randperm(B, device=dev)
            d_lat = (e_parent - e_parent[j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            d_func = (net.value(e_parent) - net.value(e_parent)[j]).abs()
            lc = (alpha * d_lat - d_func).pow(2).mean()

            # mixture-entropy bonus (encourage spread, discourage collapse)
            ent = -(w.clamp_min(1e-9).log() * w).sum(-1).mean()

            loss = lv + args.lpi * lpi + args.lrho * lrho + args.lc * lc - args.ent * ent
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot["v"] += float(lv.detach()) * B
            tot["pi"] += float(lpi.detach()) * B
            tot["rho"] += float(lrho.detach()) * B
            tot["c"] += float(lc.detach()) * B
            tot["n"] += B
        d = tot["n"]
        print(
            f"epoch {epoch}: v {tot['v']/d:.4f}  pi {tot['pi']/d:.4f}  "
            f"rho {tot['rho']/d:.4f}  c {tot['c']/d:.4f}  alpha {float(alpha):.3f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    net.cpu().eval()
    M.save_checkpoint(net, args.out)
    json.dump({"arch": M.arch_of(net)}, open(os.path.join(os.path.dirname(args.out) or ".", "config.json"), "w"))
    print(f"wrote {args.out} (arch {M.arch_of(net)})")


if __name__ == "__main__":
    main()
