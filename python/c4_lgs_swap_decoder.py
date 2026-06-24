"""Swap the reifier for a position decoder on a trained Connect 4 LGS checkpoint.

Replaces the reifier head ((LGS, next-LGS) -> 7 column logits) with a position
decoder (next-LGS -> board planes). The search then decodes a sampled next-LGS to
a target board and infers the move from it (the single newly-occupied cell's
column); if that isn't a legal single-drop successor it falls back to a random
legal move (see crates/.../lgs_net.rs::move_from_decoded).

What it does (encoder/policy/value kept byte-identical, weights and all):
  1. load the source (reifier) checkpoint + arch,
  2. build a new net with decoder=True, copy every non-edge weight across, leave
     the decoder freshly initialised (the reifier weights are dropped),
  3. freeze everything except the decoder and train it (reconstruction BCE of
     decode(E(S)) vs board planes) on the existing buffer for a couple of epochs
     — "train the decoder against the frozen encoder before cotraining",
  4. write the new checkpoint + config.json (arch now carries decoder=true).

Resume the normal cycle afterwards: c4_lgs_train.py detects the decoder head and
cotrains it (BCE reconstruction) in place of the reifier CE.

Usage:
  python c4_lgs_swap_decoder.py \
      --in  data/models/connect4_lgs/live/checkpoint.safetensors \
      --out data/models/connect4_lgs/live/checkpoint.safetensors \
      --buffer data/models/connect4_lgs/live/buffer --epochs 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from playground.model_server import c4_lgs_net as M
from c4_lgs_train import load_buffer, MAX_CHILDREN, alias_decode_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source (reifier) checkpoint")
    ap.add_argument("--out", required=True, help="checkpoint to write (may equal --in)")
    ap.add_argument("--buffer", required=True, help="dump file or dir of *.bin")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)

    src_arch = json.load(open(os.path.join(os.path.dirname(args.inp), "config.json")))["arch"]
    src = M.build_lgs_model(src_arch)
    M.load_checkpoint(src, args.inp)

    new_arch = dict(src_arch)
    new_arch["decoder"] = True
    net = M.build_lgs_model(new_arch)

    # Copy every weight present (and shape-matching) in both nets: the whole net
    # except the edge head (reifier dropped; decoder stays fresh).
    src_sd, new_sd = src.state_dict(), net.state_dict()
    shared = {k: v for k, v in src_sd.items() if k in new_sd and new_sd[k].shape == v.shape}
    new_sd_keys = set(new_sd)
    net.load_state_dict(shared, strict=False)
    fresh = sorted(new_sd_keys - set(shared))
    print(f"copied {len(shared)} shared tensors; fresh decoder tensors: {fresh}")

    # Freeze everything except the decoder head.
    edge_params = set(net.edge_parameters())
    for p in net.parameters():
        p.requires_grad = p in edge_params
    print(f"decoder trainable params: {sum(p.numel() for p in edge_params)}; "
          f"encoder/policy/value frozen")

    # eval() so the frozen encoder's BatchNorm running stats don't drift.
    net.to(dev).eval()
    opt = torch.optim.Adam(net.edge_parameters(), lr=args.lr, weight_decay=args.wd)

    parent, z, cols, visits, child = load_buffer(args.buffer)
    n = parent.shape[0]
    parent_t = torch.from_numpy(parent).to(dev)
    visits_t = torch.from_numpy(visits).to(dev)
    child_t = torch.from_numpy(child).to(dev)
    print(f"swap-decoder: {n} examples, {args.epochs} epochs, batch {args.batch}, device {dev}")

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            bp, bvis, bch = parent_t[b], visits_t[b], child_t[b]
            B = bp.shape[0]
            with torch.no_grad():
                e_parent = net.encode(bp)  # frozen encoder
                e_child = net.encode(bch.reshape(B * MAX_CHILDREN, 2, 6, 7)).reshape(B, MAX_CHILDREN, -1)
                w, mu, std = net.policy_mdn(e_parent)  # frozen MDN (for aliased samples)
            wsum = bvis.sum(dim=1, keepdim=True).clamp_min(1.0)
            pi = bvis / wsum  # (B,7), zero for absent children
            nfull = (bvis.sum(dim=1) > 0).sum().clamp_min(1).float()

            loss = F.binary_cross_entropy_with_logits(net.decode(e_parent), bp.reshape(B, -1))
            crec = torch.zeros((), device=dev)
            for c in range(MAX_CHILDREN):
                rec = F.binary_cross_entropy_with_logits(
                    net.decode(e_child[:, c]), bch[:, c].reshape(B, -1), reduction="none"
                ).mean(dim=1)  # (B,)
                crec = crec + (pi[:, c] * rec).sum()
            loss = loss + crec / nfull
            # aliased intermediate points -> nearest child's board (see c4_lgs_train)
            loss = loss + alias_decode_loss(net, w, mu, std, e_child, bch, bvis, nfull)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * B
            cnt += B
        print(f"epoch {epoch}: decoder_bce {tot/cnt:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    net.cpu().eval()
    M.save_checkpoint(net, args.out)
    json.dump({"arch": M.arch_of(net)},
              open(os.path.join(os.path.dirname(args.out) or ".", "config.json"), "w"))
    print(f"wrote {args.out} (arch {M.arch_of(net)})")


if __name__ == "__main__":
    main()
