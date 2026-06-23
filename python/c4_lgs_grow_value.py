"""Grow *only* the value head of a trained Connect 4 LGS checkpoint.

The value head is a small MLP on top of the (shared) LGS latent. When the value
MSE plateaus while the encoder/policy/reifier keep improving, we can enlarge just
that head and retrain it on the existing self-play buffer — the encoder, MDN
policy and reifier stay *byte-identical* (same version), so search behaviour is
unchanged except for sharper value estimates.

What it does:
  1. load the source checkpoint + its arch (config.json beside it),
  2. build a new net with a bigger value head (--val-hidden, e.g. 256,256),
  3. copy every non-value weight across (trunk, enc_fc, MDN, reifier, aux),
     leaving the new value head freshly initialised,
  4. freeze everything except the value head and train it (value MSE only) on
     the buffer for a couple of epochs,
  5. write the new checkpoint + config.json (arch now carries `val_hidden`).

The candle self-play net (crates/games/connect4/src/lgs_net.rs) reads
`val_hidden` from config.json and loads the matching head, so the grown
checkpoint drops straight into the existing pipeline.

Usage:
  python c4_lgs_grow_value.py \
      --in  data/models/connect4_lgs/live/checkpoint.safetensors \
      --out data/models/connect4_lgs/live/checkpoint.safetensors \
      --buffer data/models/connect4_lgs/live/buffer \
      --val-hidden 256,256 --epochs 2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from playground.model_server import c4_lgs_net as M
from c4_lgs_train import load_buffer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source checkpoint")
    ap.add_argument("--out", required=True, help="checkpoint to write (may equal --in)")
    ap.add_argument("--buffer", required=True, help="dump file or dir of *.bin")
    ap.add_argument("--val-hidden", default="256,256",
                    help="comma-separated hidden widths of the new value MLP")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    new_hidden = [int(x) for x in args.val_hidden.split(",") if x.strip()]
    if not new_hidden:
        raise SystemExit("--val-hidden must list at least one width")

    # source net (its own arch) -> new net (same arch + bigger value head)
    src_arch = json.load(open(os.path.join(os.path.dirname(args.inp), "config.json")))["arch"]
    src = M.build_lgs_model(src_arch)
    M.load_checkpoint(src, args.inp)

    new_arch = dict(src_arch)
    new_arch["val_hidden"] = new_hidden
    net = M.build_lgs_model(new_arch)

    # Copy every weight that exists (and matches shape) in both nets: that is
    # the whole net except the value head, which differs and stays fresh.
    src_sd, new_sd = src.state_dict(), net.state_dict()
    shared = {k: v for k, v in src_sd.items() if k in new_sd and new_sd[k].shape == v.shape}
    missing = [k for k in new_sd if k not in shared]
    net.load_state_dict(shared, strict=False)
    print(f"copied {len(shared)} shared tensors; fresh value-head tensors: {missing}")

    # Freeze everything except the (new) value head.
    val_params = set(net.value_parameters())
    for p in net.parameters():
        p.requires_grad = p in val_params
    n_train = sum(p.numel() for p in val_params)
    print(f"value head {src_arch.get('val_hidden', [64])} -> {new_hidden} "
          f"({n_train} trainable params); encoder/policy/reifier frozen")

    # eval() so the frozen encoder's BatchNorm running stats don't drift; value
    # head has no BN/dropout, so its grads are unaffected.
    net.to(dev).eval()
    opt = torch.optim.Adam(net.value_parameters(), lr=args.lr, weight_decay=args.wd)

    parent, z, _cols, _visits, _child = load_buffer(args.buffer)
    n = parent.shape[0]
    parent_t = torch.from_numpy(parent).to(dev)
    z_t = torch.from_numpy(z).to(dev)
    print(f"grow-value: {n} examples, {args.epochs} epochs, batch {args.batch}, device {dev}")

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            bp, bz = parent_t[b], z_t[b]
            with torch.no_grad():
                e = net.encode(bp)  # frozen encoder
            lv = F.mse_loss(net.value(e), bz)
            opt.zero_grad()
            lv.backward()
            opt.step()
            tot += float(lv.detach()) * bp.shape[0]
            cnt += bp.shape[0]
        print(f"epoch {epoch}: value_mse {tot/cnt:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    net.cpu().eval()
    M.save_checkpoint(net, args.out)
    json.dump({"arch": M.arch_of(net)},
              open(os.path.join(os.path.dirname(args.out) or ".", "config.json"), "w"))
    print(f"wrote {args.out} (arch {M.arch_of(net)})")


if __name__ == "__main__":
    main()
