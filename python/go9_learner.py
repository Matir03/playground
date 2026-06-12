"""Continuous Go9 learner (AlphaZero-style): trains nonstop from the replay
buffer the actors fill, publishing checkpoints the actors hot-swap to.

Loop: ingest any new `live/buffer/chunk_*.bin` into an in-memory replay set
(the actor prunes the on-disk window, so dropped chunks are evicted here too),
run a round of SGD (value MSE on the game-outcome z, policy CE over the 82
actions, reifier consistency on the played action), then atomically publish
`live/latest.safetensors` (actors pick it up next chunk) and periodically
snapshot `live/snapshots/genNNN/` for the evaluator/ELO.

Runs concurrently with the actors and evaluator on the one GPU (they contend).
Source /tmp/cuda_env.sh first is not needed for torch, but harmless.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.go9_net import (
    GS_DIM,
    Go9ResNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)
from rust_go9_selfplay_train import ACTIONS, read_examples  # dump reader + layout


def save_atomic(net: Go9ResNet, path: str) -> None:
    """Write a CPU copy of the weights and atomically rename into place."""
    sd = {k: v.detach().to("cpu") for k, v in net.state_dict().items()}
    from safetensors.torch import save_file

    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/go9/live")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-round", type=int, default=200)
    ap.add_argument("--snapshot-every", type=int, default=5, help="rounds per snapshot")
    ap.add_argument("--min-samples", type=int, default=4000)
    ap.add_argument("--reif-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    live = args.live
    buffer_dir = os.path.join(live, "buffer")
    latest = os.path.join(live, "latest.safetensors")
    snaps = os.path.join(live, "snapshots")
    os.makedirs(snaps, exist_ok=True)

    # Seed from the published latest (the gen-12 net we copied in).
    net = Go9ResNet(channels=args.channels, blocks=args.blocks)
    if os.path.exists(latest):
        load_checkpoint(net, latest)
        print(f"seeded from {latest}", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    cfg = json.load(open(os.path.join(live, "config.json")))
    snap_gen = int(cfg.get("generation", 0))
    print(f"learner: device {device}  batch {args.batch_size}  steps/round {args.steps_per_round}"
          f"  start gen {snap_gen}", flush=True)

    cache: dict[str, tuple] = {}  # chunk path -> arrays
    big: tuple | None = None      # concatenated arrays for sampling
    round_i = 0
    reif_w = args.reif_weight

    def refresh() -> bool:
        files = set(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))
        changed = False
        for f in files - cache.keys():
            try:
                cache[f] = read_examples(f)
                changed = True
            except Exception:
                pass  # mid-rename or short read; pick it up next refresh
        for f in list(cache.keys() - files):
            del cache[f]
            changed = True
        return changed

    while True:
        if refresh() or big is None:
            if cache:
                cols = list(zip(*cache.values()))
                big = tuple(np.concatenate(c) for c in cols)
        n = big[0].shape[0] if big else 0
        if n < args.min_samples:
            time.sleep(2.0)
            continue

        planes, policy, value, played, has_next, next_planes = big
        net.train()
        t0 = time.time()
        ev = ep = er = 0.0
        for _ in range(args.steps_per_round):
            b = rng.integers(0, n, size=args.batch_size)
            pl = torch.from_numpy(planes[b]).to(device)
            tgt_pi = torch.from_numpy(policy[b]).to(device)
            tgt_z = torch.from_numpy(value[b]).to(device)
            gs = net.encode_batch(pl)
            v_pred = net.value_batch(gs)
            logits = net.pol_prior(gs)
            loss_v = F.mse_loss(v_pred, tgt_z)
            loss_p = -(tgt_pi * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss = loss_v + loss_p
            loss_r = torch.zeros((), device=device)
            if reif_w > 0:
                npl = torch.from_numpy(next_planes[b]).to(device)
                hn = torch.from_numpy(has_next[b]).to(device)
                pa = torch.from_numpy(played[b]).to(device)
                pred_next = torch.tanh(net.pol_next(gs)).view(-1, ACTIONS, GS_DIM)
                pred_a = pred_next[torch.arange(pred_next.shape[0], device=device), pa]
                with torch.no_grad():
                    tgt_next = net.encode_batch(npl)
                per = ((pred_a - tgt_next) ** 2).mean(1) * hn
                loss_r = per.sum() / hn.sum().clamp(min=1.0)
                loss = loss + reif_w * loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
            ev += float(loss_v); ep += float(loss_p); er += float(loss_r)

        spr = args.steps_per_round
        net.eval()
        save_atomic(net, latest)  # publish for actor hot-swap
        round_i += 1
        snap = ""
        if round_i % args.snapshot_every == 0:
            snap_gen += 1
            d = os.path.join(snaps, f"gen{snap_gen:03d}")
            os.makedirs(d, exist_ok=True)
            save_checkpoint(net.to("cpu"), os.path.join(d, "checkpoint.safetensors"))
            net.to(device)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"arch": arch_of(net), "generation": snap_gen}, f)
            snap = f"  -> snapshot gen{snap_gen}"

        steps_s = spr / (time.time() - t0)
        print(f"round {round_i}  samples {n}  chunks {len(cache)}  "
              f"loss v {ev/spr:.4f} p {ep/spr:.4f} r {er/spr:.4f}  "
              f"{steps_s:.0f} steps/s{snap}", flush=True)


if __name__ == "__main__":
    main()
