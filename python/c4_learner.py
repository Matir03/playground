"""Continuous Connect 4 learner (AlphaZero-style), the async counterpart to the
Go9 learner. Trains nonstop from the replay buffer the actor fills, publishing
checkpoints the actor hot-swaps to. Connect 4 has no reifier (reification is a
structural column decode), so the loss is just value MSE on the game-outcome z +
policy CE on the PUCT visit distribution.

Throttle: train on at most --max-reuse x the actor's produced samples. Pacing:
stay at most --eval-lead snapshots ahead of the evaluator (reads
processed_through from live/scores.json).
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

from playground.model_server.net import (
    ConnectFourResNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)

PLANES, ACTIONS = 2 * 6 * 7, 7
RECORD_F32 = PLANES + ACTIONS + 1  # 92: planes(84) + policy(7) + value(1)


def read_examples(path: str):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        a = np.frombuffer(f.read(), dtype="<f4").reshape(n, RECORD_F32)
    planes = a[:, :PLANES].reshape(n, 2, 6, 7).copy()
    policy = a[:, PLANES:PLANES + ACTIONS].copy()
    value = a[:, PLANES + ACTIONS].copy()
    return planes, policy, value


def save_atomic(net, path: str) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().to("cpu") for k, v in net.state_dict().items()}
    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/connect4/live")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps-per-snapshot", type=int, default=400)
    ap.add_argument("--max-reuse", type=float, default=4.0)
    ap.add_argument("--eval-lead", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=4000)
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
    scores_path = os.path.join(live, "scores.json")
    os.makedirs(snaps, exist_ok=True)

    net = ConnectFourResNet(channels=args.channels, blocks=args.blocks)
    if os.path.exists(latest):
        load_checkpoint(net, latest)
        print(f"seeded from {latest}", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    anchor_gen = int(json.load(open(os.path.join(live, "config.json"))).get("generation", 0))

    cache: dict[str, tuple] = {}
    big: tuple | None = None
    produced = 0
    consumed = 0

    def refresh() -> bool:
        nonlocal produced
        files = set(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))
        changed = False
        for f in files - cache.keys():
            try:
                ex = read_examples(f)
            except Exception:
                continue
            cache[f] = ex
            produced += ex[0].shape[0]
            changed = True
        for f in list(cache.keys() - files):
            del cache[f]
            changed = True
        return changed

    def processed_through() -> int:
        try:
            return int(json.load(open(scores_path)).get("processed_through", anchor_gen))
        except Exception:
            return anchor_gen

    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    snap_gen = max(existing + [anchor_gen])
    refresh()
    print(f"c4 learner: device {device} batch {args.batch_size} throttle {args.max_reuse}x "
          f"eval-lead {args.eval_lead} snapshot/{args.steps_per_snapshot} continuing after gen {snap_gen}",
          flush=True)

    BLOCK = 100
    steps_into_snap = 0
    lv = lp = 0.0
    nb = 0
    while True:
        if refresh() or big is None:
            if cache:
                cols = list(zip(*cache.values()))
                big = tuple(np.concatenate(c) for c in cols)
        n = big[0].shape[0] if big else 0
        if n < args.min_samples:
            time.sleep(2.0)
            continue

        if steps_into_snap == 0:
            need = (snap_gen + 1) - args.eval_lead
            if need > anchor_gen and processed_through() < need:
                time.sleep(2.0)
                continue

        budget = int(args.max_reuse * produced) - consumed
        if budget < args.batch_size:
            time.sleep(2.0)
            continue

        planes, policy, value = big
        block = min(BLOCK, args.steps_per_snapshot - steps_into_snap, budget // args.batch_size)
        net.train()
        for _ in range(block):
            b = rng.integers(0, n, size=args.batch_size)
            pl = torch.from_numpy(planes[b]).to(device)
            tgt_pi = torch.from_numpy(policy[b]).to(device)
            tgt_z = torch.from_numpy(value[b]).to(device)
            v_pred, logits = net(pl)
            loss_v = F.mse_loss(v_pred, tgt_z)
            loss_p = -(tgt_pi * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss = loss_v + loss_p
            opt.zero_grad()
            loss.backward()
            opt.step()
            consumed += args.batch_size
            lv += float(loss_v); lp += float(loss_p); nb += 1
        steps_into_snap += block
        net.eval()
        save_atomic(net, latest)

        if steps_into_snap >= args.steps_per_snapshot:
            snap_gen += 1
            d = os.path.join(snaps, f"gen{snap_gen:03d}")
            os.makedirs(d, exist_ok=True)
            save_checkpoint(net.to("cpu"), os.path.join(d, "checkpoint.safetensors"))
            net.to(device)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"arch": arch_of(net), "generation": snap_gen}, f)
            print(f"snapshot gen{snap_gen}  samples {n}  produced {produced} consumed {consumed} "
                  f"({consumed/max(1,produced):.1f}x)  loss v {lv/max(1,nb):.4f} p {lp/max(1,nb):.4f}",
                  flush=True)
            steps_into_snap = 0
            lv = lp = 0.0
            nb = 0


if __name__ == "__main__":
    main()
