"""Continuous Go9 learner for the TRADITIONAL AlphaZero net (Go9AZNet) — the
benchmark baseline against the reification net's `go9_learner.py`.

Same async actor/learner setup, but the net is the vanilla AlphaZero architecture
(direct conv policy + value heads over the board, no GS latent, no reifier), so
the loss is just value MSE on the game-outcome z + policy CE over the 82 actions —
no reifier-consistency term. Trains from scratch: if `live/latest.safetensors`
doesn't exist, it writes a random-init gen-0 net (+config.json) so the actor has
something to hot-swap from.

Default live dir is separate (`data/models/go9_az/live`) so it never clobbers the
reification run. Pair with:
  actor: go9-engine selfplay-loop --live data/models/go9_az/live ... (arch is read
         from config.json, so the engine builds the go9-az net automatically)
  eval : go9_eval_loop.py --live data/models/go9_az/live (net-vs-net via eval-match)
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
    Go9AZNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)
from rust_go9_selfplay_train import read_examples  # dump reader + layout


def save_atomic(net: Go9AZNet, path: str) -> None:
    sd = {k: v.detach().to("cpu") for k, v in net.state_dict().items()}
    from safetensors.torch import save_file

    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)


def export_onnx_atomic(net: Go9AZNet, path: str) -> None:
    """Export value+policy to ONNX (dynamic batch) for the ORT actor to hot-swap to."""
    import torch

    was_training = net.training
    net.eval()
    dev = next(net.parameters()).device
    tmp = path + ".tmp"
    torch.onnx.export(
        net, torch.zeros(1, 3, 9, 9, device=dev), tmp,
        input_names=["planes"], output_names=["value", "policy"],
        dynamic_axes={"planes": {0: "b"}, "value": {0: "b"}, "policy": {0: "b"}},
        opset_version=17, dynamo=False,
    )
    os.replace(tmp, path)
    if was_training:
        net.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/go9_az/live")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-snapshot", type=int, default=300)
    ap.add_argument("--max-reuse", type=float, default=4.0,
                    help="cap samples trained <= max_reuse * samples the actor produced")
    ap.add_argument("--eval-lead", type=int, default=3,
                    help="stay at most this many snapshots ahead of the evaluator")
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
    cfg_path = os.path.join(live, "config.json")
    os.makedirs(snaps, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    net = Go9AZNet(channels=args.channels, blocks=args.blocks)
    if os.path.exists(latest):
        load_checkpoint(net, latest)
        print(f"seeded from {latest}", flush=True)
    else:
        # From scratch: publish a random gen-0 net + config so the actor can start.
        save_atomic(net, latest)
        with open(cfg_path, "w") as f:
            json.dump({"arch": arch_of(net), "generation": 0}, f, indent=2)
        print(f"from scratch: wrote random gen-0 {latest}", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # Publish ONNX so an ORT-backed actor has weights to start from.
    onnx_path = os.path.join(live, "model.onnx")
    export_onnx_atomic(net, onnx_path)

    anchor_gen = int(json.load(open(cfg_path)).get("generation", 0))
    elo_path = os.path.join(live, "elo.json")

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
            return int(json.load(open(elo_path)).get("processed_through", anchor_gen))
        except Exception:
            return anchor_gen

    def train_block(big, k: int):
        nonlocal consumed
        planes, policy, value = big[0], big[1], big[2]  # ignore played/has_next/next_planes
        n = planes.shape[0]
        net.train()
        acc = [0.0, 0.0]
        for _ in range(k):
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
            acc[0] += float(loss_v)
            acc[1] += float(loss_p)
        return acc

    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    snap_gen = max(existing + [anchor_gen])
    refresh()
    print(f"go9-az learner: device {device}  batch {args.batch_size}  throttle {args.max_reuse}x  "
          f"eval-lead {args.eval_lead}  snapshot/{args.steps_per_snapshot}  after gen {snap_gen}",
          flush=True)

    BLOCK = 50
    steps_into_snap = 0
    losses = [0.0, 0.0]
    nblk = 0
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

        block = min(BLOCK, args.steps_per_snapshot - steps_into_snap, budget // args.batch_size)
        acc = train_block(big, block)
        losses[0] += acc[0]
        losses[1] += acc[1]
        nblk += block
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
            print(f"snapshot gen{snap_gen}  samples {n}  chunks {len(cache)}  "
                  f"produced {produced}  consumed {consumed} ({consumed/max(1,produced):.1f}x)  "
                  f"loss v {losses[0]/max(1,nblk):.4f} p {losses[1]/max(1,nblk):.4f}", flush=True)
            steps_into_snap = 0
            losses = [0.0, 0.0]
            nblk = 0


if __name__ == "__main__":
    main()
