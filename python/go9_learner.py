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
    LGS_DIM,
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


def export_onnx_atomic(net: Go9ResNet, path: str) -> None:
    """Export the value+policy path to ONNX (dynamic batch) for the ORT actor to
    hot-swap to. Written atomically so the actor never reads a half-written file."""
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
    ap.add_argument("--live", default="data/models/go9/live")
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
    # Publish an initial ONNX so an ORT-backed actor has weights to start from.
    onnx_path = os.path.join(live, "model.onnx")
    export_onnx_atomic(net, onnx_path)

    cfg = json.load(open(os.path.join(live, "config.json")))
    anchor_gen = int(cfg.get("generation", 0))
    elo_path = os.path.join(live, "elo.json")
    reif_w = args.reif_weight

    cache: dict[str, tuple] = {}  # chunk path -> arrays
    big: tuple | None = None      # concatenated arrays for sampling
    produced = 0  # cumulative samples the actor has produced (4x-throttle base)
    consumed = 0  # cumulative samples trained on

    def refresh() -> bool:
        nonlocal produced
        files = set(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))
        changed = False
        for f in files - cache.keys():
            try:
                ex = read_examples(f)
            except Exception:
                continue  # mid-rename / short read; pick it up next refresh
            cache[f] = ex
            produced += ex[0].shape[0]
            changed = True
        for f in list(cache.keys() - files):
            del cache[f]
            changed = True
        return changed

    def processed_through() -> int:
        """Highest snapshot gen the evaluator has finished (or skipped)."""
        try:
            return int(json.load(open(elo_path)).get("processed_through", anchor_gen))
        except Exception:
            return anchor_gen

    def train_block(big, k: int):
        nonlocal consumed
        planes, policy, value, played, has_next, next_planes = big
        n = planes.shape[0]
        net.train()
        acc = [0.0, 0.0, 0.0]
        for _ in range(k):
            b = rng.integers(0, n, size=args.batch_size)
            pl = torch.from_numpy(planes[b]).to(device)
            tgt_pi = torch.from_numpy(policy[b]).to(device)
            tgt_z = torch.from_numpy(value[b]).to(device)
            lgs = net.encode_batch(pl)
            v_pred = net.value_batch(lgs)
            logits = net.pol_prior(lgs)
            loss_v = F.mse_loss(v_pred, tgt_z)
            loss_p = -(tgt_pi * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss = loss_v + loss_p
            loss_r = torch.zeros((), device=device)
            if reif_w > 0:
                npl = torch.from_numpy(next_planes[b]).to(device)
                hn = torch.from_numpy(has_next[b]).to(device)
                pa = torch.from_numpy(played[b]).to(device)
                pred_next = torch.tanh(net.pol_next(lgs)).view(-1, ACTIONS, LGS_DIM)
                pred_a = pred_next[torch.arange(pred_next.shape[0], device=device), pa]
                with torch.no_grad():
                    tgt_next = net.encode_batch(npl)
                per = ((pred_a - tgt_next) ** 2).mean(1) * hn
                loss_r = per.sum() / hn.sum().clamp(min=1.0)
                loss = loss + reif_w * loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
            consumed += args.batch_size
            acc[0] += float(loss_v); acc[1] += float(loss_p); acc[2] += float(loss_r)
        return acc

    # Snapshot numbering continues from the highest existing snapshot.
    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    snap_gen = max(existing + [anchor_gen])
    refresh()  # seed `produced` with the current buffer
    print(f"learner: device {device}  batch {args.batch_size}  "
          f"throttle {args.max_reuse}x actor  eval-lead {args.eval_lead}  "
          f"snapshot/{args.steps_per_snapshot} steps  continuing after gen {snap_gen}", flush=True)

    BLOCK = 50
    steps_into_snap = 0
    losses = [0.0, 0.0, 0.0]
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

        # Eval-lead gate: don't begin the next snapshot (snap_gen+1) until the
        # evaluator has finished (snap_gen+1) - eval_lead.
        if steps_into_snap == 0:
            need = (snap_gen + 1) - args.eval_lead
            if need > anchor_gen and processed_through() < need:
                time.sleep(2.0)
                continue

        # 4x throttle: never train on more than max_reuse * produced samples.
        budget = int(args.max_reuse * produced) - consumed
        if budget < args.batch_size:
            time.sleep(2.0)
            continue

        block = min(BLOCK, args.steps_per_snapshot - steps_into_snap, budget // args.batch_size)
        acc = train_block(big, block)
        for i in range(3):
            losses[i] += acc[i]
        nblk += block
        steps_into_snap += block
        net.eval()
        save_atomic(net, latest)  # publish for actor hot-swap

        if steps_into_snap >= args.steps_per_snapshot:
            snap_gen += 1
            d = os.path.join(snaps, f"gen{snap_gen:03d}")
            os.makedirs(d, exist_ok=True)
            save_checkpoint(net.to("cpu"), os.path.join(d, "checkpoint.safetensors"))
            net.to(device)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"arch": arch_of(net), "generation": snap_gen}, f)
            export_onnx_atomic(net, onnx_path)  # refresh for the ORT actor's hot-swap
            print(f"snapshot gen{snap_gen}  samples {n}  chunks {len(cache)}  "
                  f"produced {produced}  consumed {consumed} ({consumed/max(1,produced):.1f}x)  "
                  f"loss v {losses[0]/max(1,nblk):.4f} p {losses[1]/max(1,nblk):.4f} "
                  f"r {losses[2]/max(1,nblk):.4f}", flush=True)
            steps_into_snap = 0
            losses = [0.0, 0.0, 0.0]
            nblk = 0


if __name__ == "__main__":
    main()
