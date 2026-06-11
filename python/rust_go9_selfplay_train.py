"""9×9 Go self-play training, driven by the Rust generator.

Each generation: dump the current net, shell out to `go9-engine selfplay-gen`
(games on native threads, batched GPU inference server, no GIL), read the
examples, and train in torch — value MSE on the game-outcome z, policy CE on the
MCTS visit distribution (82 actions), and a reifier-consistency loss training
`pol_next(gs)[a_played]` toward `stopgrad(encode(next_planes))` (MuZero 1-step).
Checkpoints land in `<out>/genNNN/` for play + rating.

Data stays on the host and is moved to the GPU per batch (the 96ch/10b net plus
next_planes won't comfortably fit a full window on an 8GB card).

Run with the CUDA env sourced (see /tmp/cuda_env.sh) so the Rust binary finds the
GPU libs:  source /tmp/cuda_env.sh && python rust_go9_selfplay_train.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from collections import deque

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

CELLS = 81
PLANES = 3 * CELLS          # 243
ACTIONS = CELLS + 1         # 82
# Record layout (must match crates/games/go9/src/selfplay.rs):
# planes(243) + policy(82) + value(1) + played_action(1) + has_next(1) + next_planes(243)
RECORD_F32 = PLANES + ACTIONS + 1 + 1 + 1 + PLANES


def read_examples(path: str):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(n, RECORD_F32)
    o = 0
    planes = arr[:, o:o + PLANES].reshape(n, 3, 9, 9).copy(); o += PLANES
    policy = arr[:, o:o + ACTIONS].copy(); o += ACTIONS
    value = arr[:, o].copy(); o += 1
    played = arr[:, o].astype(np.int64).copy(); o += 1
    has_next = arr[:, o].copy(); o += 1
    next_planes = arr[:, o:o + PLANES].reshape(n, 3, 9, 9).copy(); o += PLANES
    return planes, policy, value, played, has_next, next_planes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/models/go9/selfplay")
    ap.add_argument("--rust-bin", default="target/release/go9-engine")
    ap.add_argument("--init", default=None, help="warm-start checkpoint")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--games-per-gen", type=int, default=128)
    ap.add_argument("--node-budget", type=int, default=600)
    ap.add_argument("--move-ms", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer-gens", type=int, default=3)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=int, default=16)
    ap.add_argument("--reif-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    net = Go9ResNet(channels=args.channels, blocks=args.blocks)

    existing = sorted(glob.glob(os.path.join(args.out, "gen*")))
    start_gen = int(os.path.basename(existing[-1])[3:]) if existing else 0
    if args.init:
        load_checkpoint(net, args.init)
        print(f"warm-started from {args.init}", flush=True)
    elif start_gen > 0:
        resume = os.path.join(args.out, f"gen{start_gen:03d}", "checkpoint.safetensors")
        load_checkpoint(net, resume)
        print(f"resumed from {resume} (after gen {start_gen})", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    budget = (f"node-budget {args.node_budget}" if args.node_budget > 0
              else f"move-ms {args.move_ms:.0f}")
    print(f"device {device}  games/gen {args.games_per_gen}  {budget}  "
          f"threads {args.threads}  net {args.channels}ch/{args.blocks}b  "
          f"reif-weight {args.reif_weight}", flush=True)

    work_dir = os.path.join(args.out, "_worker")
    os.makedirs(work_dir, exist_ok=True)
    work_ckpt = os.path.join(work_dir, "checkpoint.safetensors")
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump({"arch": arch_of(net)}, f)
    examples_bin = os.path.join(work_dir, "examples.bin")

    window: deque = deque(maxlen=args.buffer_gens)
    history: list = []
    ckpt = os.path.join(args.out, "checkpoint.safetensors")

    for gen in range(args.generations):
        gen_num = start_gen + gen + 1
        t0 = time.time()

        # 1) dump current weights for the Rust generator.
        net.eval(); net.to("cpu")
        save_checkpoint(net, work_ckpt)
        net.to(device)

        # 2) generate one generation of self-play in Rust.
        cmd = [
            args.rust_bin, "selfplay-gen",
            "--model", work_ckpt, "--out", examples_bin,
            "--cuda", "--batched", "--max-batch", str(args.max_batch),
            "--games", str(args.games_per_gen), "--threads", str(args.threads),
            "--c-puct", str(args.c_puct), "--temp-moves", str(args.temp_moves),
            "--seed", str(args.seed * 1_000_000 + gen_num),
        ]
        if args.node_budget > 0:
            cmd += ["--node-budget", str(args.node_budget)]
        else:
            cmd += ["--move-ms", str(args.move_ms)]
        proc = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        print(proc.stderr.rstrip(), flush=True)
        m = re.search(r"([\d.]+) nodes/move", proc.stderr)
        nodes_per_move = float(m.group(1)) if m else float("nan")

        # 3) replay window (drops games older than buffer-gens). Keep on host.
        window.append(read_examples(examples_bin))
        planes = np.concatenate([w[0] for w in window])
        policy = np.concatenate([w[1] for w in window])
        value = np.concatenate([w[2] for w in window])
        played = np.concatenate([w[3] for w in window])
        has_next = np.concatenate([w[4] for w in window])
        next_planes = np.concatenate([w[5] for w in window])
        n = planes.shape[0]

        net.train()
        rng = np.random.default_rng(args.seed + gen_num)
        idx = np.arange(n)
        reif_w = args.reif_weight
        ep_v = ep_p = ep_r = ep_n = 0.0
        for _ in range(args.epochs):
            rng.shuffle(idx)
            ep_v = ep_p = ep_r = ep_n = 0.0
            for i in range(0, n, args.batch_size):
                b = idx[i : i + args.batch_size]
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
                opt.zero_grad(); loss.backward(); opt.step()
                bn = len(b)
                ep_v += float(loss_v) * bn
                ep_p += float(loss_p) * bn
                ep_r += float(loss_r) * bn
                ep_n += bn
        value_loss, policy_loss, reif_loss = ep_v / ep_n, ep_p / ep_n, ep_r / ep_n

        # 4) checkpoint this generation (+ mirror to top level for serving).
        net.eval()
        gen_dir = os.path.join(args.out, f"gen{gen_num:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        cfg = {"arch": arch_of(net), "generation": gen_num}
        net.to("cpu")
        save_checkpoint(net, os.path.join(gen_dir, "checkpoint.safetensors"))
        save_checkpoint(net, ckpt)
        net.to(device)
        for path in (os.path.join(gen_dir, "config.json"), os.path.join(args.out, "config.json")):
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)

        history.append({
            "gen": gen_num, "examples_this_gen": int(window[-1][0].shape[0]),
            "buffer_gens": len(window), "buffer_examples": int(n),
            "nodes_per_move": nodes_per_move, "value_loss": value_loss,
            "policy_loss": policy_loss, "reif_loss": reif_loss,
        })
        with open(os.path.join(args.out, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        reif_note = f" r {reif_loss:.4f}" if reif_w > 0 else ""
        print(f"gen {gen_num} (+{gen + 1}/{args.generations})  "
              f"this_gen {window[-1][0].shape[0]}  buffer {len(window)}gens/{n}ex  "
              f"nodes/move {nodes_per_move:.0f}  "
              f"loss v {value_loss:.4f} p {policy_loss:.4f}{reif_note}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
