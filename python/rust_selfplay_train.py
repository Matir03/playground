"""Self-play training driven by the Rust generator.

Each generation: dump the current net to a checkpoint, shell out to
`connect4-engine selfplay-gen` (which plays the games on native threads, no GIL,
with a batched GPU inference server) to produce training examples, then train the
net on them in torch — exactly the train step of `training/selfplay.py`, but with
game generation done in Rust. Checkpoints land in `<out>/genNNN/` so
`playground-eval rate` can draw the Elo curve.

Run with the CUDA env sourced (see /tmp/cuda_env.sh) so the Rust binary finds the
GPU libs:  source /tmp/cuda_env.sh && python rust_selfplay_train.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.net import (
    ConnectFourResNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)

RECORD_F32 = 2 * 6 * 7 + 7 + 1  # planes + policy + value (must match selfplay.rs)


def read_examples(path: str):
    """Read the Rust example dump: u64 count, then count×RECORD_F32 little-endian f32."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        arr = np.frombuffer(f.read(), dtype="<f4")
    arr = arr.reshape(n, RECORD_F32)
    planes = arr[:, :84].reshape(n, 2, 6, 7).copy()
    policy = arr[:, 84:91].copy()
    value = arr[:, 91].copy()
    return planes, policy, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/models/connect4/selfplay-rust")
    ap.add_argument("--rust-bin", default="target/release/connect4-engine")
    ap.add_argument("--init", default=None, help="warm-start checkpoint")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--games-per-gen", type=int, default=96)
    ap.add_argument("--move-ms", type=float, default=1000.0)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer-gens", type=int, default=4)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    net = ConnectFourResNet(channels=args.channels, blocks=args.blocks)

    # Resume: continue numbering from existing gens, warm-start from the latest
    # (or from --init).
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
    print(f"device {device}  games/gen {args.games_per_gen}  move-ms {args.move_ms:.0f}  "
          f"threads {args.threads}", flush=True)

    # The worker checkpoint the Rust generator loads each gen; it reads the arch
    # from a sibling config.json, so write that too.
    work_dir = os.path.join(args.out, "_worker")
    os.makedirs(work_dir, exist_ok=True)
    work_ckpt = os.path.join(work_dir, "checkpoint.safetensors")
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump({"arch": arch_of(net)}, f)
    examples_bin = os.path.join(work_dir, "examples.bin")

    window: deque = deque(maxlen=args.buffer_gens)
    ckpt = os.path.join(args.out, "checkpoint.safetensors")

    for gen in range(args.generations):
        gen_num = start_gen + gen + 1
        t0 = time.time()

        # 1) dump current weights for the Rust generator (safetensors wants CPU).
        net.eval()
        net.to("cpu")
        save_checkpoint(net, work_ckpt)
        net.to(device)

        # 2) generate one generation of self-play in Rust.
        cmd = [
            args.rust_bin, "selfplay-gen",
            "--model", work_ckpt, "--out", examples_bin,
            "--cuda", "--batched", "--max-batch", str(args.max_batch),
            "--games", str(args.games_per_gen), "--threads", str(args.threads),
            "--move-ms", str(args.move_ms),
            "--c-puct", str(args.c_puct), "--temp-moves", str(args.temp_moves),
            "--seed", str(args.seed * 1_000_000 + gen_num),
        ]
        subprocess.run(cmd, check=True)
        planes_np, policy_np, value_np = read_examples(examples_bin)

        # 3) train on a replay window of the last buffer-gens generations.
        window.append((planes_np, policy_np, value_np))
        planes = torch.from_numpy(np.concatenate([w[0] for w in window])).to(device)
        target_pi = torch.from_numpy(np.concatenate([w[1] for w in window])).to(device)
        target_z = torch.from_numpy(np.concatenate([w[2] for w in window])).to(device)

        net.train()
        rng = np.random.default_rng(args.seed + gen_num)
        idx = np.arange(planes.shape[0])
        for _ in range(args.epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), args.batch_size):
                b = torch.from_numpy(idx[i : i + args.batch_size]).to(device)
                v_pred, logits = net(planes[b])
                loss_v = F.mse_loss(v_pred, target_z[b])
                loss_p = -(target_pi[b] * F.log_softmax(logits, dim=1)).sum(1).mean()
                loss = loss_v + loss_p
                opt.zero_grad()
                loss.backward()
                opt.step()

        # 4) checkpoint this generation (+ mirror to the top level for serving).
        net.eval()
        gen_dir = os.path.join(args.out, f"gen{gen_num:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        cfg = {"arch": arch_of(net), "generation": gen_num, "winrate_vs_random": None}
        net.to("cpu")
        save_checkpoint(net, os.path.join(gen_dir, "checkpoint.safetensors"))
        save_checkpoint(net, ckpt)
        net.to(device)
        for path in (os.path.join(gen_dir, "config.json"), os.path.join(args.out, "config.json")):
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)

        print(f"gen {gen_num} (+{gen + 1}/{args.generations})  "
              f"examples {planes.shape[0]} (this gen {planes_np.shape[0]})  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
