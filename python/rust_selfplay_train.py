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
import re
import subprocess
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.net import (
    LGS_DIM,
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


def child_planes_and_mask(planes: np.ndarray):
    """For each state's side-to-move planes, the planes of each of the 7 children
    plus a legality mask — derived purely in the own/opponent frame (no board).

    planes[:,0]=own (side to move), planes[:,1]=opponent, row 0 = bottom. Playing
    column c drops an own stone in the lowest empty row, then it's the opponent's
    turn, so the child's side-to-move planes swap: own_child = opponent,
    opp_child = own + the new stone. Returns (child[N,7,2,6,7], mask[N,7]).
    """
    n = planes.shape[0]
    own, opp = planes[:, 0], planes[:, 1]          # [N,6,7]
    occupied = own + opp
    height = occupied.sum(axis=1)                   # [N,7] filled cells per column
    child = np.zeros((n, 7, 2, 6, 7), dtype=np.float32)
    mask = (height < 6).astype(np.float32)          # [N,7] legal = not full
    rows = np.arange(n)
    for c in range(7):
        h = np.clip(height[:, c].astype(int), 0, 5)  # drop row (clamped for full)
        own_next = own.copy()
        own_next[rows, h, c] = 1.0                    # place the mover's stone
        child[:, c, 0] = opp                          # perspective flip
        child[:, c, 1] = own_next
    return child, mask


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
    ap.add_argument("--node-budget", type=int, default=0,
                    help="per-move node budget; overrides --move-ms when > 0")
    ap.add_argument("--reif-weight", type=float, default=0.0,
                    help="weight of the reifier-consistency aux loss (0 = off). "
                         "Trains pol_next(lgs)[c] to match encode(child_c) "
                         "(stop-grad target), MuZero-style latent dynamics.")
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
    budget = (f"node-budget {args.node_budget}" if args.node_budget > 0
              else f"move-ms {args.move_ms:.0f}")
    print(f"device {device}  games/gen {args.games_per_gen}  {budget}  "
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
    history: list = []
    ckpt = os.path.join(args.out, "checkpoint.safetensors")

    for gen in range(args.generations):
        gen_num = start_gen + gen + 1
        t0 = time.time()

        # 1) dump current weights for the Rust generator (safetensors wants CPU).
        net.eval()
        net.to("cpu")
        save_checkpoint(net, work_ckpt)
        net.to(device)

        # 2) generate one generation of self-play in Rust (capture stderr to
        #    pull out nodes/move, which the Rust generator reports).
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
        planes_np, policy_np, value_np = read_examples(examples_bin)

        # 3) train on a replay window of the last buffer-gens generations. The
        #    deque(maxlen) drops games older than that automatically — stale,
        #    off-policy games stop being trained on once they age out.
        window.append((planes_np, policy_np, value_np))
        planes_all = np.concatenate([w[0] for w in window])
        planes = torch.from_numpy(planes_all).to(device)
        target_pi = torch.from_numpy(np.concatenate([w[1] for w in window])).to(device)
        target_z = torch.from_numpy(np.concatenate([w[2] for w in window])).to(device)

        reif_w = args.reif_weight
        if reif_w > 0:
            # Per-state child planes + legality mask for the reifier target.
            child_np, mask_np = child_planes_and_mask(planes_all)
            child_t = torch.from_numpy(child_np).to(device)   # [N,7,2,6,7]
            mask_t = torch.from_numpy(mask_np).to(device)     # [N,7]

        net.train()
        rng = np.random.default_rng(args.seed + gen_num)
        idx = np.arange(planes.shape[0])
        # Track the final epoch's mean losses as the converged-fit signal.
        ep_v = ep_p = ep_r = ep_n = 0.0
        for epoch in range(args.epochs):
            rng.shuffle(idx)
            ep_v = ep_p = ep_r = ep_n = 0.0
            for i in range(0, len(idx), args.batch_size):
                b = torch.from_numpy(idx[i : i + args.batch_size]).to(device)
                lgs = net.encode_batch(planes[b])
                v_pred = net.value_batch(lgs)
                logits = net.pol_prior(lgs)
                loss_v = F.mse_loss(v_pred, target_z[b])
                loss_p = -(target_pi[b] * F.log_softmax(logits, dim=1)).sum(1).mean()
                loss = loss_v + loss_p
                loss_r = torch.zeros((), device=device)
                if reif_w > 0:
                    # Predicted next-LGS per column vs the encoder's own embedding
                    # of the actual child (stop-grad target), masked to legal moves.
                    pred_next = torch.tanh(net.pol_next(lgs)).view(-1, 7, LGS_DIM)
                    with torch.no_grad():
                        ch = child_t[b].view(-1, 2, 6, 7)
                        tgt_next = net.encode_batch(ch).view(-1, 7, LGS_DIM)
                    msk = mask_t[b]
                    per = ((pred_next - tgt_next) ** 2).mean(-1) * msk   # [B,7]
                    loss_r = per.sum() / msk.sum().clamp(min=1.0)
                    loss = loss + reif_w * loss_r
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_v += float(loss_v) * b.shape[0]
                ep_p += float(loss_p) * b.shape[0]
                ep_r += float(loss_r) * b.shape[0]
                ep_n += b.shape[0]
        value_loss, policy_loss, reif_loss = ep_v / ep_n, ep_p / ep_n, ep_r / ep_n

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

        buf_examples = int(planes.shape[0])
        history.append({
            "gen": gen_num,
            "examples_this_gen": int(planes_np.shape[0]),
            "buffer_gens": len(window),
            "buffer_examples": buf_examples,
            "nodes_per_move": nodes_per_move,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "reif_loss": reif_loss,
            "total_loss": value_loss + policy_loss,
        })
        with open(os.path.join(args.out, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        reif_note = f" r {reif_loss:.4f}" if reif_w > 0 else ""
        print(f"gen {gen_num} (+{gen + 1}/{args.generations})  "
              f"this_gen {planes_np.shape[0]}  buffer {len(window)}gens/{buf_examples}ex  "
              f"nodes/move {nodes_per_move:.0f}  "
              f"loss v {value_loss:.4f} p {policy_loss:.4f}{reif_note} "
              f"tot {value_loss + policy_loss:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
