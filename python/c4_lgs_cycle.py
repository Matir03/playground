"""Generational training cycle for the Connect 4 reification-based LGS net.

Alternates self-play (the central-batched lockstep engine) with a learner, with a
sliding-window replay buffer — the simple generational cadence (cf.
amazons_train_cycle.py). One generation:

  1. actor : `connect4-engine lgs-gen` → live/buffer/chunk_<gen>.bin
  2. prune : keep the last --window chunks
  3. learn : `c4_lgs_train.py --buffer live/buffer` → live/checkpoint.safetensors

The next generation's self-play hot-swaps the new checkpoint. Run with
/tmp/cuda_env.sh sourced and PYTHONPATH=python/src when using --cuda.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def sh(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prune(buffer: str, keep: int) -> None:
    chunks = sorted(glob.glob(os.path.join(buffer, "chunk_*.bin")))
    for path in chunks[:-keep] if keep > 0 else []:
        os.remove(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/connect4_lgs/live")
    ap.add_argument("--engine", default="target/release/connect4-engine")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gens", type=int, default=0, help="generations to run (0 = unbounded)")
    ap.add_argument("--games", type=int, default=512, help="self-play games per gen")
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--temp-moves", type=int, default=8)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=64, help="games in flight per worker")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=10, help="replay buffer chunks to keep")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--device", default="cpu", help="learner device (cpu/cuda)")
    ap.add_argument("--eval-every", type=int, default=1, help="run lgs-eval vs solver every N gens (0 disables)")
    ap.add_argument("--eval-positions", type=int, default=200)
    ap.add_argument("--eval-sims", type=int, default=96)
    ap.add_argument("--eval-opening", type=int, default=6)
    ap.add_argument("--ladder-every", type=int, default=1, help="run BT-ELO ladder + promotion every N gens (0 disables)")
    ap.add_argument("--match-games", type=int, default=40)
    ap.add_argument("--match-sims", type=int, default=64)
    ap.add_argument("--match-random", type=int, default=3, help="random previous gens to also play each ladder step")
    # Continuous actor/learner: generate self-play chunks back-to-back for
    # --train-interval-secs (full replenishing pool, games >> pool), then train
    # once. 0 = legacy one-chunk-per-train behaviour.
    ap.add_argument("--train-interval-secs", type=int, default=0)
    ap.add_argument("--chunk-games", type=int, default=4096, help="games per self-play chunk (>> pool)")
    args = ap.parse_args()
    import c4_lgs_ladder as ladder
    import time

    buffer = os.path.join(args.live, "buffer")
    ckpt = os.path.join(args.live, "checkpoint.safetensors")
    os.makedirs(buffer, exist_ok=True)
    train = "python/c4_lgs_train.py"

    if not os.path.exists(ckpt):
        print("[init] no checkpoint; seeding a random LGS net", flush=True)
        sh([args.python, train, "--init", args.live, "--channels", str(args.channels), "--blocks", str(args.blocks)])

    gens_dir = os.path.join(args.live, "gens")
    ladder_path = os.path.join(args.live, "ladder.json")
    chunk_seq = len(glob.glob(os.path.join(buffer, "chunk_*.bin")))
    # Resume gen numbering from existing archives so restarts never overwrite a
    # gen_NNNNNN snapshot (which would corrupt the ladder's match history).
    existing = sorted(glob.glob(os.path.join(gens_dir, "gen_*.safetensors")))
    gen = (int(os.path.basename(existing[-1])[4:10]) + 1) if existing else 0
    while args.gens == 0 or gen < args.gens:
        # Self-play phase: keep a full replenishing pool busy generating chunks
        # (games >> pool) back-to-back until the train interval elapses, so the
        # actor runs at bench throughput between (infrequent) training steps.
        print(f"\n=== round {gen} : self-play ===", flush=True)
        round_start = time.time()
        chunk_games = args.chunk_games if args.train_interval_secs > 0 else args.games
        while True:
            out = os.path.join(buffer, f"chunk_{chunk_seq:06}.bin")
            gen_cmd = [
                args.engine, "lgs-gen", "--model", ckpt, "--out", out,
                "--games", str(chunk_games), "--sims", str(args.sims),
                "--temp-moves", str(args.temp_moves), "--concurrency", str(args.concurrency),
                "--threads", str(args.threads), "--max-batch", str(args.max_batch),
                "--seed", str(chunk_seq * 1_000_003),
            ]
            if args.cuda:
                gen_cmd.append("--cuda")
            sh(gen_cmd)
            chunk_seq += 1
            prune(buffer, args.window)
            if args.train_interval_secs <= 0 or time.time() - round_start >= args.train_interval_secs:
                break
        print(f"=== round {gen} : train ===", flush=True)
        sh([
            args.python, train, "--buffer", buffer, "--in", ckpt, "--out", ckpt,
            "--epochs", str(args.epochs), "--batch", str(args.batch), "--lr", str(args.lr),
            "--device", args.device,
        ])
        if args.eval_every and (gen + 1) % args.eval_every == 0:
            eval_cmd = [
                args.engine, "lgs-eval", "--model", ckpt,
                "--positions", str(args.eval_positions), "--sims", str(args.eval_sims),
                "--opening", str(args.eval_opening), "--seed", str(gen),
            ]
            if args.cuda:
                eval_cmd.append("--cuda")
            sh(eval_cmd)
        if args.ladder_every and (gen + 1) % args.ladder_every == 0:
            try:
                L, name, promoted, elo = ladder.step(
                    args.engine, ckpt, gens_dir, ladder_path, gen, args.cuda,
                    games=args.match_games, sims=args.match_sims,
                    n_random=args.match_random, seed=gen * 7919,
                )
                tag = "PROMOTED to champion" if promoted else f"champion stays {L['best']}"
                print(f"LADDER gen {gen} elo {elo.get(name, float('nan')):.0f}  "
                      f"best {L['best']} ({elo.get(L['best'], float('nan')):.0f})  {tag}", flush=True)
                print(f"  top: {ladder.elo_table(L)}", flush=True)
            except Exception as e:  # never let the ladder kill training
                print(f"[ladder error] {e}", flush=True)
        gen += 1

    print("\n[done] cycle complete", flush=True)


if __name__ == "__main__":
    main()
