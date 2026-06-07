"""Self-play generation throughput benchmark (Python side of the Rust compare).

Times ONLY game generation (no training) at a fixed simulation count, so it's
apples-to-apples with `connect4-engine selfplay-bench`. Reports sims/sec.

  python bench_selfplay.py --model <ckpt> --games 64 --workers 32 --sims 150 --device cuda
"""

from __future__ import annotations

import argparse
import time

import torch

from playground.model_server.net import ConnectFourResNet, load_checkpoint
from playground.training.selfplay import _generate_games_mp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ConnectFourResNet(channels=args.channels, blocks=args.blocks)
    load_checkpoint(net, args.model)
    net.to(device)
    net.eval()

    params = {
        "sims": args.sims, "c_puct": 1.5, "dir_alpha": 1.0, "dir_eps": 0.25,
        "temp_moves": 10, "move_ms": 0.0,  # fixed sims, not a time budget
    }
    print(
        f"python self-play: {args.games} games, {args.workers} workers, "
        f"{args.sims} sims/move, device {device}, model {args.model}",
        flush=True,
    )
    t0 = time.time()
    results, avg_batch = _generate_games_mp(
        net, device, args.games, params, args.workers, args.max_batch, args.seed
    )
    elapsed = time.time() - t0

    total_sims = 0
    examples = 0
    for ex, avg_sims in results:
        examples += len(ex)
        total_sims += int(round(avg_sims * len(ex)))
    sims_per_s = total_sims / elapsed
    games_per_s = args.games / elapsed
    print(
        f"games {args.games}  examples {examples}  total_sims {total_sims}  "
        f"time {elapsed:.2f}s  sims/s {sims_per_s:.0f}  games/s {games_per_s:.2f}  "
        f"avg_batch {avg_batch:.0f}"
    )


if __name__ == "__main__":
    main()
