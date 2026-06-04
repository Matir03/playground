"""Training CLI: save the random net, generate perfect-bot data, train.

Subcommands:
  save-random  — instantiate the random-init net and save it as a checkpoint.
  gen-data     — play perfect-vs-random games, record perfect-bot moves only.
  train        — behavioral-clone the perfect bot from the recorded data.

See docs/training.md (model registry layout) and docs/algorithm.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _save_random(out: str, seed: int) -> None:
    import torch

    from playground.model_server.net import ConnectFourModel, save_checkpoint

    torch.manual_seed(seed)
    model = ConnectFourModel()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_checkpoint(model, out)
    meta_path = os.path.join(os.path.dirname(out), "meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {"kind": "random-init", "seed": seed, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
            f,
            indent=2,
        )
    print(f"saved random net to {out}")


def _gen_data(args) -> None:
    import numpy as np

    from .generate import generate

    states, cols, values = generate(
        args.engine, args.games, args.max_opening, args.nn_visits, args.nn_model, args.seed
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, states=states, cols=cols, values=values)
    print(f"wrote {len(cols)} perfect-bot positions to {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="playground training")
    sub = p.add_subparsers(dest="cmd", required=True)

    sr = sub.add_parser("save-random", help="save the random-init net as a checkpoint")
    sr.add_argument("--out", default="data/models/connect4/random/checkpoint.safetensors")
    sr.add_argument("--seed", type=int, default=0)

    gd = sub.add_parser(
        "gen-data", help="annotate random-net self-play with the perfect bot's policy + value"
    )
    gd.add_argument("--engine", default="127.0.0.1:7100")
    gd.add_argument("--games", type=int, default=300)
    gd.add_argument("--max-opening", type=int, default=8)
    gd.add_argument("--nn-visits", type=int, default=16)
    gd.add_argument("--nn-model", default="random", help="engine model name for self-play moves")
    gd.add_argument("--seed", type=int, default=0)
    gd.add_argument("--out", default="data/datasets/connect4_perfect.npz")

    sp = sub.add_parser("selfplay", help="train a larger net by pure self-play (no solver)")
    sp.add_argument("--out", default="data/models/connect4/selfplay")
    sp.add_argument("--channels", type=int, default=64)
    sp.add_argument("--blocks", type=int, default=5)
    sp.add_argument("--generations", type=int, default=8)
    sp.add_argument("--games-per-gen", type=int, default=120)
    sp.add_argument("--sims", type=int, default=60)
    sp.add_argument("--epochs", type=int, default=4)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--buffer-gens", type=int, default=4)
    sp.add_argument("--c-puct", type=float, default=1.5)
    sp.add_argument("--dir-alpha", type=float, default=1.0)
    sp.add_argument("--dir-eps", type=float, default=0.25)
    sp.add_argument("--temp-moves", type=int, default=10)
    sp.add_argument("--eval-games", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)

    tr = sub.add_parser("train", help="behavioral-clone the perfect bot")
    tr.add_argument("--data", default="data/datasets/connect4_perfect.npz")
    tr.add_argument("--out", default="data/models/connect4/bc-perfect")
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--val-frac", type=float, default=0.1)
    tr.add_argument("--init", default=None, help="optional checkpoint to warm-start from")
    tr.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "save-random":
        _save_random(args.out, args.seed)
    elif args.cmd == "gen-data":
        _gen_data(args)
    elif args.cmd == "selfplay":
        from .selfplay import selfplay_train

        selfplay_train(
            args.out, args.channels, args.blocks, args.generations, args.games_per_gen,
            args.sims, args.epochs, args.batch_size, args.lr, args.buffer_gens,
            args.c_puct, args.dir_alpha, args.dir_eps, args.temp_moves, args.eval_games, args.seed,
        )
    elif args.cmd == "train":
        from .train import train

        train(
            args.data,
            args.out,
            args.epochs,
            args.batch_size,
            args.lr,
            args.val_frac,
            args.init,
            args.seed,
        )


if __name__ == "__main__":
    main()
