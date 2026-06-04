"""Eval CLI: rate the self-play generations (and baselines) by an Elo curve."""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np


def _participants(run_dir: str, baselines: bool) -> list[tuple[str, str]]:
    """(name, checkpoint) for each self-play generation plus baselines."""
    out: list[tuple[str, str]] = []
    if baselines:
        for name, path in [
            ("random", "data/models/connect4/random/checkpoint.safetensors"),
            ("bc-perfect", "data/models/connect4/bc-perfect/checkpoint.safetensors"),
        ]:
            if os.path.exists(path):
                out.append((name, path))
    for gen_dir in sorted(glob.glob(os.path.join(run_dir, "gen*"))):
        ckpt = os.path.join(gen_dir, "checkpoint.safetensors")
        if os.path.exists(ckpt):
            out.append((os.path.basename(gen_dir), ckpt))
    return out


def _rate(args) -> None:
    from .arena import load_net, match
    from .rating import fit_elo

    parts = _participants(args.run, not args.no_baselines)
    if len(parts) < 2:
        raise SystemExit("need at least two participants (run selfplay first)")
    names = [n for n, _ in parts]
    print("participants:", ", ".join(names))
    nets = {n: load_net(p) for n, p in parts}

    rng = np.random.default_rng(args.seed)
    k = len(names)
    score = np.zeros((k, k))
    games = np.zeros((k, k))
    for ia, ib in itertools.combinations(range(k), 2):
        a, b = names[ia], names[ib]
        sa = match(nets[a], nets[b], args.games, args.sims, rng)  # a's avg score
        score[ia][ib] += sa * args.games
        score[ib][ia] += (1 - sa) * args.games
        games[ia][ib] += args.games
        games[ib][ia] += args.games
        print(f"  {a} vs {b}: {sa:.2f}", flush=True)

    elo = fit_elo(names, score, games, anchor="random")
    print("\n=== Elo (random = 0) ===")
    for name in sorted(elo, key=lambda n: elo[n]):
        bar = "#" * max(0, int((elo[name] + 200) / 25))
        print(f"  {name:12} {elo[name]:7.0f}  {bar}")

    out = os.path.join(args.run, "ratings.json")
    with open(out, "w") as f:
        json.dump({"elo": elo, "games_per_pair": args.games, "sims": args.sims}, f, indent=2)
    print(f"\nwrote {out}")
    gens = [(n, elo[n]) for n in names if n.startswith("gen")]
    if gens:
        print("generation curve:", "  ".join(f"{n}:{e:.0f}" for n, e in gens))


def main() -> None:
    p = argparse.ArgumentParser(description="playground eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate", help="round-robin + Elo over self-play generations")
    r.add_argument("--run", default="data/models/connect4/selfplay")
    r.add_argument("--games", type=int, default=12, help="games per pair")
    r.add_argument("--sims", type=int, default=24, help="MCTS sims per move")
    r.add_argument("--no-baselines", action="store_true", help="generations only")
    r.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.cmd == "rate":
        _rate(args)


if __name__ == "__main__":
    main()
