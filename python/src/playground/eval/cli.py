"""Eval CLI entry point. Stub."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="playground eval")
    parser.add_argument("--game", required=True)
    parser.add_argument("--player", action="append", required=True,
                        help="URI for a tournament participant; repeat for each player")
    parser.add_argument("--games-per-pair", type=int, default=20)
    args = parser.parse_args()

    raise SystemExit(f"eval over {len(args.player)} players: implement at M2")


if __name__ == "__main__":
    main()
