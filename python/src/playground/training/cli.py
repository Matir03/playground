"""Training CLI entry point. Stub."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="playground training")
    parser.add_argument("--game", required=True)
    parser.add_argument("--stage", required=True, choices=["encoder-reifier", "eval", "policy"])
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raise SystemExit(f"train {args.stage}: implement at M2")


if __name__ == "__main__":
    main()
