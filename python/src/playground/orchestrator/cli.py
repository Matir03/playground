"""Orchestrator CLI entry point. Stub."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="playground orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run OrchestratorService (admin surface)")
    start = sub.add_parser("start-run", help="start a new self-play run")
    start.add_argument("--game", required=True)
    start.add_argument("--config", required=True)
    args = parser.parse_args()

    raise SystemExit(f"orchestrator {args.cmd}: implement in M2")


if __name__ == "__main__":
    main()
