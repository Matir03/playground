"""Generational training orchestrator for Hex AZ.

The actor (`hex-engine selfplay-loop`) runs continuously and separately. This loop
drives the *generational* cadence: every --period seconds of pure self-play it
spawns the learner to train exactly ONE generation, then (every --eval-every
generations) rates the new net two ways — against the previous snapshot for
relative progress, and against MoHex for absolute strength. Each is a fresh,
short-lived process that EXITS when done, so between trainings the actor
self-plays uncontended and the GPU holds its boost clock (see [[amazons_az]] for
why that matters on this box).

Run with /tmp/cuda_env.sh sourced and PYTHONPATH=python/src. The actor must
already be running; this loop does not own it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def snapshots(live: str) -> list[tuple[int, str]]:
    """(generation, checkpoint path) for every snapshot, oldest first."""
    out = []
    for d in glob.glob(os.path.join(live, "snapshots", "gen*")):
        m = re.fullmatch(r"gen(\d+)", os.path.basename(d))
        ckpt = os.path.join(d, "checkpoint.safetensors")
        if m and os.path.exists(ckpt):
            out.append((int(m.group(1)), ckpt))
    return sorted(out)


def record_result(live: str, gen: int, key: str, score: float) -> None:
    path = os.path.join(live, "results.json")
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
    data.setdefault(str(gen), {})[key] = score
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def score_from(output: str, prefix: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(prefix):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/hex_az/live")
    ap.add_argument("--engine", default="target/release/hex-engine")
    ap.add_argument("--python", default="python/.venv/bin/python")
    ap.add_argument("--period", type=float, default=900.0,
                    help="seconds of self-play (actor only) between generations")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--cuda", action="store_true")
    # Evaluation contends with the actor, so keep it occasional and small.
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=32)
    ap.add_argument("--eval-node-budget", type=int, default=200)
    ap.add_argument("--eval-threads", type=int, default=8)
    ap.add_argument("--mohex-bin", default="mohex")
    ap.add_argument("--mohex-games", type=int, default=4)
    ap.add_argument("--mohex-sims", type=int, default=400)
    ap.add_argument("--mohex-playouts", type=int, default=1000)
    args = ap.parse_args()

    cuda = ["--cuda"] if args.cuda else []
    learner = [args.python, "python/hex_az_learner.py", "--live", args.live,
               "--channels", str(args.channels), "--blocks", str(args.blocks),
               "--gen-examples", "0", "--exit-after-gens", "1"]

    print(f"train-cycle: {args.period:.0f}s self-play per generation; learner spawned per gen; "
          f"eval every {args.eval_every} gen(s)", flush=True)

    n = 0
    while True:
        print(f"[cycle] self-play {args.period:.0f}s (actor only) ...", flush=True)
        time.sleep(args.period)

        t0 = time.time()
        rc = subprocess.run(learner).returncode
        print(f"[cycle] learner trained 1 gen, exit {rc} ({time.time() - t0:.0f}s)", flush=True)
        if rc != 0:
            print("[cycle] WARNING: learner exited non-zero; continuing", flush=True)

        n += 1
        if n % args.eval_every:
            continue

        snaps = snapshots(args.live)
        if not snaps:
            continue
        gen, ckpt = snaps[-1]

        # Relative: the new net against the previous snapshot.
        if len(snaps) >= 2:
            prev = snaps[-2][1]
            r = subprocess.run(
                [args.engine, "eval-match", "--a", ckpt, "--b", prev, *cuda,
                 "--games", str(args.eval_games), "--node-budget", str(args.eval_node_budget),
                 "--threads", str(args.eval_threads)],
                capture_output=True, text=True)
            score = score_from(r.stdout, "A_score")
            if score is not None:
                record_result(args.live, gen, "vs_prev", score)
                print(f"[cycle] gen{gen} vs gen{snaps[-2][0]}: {score:.3f}", flush=True)

        # Absolute: the new net against MoHex.
        r = subprocess.run(
            [args.engine, "vs-mohex", "--model", ckpt, *cuda,
             "--mohex-bin", args.mohex_bin, "--games", str(args.mohex_games),
             "--sims", str(args.mohex_sims), "--mohex-playouts", str(args.mohex_playouts)],
            capture_output=True, text=True)
        score = score_from(r.stdout, "net_vs_mohex")
        if score is not None:
            record_result(args.live, gen, "vs_mohex", score)
            print(f"[cycle] gen{gen} vs mohex({args.mohex_playouts} playouts): {score:.3f}",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
