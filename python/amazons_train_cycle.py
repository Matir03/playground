"""Generational training orchestrator for Amazons AZ.

The actor (`amazons-engine selfplay-loop`) runs continuously and separately. This
loop drives the *generational* cadence: every --period seconds of pure self-play,
it spawns the learner to train exactly ONE generation, then the evaluator to rate
that generation against the gauntlet — each as a fresh, short-lived process that
EXITS when done. So between trainings the learner and evaluator are fully gone (not
just stopped): the actor self-plays uncontended and the GPU holds its boost clock
(~2565 MHz / ~26k nps vs the ~16k it falls to when its demand dips). See
[[amazons_az]] for why that matters.

Run it with /tmp/cuda_env.sh sourced (the evaluator shells out to the CUDA Rust
binary) and PYTHONPATH=python/src. The actor must already be running; this loop
does not own it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/amazons_az/live")
    ap.add_argument("--period", type=float, default=900.0,
                    help="seconds of self-play (actor only) between generations")
    ap.add_argument("--python", default="python/.venv/bin/python")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--eval-node-budget", type=int, default=100)
    ap.add_argument("--eval-threads", type=int, default=8)
    ap.add_argument("--eval-n-opponents", type=int, default=2)
    # Eval is GPU-inefficient (few games -> tiny batches; ~2min/match) and does NOT
    # pause the actor, so it contends. Run it only every Nth generation to keep the
    # contended window a small fraction of wall time.
    ap.add_argument("--eval-every", type=int, default=3)
    args = ap.parse_args()

    learner = [args.python, "python/amazons_az_learner.py", "--live", args.live,
               "--channels", str(args.channels), "--blocks", str(args.blocks),
               "--eval-lead", "9999", "--gen-examples", "0", "--exit-after-gens", "1"]
    evaluator = [args.python, "python/amazons_eval_loop.py", "--live", args.live,
                 "--games", str(args.eval_games), "--node-budget", str(args.eval_node_budget),
                 "--threads", str(args.eval_threads), "--n-opponents",
                 str(args.eval_n_opponents), "--once"]

    print(f"train-cycle: {args.period:.0f}s self-play per generation; learner spawned per "
          f"gen; eval every {args.eval_every} gen(s) (no actor pause)", flush=True)

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
        if n % args.eval_every == 0:
            t0 = time.time()
            rc = subprocess.run(evaluator).returncode
            print(f"[cycle] evaluator rated latest gen, exit {rc} ({time.time() - t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
