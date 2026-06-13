"""Continuous Connect 4 evaluator: score each new learner snapshot against the
PERFECT solver (absolute distance-to-perfect, not chained self-play ELO).

To keep the GPU free (and the matches fast + load-independent), the evaluator:
  * runs INFREQUENTLY — only the latest snapshot, once it is `--every` gens past
    the last one scored (intermediate snapshots are skipped, not queued);
  * PAUSES self-play + training around each match — it raises `live/PAUSE`, waits
    for the actor + learner to park (they drop `actor.paused` / `learner.paused`),
    runs `connect4-engine eval-solver` (parallel games, batched on the full GPU at
    a FIXED node budget), then clears PAUSE so they resume.

It records the score in live/scores.json, promotes the best-by-solver-score net
to live/best.safetensors, and advances a processed_through watermark.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time


def run_eval(rust_bin, ckpt, games, node_budget, opening, threads, max_batch, seed) -> float:
    cmd = [
        rust_bin, "eval-solver", "--model", ckpt, "--cuda",
        "--games", str(games), "--node-budget", str(node_budget),
        "--opening", str(opening), "--max-batch", str(max_batch),
        "--seed", str(seed),
    ]
    if threads:
        cmd += ["--threads", str(threads)]
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    m = re.search(r"net_vs_solver\s+([\d.]+)", out.stdout)
    if not m:
        raise RuntimeError(f"no net_vs_solver in output:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def request_pause(live, timeout) -> bool:
    """Raise PAUSE and wait (up to timeout) for actor + learner to park. Returns
    True if both parked; False if we timed out (we proceed with the eval anyway —
    a missing/dead component shouldn't block forever)."""
    open(os.path.join(live, "PAUSE"), "w").close()
    actor_m = os.path.join(live, "actor.paused")
    learner_m = os.path.join(live, "learner.paused")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(actor_m) and os.path.exists(learner_m):
            return True
        time.sleep(1.0)
    return False


def release_pause(live) -> None:
    try:
        os.remove(os.path.join(live, "PAUSE"))
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/connect4/live")
    ap.add_argument("--rust-bin", default="target/release/connect4-engine")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--node-budget", type=int, default=200)
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--every", type=int, default=5,
                    help="evaluate the latest snapshot only once it is this many gens "
                         "past the last one scored (run less often).")
    ap.add_argument("--pause-timeout", type=float, default=600.0)
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()

    live = args.live
    snaps = os.path.join(live, "snapshots")
    best_ckpt = os.path.join(live, "best.safetensors")
    scores_path = os.path.join(live, "scores.json")
    anchor_gen = int(json.load(open(os.path.join(live, "config.json"))).get("generation", 0))

    if os.path.exists(scores_path):
        state = json.load(open(scores_path))
    else:
        state = {"anchor_gen": anchor_gen, "best_gen": anchor_gen, "best_score": -1.0,
                 "scores": {}, "history": []}
    # Skip any stale backlog on (re)start; pace new snapshots from here.
    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    pt = max([anchor_gen, int(state.get("processed_through", anchor_gen))] + existing)
    state["processed_through"] = pt
    with open(scores_path, "w") as f:
        json.dump(state, f, indent=2)
    # A fresh evaluator should never inherit a stale PAUSE flag.
    release_pause(live)
    print(f"c4 evaluator: anchor gen {anchor_gen}, processed_through {pt} (backlog skipped), "
          f"best gen {state['best_gen']} score {state['best_score']:.3f}, "
          f"{args.games} games vs solver @ node {args.node_budget}, every {args.every} gens, "
          f"{args.threads} threads", flush=True)

    while True:
        # Latest available snapshot.
        gens = []
        for d in glob.glob(os.path.join(snaps, "gen*")):
            ck = os.path.join(d, "checkpoint.safetensors")
            if os.path.exists(ck) and os.path.basename(d)[3:].isdigit():
                gens.append((int(os.path.basename(d)[3:]), ck))
        if gens:
            gen, ck = max(gens)
            if gen - state["processed_through"] >= args.every:
                parked = request_pause(live, args.pause_timeout)
                try:
                    sc = run_eval(args.rust_bin, ck, args.games, args.node_budget,
                                  args.opening, args.threads, args.max_batch, seed=gen)
                except Exception as e:
                    release_pause(live)
                    print(f"gen{gen}: eval failed: {e}", flush=True)
                    time.sleep(args.interval)
                    continue
                release_pause(live)
                promoted = sc > state["best_score"]
                if promoted:
                    shutil.copyfile(ck, best_ckpt)
                    state["best_gen"] = gen
                    state["best_score"] = sc
                state["scores"][str(gen)] = sc
                state["history"].append({"gen": gen, "score": sc, "promoted": promoted})
                state["processed_through"] = gen
                with open(scores_path, "w") as f:
                    json.dump(state, f, indent=2)
                tag = "PROMOTED" if promoted else f"best gen{state['best_gen']} ({state['best_score']:.3f})"
                pk = "" if parked else " [pause-timeout: ran with contention]"
                print(f"gen{gen} vs solver: {sc:.3f}  [{tag}]{pk}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
