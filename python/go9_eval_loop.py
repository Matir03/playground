"""Continuous Go9 evaluator: rate each new learner snapshot vs the current best
and track ELO. When a new `live/snapshots/genNNN` appears, play it against the
reigning best via `go9-engine eval-match`; convert the score to a relative ELO
(`best_elo + 400*log10(s/(1-s))`); if it scores over the promotion threshold,
promote it to `live/best.safetensors` (what we'd serve). ELO history is written
to `live/elo.json`.

Anchor: the seed gen (gen 12) = ELO 0. Single GPU, so this contends with the
actors + learner; it runs only when a new snapshot lands. Needs /tmp/cuda_env.sh
sourced (for the Rust binary's GPU libs).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import time


def run_match(rust_bin, ckpt_a, ckpt_b, games, node_budget, threads, max_batch, seed) -> float:
    cmd = [
        rust_bin, "eval-match", "--model-a", ckpt_a, "--model-b", ckpt_b,
        "--cuda", "--games", str(games), "--node-budget", str(node_budget),
        "--threads", str(threads), "--max-batch", str(max_batch), "--seed", str(seed),
    ]
    out = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    m = re.search(r"A_score\s+([\d.]+)", out.stdout)
    if not m:
        raise RuntimeError(f"no A_score in eval-match output:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/go9/live")
    ap.add_argument("--rust-bin", default="target/release/go9-engine")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--node-budget", type=int, default=128)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--promote", type=float, default=0.55, help="score to replace best")
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()

    live = args.live
    snaps = os.path.join(live, "snapshots")
    best_ckpt = os.path.join(live, "best.safetensors")
    elo_path = os.path.join(live, "elo.json")

    cfg = json.load(open(os.path.join(live, "config.json")))
    anchor_gen = int(cfg.get("generation", 0))

    # Resume ELO state if present.
    if os.path.exists(elo_path):
        state = json.load(open(elo_path))
    else:
        state = {"anchor_gen": anchor_gen, "best_gen": anchor_gen, "best_elo": 0.0,
                 "elo": {str(anchor_gen): 0.0}, "history": []}
    seen = {int(h["gen"]) for h in state["history"]}
    print(f"evaluator: anchor gen {state['anchor_gen']} = ELO 0  best gen {state['best_gen']}  "
          f"games {args.games}  node-budget {args.node_budget}", flush=True)

    while True:
        gens = []
        for d in glob.glob(os.path.join(snaps, "gen*")):
            ck = os.path.join(d, "checkpoint.safetensors")
            if os.path.exists(ck):
                gens.append((int(os.path.basename(d)[3:]), ck))
        for gen, ck in sorted(gens):
            if gen in seen:
                continue
            try:
                s = run_match(args.rust_bin, ck, best_ckpt, args.games, args.node_budget,
                              args.threads, args.max_batch, seed=gen)
            except Exception as e:
                print(f"gen{gen}: eval failed: {e}", flush=True)
                time.sleep(args.interval)
                continue
            sc = min(0.99, max(0.01, s))
            elo = state["best_elo"] + 400.0 * math.log10(sc / (1.0 - sc))
            promoted = s > args.promote
            if promoted:
                shutil.copyfile(ck, best_ckpt)
                state["best_gen"] = gen
                state["best_elo"] = elo
            state["elo"][str(gen)] = elo
            state["history"].append({"gen": gen, "vs_best": state["best_gen"],
                                     "score": s, "elo": elo, "promoted": promoted})
            with open(elo_path, "w") as f:
                json.dump(state, f, indent=2)
            seen.add(gen)
            tag = "PROMOTED" if promoted else f"kept best gen{state['best_gen']}"
            print(f"gen{gen} vs best: score {s:.3f}  ELO {elo:+.0f}  [{tag}]", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
