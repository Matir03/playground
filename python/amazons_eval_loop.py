"""Continuous Amazons evaluator: rate each new learner snapshot against a POOL of
opponents (current best + previous best + 2 random recent) and track a pooled ELO,
promoting to `live/best.safetensors` when the pooled rating beats the current best.
Same design as go9_eval_loop.py; plays via `amazons-engine eval-match`.

Self-play ELO only — there's no external Amazons yardstick (no GNU-Go equivalent).
Needs /tmp/cuda_env.sh sourced (for the Rust binary's GPU libs).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
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


def fit_elo(opp_elos: list[float], scores: list[float]) -> float:
    """Rating R whose expected total score vs the pool matches the observed total."""
    total = sum(scores)
    lo, hi = -20000.0, 20000.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        exp = sum(1.0 / (1.0 + 10.0 ** ((e - mid) / 400.0)) for e in opp_elos)
        if exp < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def select_pool(gen, ckpt_dir, elo, best_gen, prev_best_gen):
    def ck(g):
        return os.path.join(ckpt_dir, f"gen{g:03d}", "checkpoint.safetensors")

    pool, used = [], set()
    for label, g in (("best", best_gen), ("prev", prev_best_gen)):
        if g is not None and g not in used and str(g) in elo and os.path.exists(ck(g)):
            pool.append((label, g, elo[str(g)], ck(g)))
            used.add(g)
    recent = [g for g in range(gen - 1, gen - 11, -1)
              if g not in used and str(g) in elo and os.path.exists(ck(g))]
    for g in random.sample(recent, min(2, len(recent))):
        pool.append(("rand", g, elo[str(g)], ck(g)))
        used.add(g)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/amazons_az/live")
    ap.add_argument("--rust-bin", default="target/release/amazons-engine")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--node-budget", type=int, default=400)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()

    live = args.live
    snaps = os.path.join(live, "snapshots")
    best_ckpt = os.path.join(live, "best.safetensors")
    elo_path = os.path.join(live, "elo.json")

    cfg = json.load(open(os.path.join(live, "config.json")))
    anchor_gen = int(cfg.get("generation", 0))

    if os.path.exists(elo_path):
        state = json.load(open(elo_path))
    else:
        state = {"anchor_gen": anchor_gen, "best_gen": anchor_gen, "prev_best_gen": anchor_gen,
                 "best_elo": 0.0, "elo": {str(anchor_gen): 0.0}, "history": []}
    state.setdefault("prev_best_gen", state.get("best_gen", anchor_gen))

    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    pt = max([anchor_gen, int(state.get("processed_through", anchor_gen))] + existing)
    state["processed_through"] = pt
    with open(elo_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"evaluator: anchor gen {state['anchor_gen']} = ELO 0  best gen {state['best_gen']}  "
          f"processed_through gen {pt} (backlog skipped)  games {args.games}  "
          f"node-budget {args.node_budget}", flush=True)

    while True:
        gens = []
        for d in glob.glob(os.path.join(snaps, "gen*")):
            ck = os.path.join(d, "checkpoint.safetensors")
            if os.path.exists(ck):
                gens.append((int(os.path.basename(d)[3:]), ck))
        for gen, ck in sorted(gens):
            if gen <= state["processed_through"]:
                continue
            pool = select_pool(gen, snaps, state["elo"], state["best_gen"],
                               state.get("prev_best_gen"))
            try:
                results = []
                for i, (label, og, oe, ock) in enumerate(pool):
                    s = run_match(args.rust_bin, ck, ock, args.games, args.node_budget,
                                  args.threads, args.max_batch, seed=gen * 100 + i)
                    results.append((label, og, oe, s))
            except Exception as e:
                print(f"gen{gen}: eval failed: {e}", flush=True)
                time.sleep(args.interval)
                continue
            if not results:
                elo = 0.0
            else:
                elo = fit_elo([oe for _, _, oe, _ in results], [s for _, _, _, s in results])
            vs_best = next((s for lbl, _, _, s in results if lbl == "best"), None)
            promoted = elo > state["best_elo"]
            if promoted:
                state["prev_best_gen"] = state["best_gen"]
                shutil.copyfile(ck, best_ckpt)
                state["best_gen"] = gen
                state["best_elo"] = elo
            state["elo"][str(gen)] = elo
            state["history"].append({
                "gen": gen, "elo": elo, "promoted": promoted,
                "pool": [{"label": l, "gen": g, "score": s} for l, g, _, s in results],
            })
            state["processed_through"] = gen
            with open(elo_path, "w") as f:
                json.dump(state, f, indent=2)
            pool_str = " ".join(f"{l}{g}:{s:.2f}" for l, g, _, s in results)
            tag = "PROMOTED" if promoted else f"kept best gen{state['best_gen']}"
            vb = f"{vs_best:.2f}" if vs_best is not None else "n/a"
            print(f"gen{gen}: pooled ELO {elo:+.0f}  vs_best {vb}  [{pool_str}]  [{tag}]", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
