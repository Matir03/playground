"""Continuous Amazons evaluator: rate each new learner snapshot against a gauntlet
of opponents — the current best, the previous best, a random sample of OLDER bests
(a Hall of Fame of every model ever promoted), plus a couple of random recents for
graph connectivity — and maintain a GLOBAL rating over every game ever played.

Ratings come from a Bradley-Terry (logistic) fit over the cumulative win matrix, not
a one-sided ELO of the new model against fixed opponents. That means:
  * a single lucky sweep can't pin an unbeatable rating (the old pooled-ELO bug —
    gen21's +6382 froze promotion for 35 gens);
  * playing a new model against an old one updates BOTH their ratings (the whole
    graph is re-fit each pass, warm-started so it's cheap);
  * a new gen is promoted iff its fitted rating exceeds the current best's.

The fit is regularized by a small number of virtual games against an anchor at rating
0 (`--prior`), which keeps undefeated models finite and fixes the rating gauge.

Self-play rating only — there's no external Amazons yardstick (no GNU-Go equivalent).
Needs /tmp/cuda_env.sh sourced (for the Rust binary's GPU libs).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import time

# eval-match uses candle (the snapshots are .safetensors, not ONNX); to keep it from
# contending with the TRT actor for the GPU, we SIGSTOP the actor + learner while a
# generation's matches run, then SIGCONT. Patterns identify their argv.
_PAUSE_PATS = ["amazons-engine selfplay-loop", "amazons_az_learner.py"]


def _pids(pat: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def set_paused(paused: bool) -> None:
    sig = signal.SIGSTOP if paused else signal.SIGCONT
    for pat in _PAUSE_PATS:
        for pid in _pids(pat):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def run_match(rust_bin, ckpt_a, ckpt_b, games, node_budget, threads, max_batch, seed) -> float:
    cmd = [
        rust_bin, "eval-match", "--model-a", ckpt_a, "--model-b", ckpt_b,
        "--cuda", "--games", str(games), "--node-budget", str(node_budget),
        "--threads", str(threads), "--max-batch", str(max_batch), "--seed", str(seed),
    ]
    # Timeout so a stuck match can't leave the actor/learner paused forever.
    out = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, timeout=900)
    m = re.search(r"A_score\s+([\d.]+)", out.stdout)
    if not m:
        raise RuntimeError(f"no A_score in eval-match output:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


# ---- cumulative win matrix + Bradley-Terry rating ---------------------------------

def _key(a: int, b: int) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"{lo},{hi}"


def accumulate(wins: dict, a: int, b: int, score_a: float, games: int) -> None:
    """Add `games` games between a and b (a scored `score_a` of them) to the matrix.
    Stored per unordered pair "lo,hi" as [wins_lo, wins_hi]; fractional is fine."""
    k = _key(a, b)
    wlo, whi = wins.get(k, [0.0, 0.0])
    wa, wb = score_a * games, (1.0 - score_a) * games
    if a < b:
        wlo, whi = wlo + wa, whi + wb
    else:
        wlo, whi = wlo + wb, whi + wa
    wins[k] = [wlo, whi]


def fit_bt(wins: dict, extra_players, warm: dict, prior: float = 2.0,
           iters: int = 2000, tol: float = 1e-9) -> dict:
    """Bradley-Terry MM fit (Hunter 2004) over the cumulative win matrix.
    Returns {gen: rating}. Each player also gets `prior` virtual games split 50/50
    against a phantom anchor at rating 0 — this bounds undefeated players and pins
    the gauge. Warm-started from `warm` ratings for fast convergence."""
    Wtot: dict[int, float] = {}
    opps: dict[int, dict[int, float]] = {}

    def touch(p):
        Wtot.setdefault(p, 0.0)
        opps.setdefault(p, {})

    for k, (wi, wj) in wins.items():
        i, j = (int(x) for x in k.split(","))
        touch(i); touch(j)
        g = wi + wj
        opps[i][j] = opps[i].get(j, 0.0) + g
        opps[j][i] = opps[j].get(i, 0.0) + g
        Wtot[i] += wi
        Wtot[j] += wj
    for p in extra_players:
        touch(p)

    players = list(opps.keys())
    if not players:
        return {}
    # strengths pi = 10**(rating/400); anchor (phantom) is pi = 1 (rating 0).
    pi = {p: 10.0 ** (warm.get(str(p), 0.0) / 400.0) for p in players}
    for _ in range(iters):
        max_chg = 0.0
        for p in players:
            W = Wtot[p] + prior * 0.5
            denom = prior / (pi[p] + 1.0)  # virtual games vs anchor (pi=1)
            for q, g in opps[p].items():
                denom += g / (pi[p] + pi[q])
            new = W / denom if denom > 0 else pi[p]
            if pi[p] > 0 and new > 0:
                max_chg = max(max_chg, abs(math.log(new / pi[p])))
            pi[p] = new
        if max_chg < tol:
            break
    return {p: 400.0 * math.log10(pi[p]) for p in players}


def select_pool(gen, ckpt_dir, best_gen, n_opponents):
    """Gauntlet for `gen`: always the current best (promotion anchor + graph
    connectivity to the reference) plus a random sample from ALL other generations
    with surviving checkpoints. Sampling every gen (not just promoted ones) keeps the
    cumulative rating graph dense and unbiased."""
    def ck(g):
        return os.path.join(ckpt_dir, f"gen{g:03d}", "checkpoint.safetensors")

    pool, used = [], {gen}
    if best_gen is not None and best_gen not in used and os.path.exists(ck(best_gen)):
        pool.append(("best", best_gen, ck(best_gen)))
        used.add(best_gen)
    allg = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(ckpt_dir, "gen*"))
            if os.path.basename(d)[3:].isdigit()]
    others = [g for g in allg if g not in used and os.path.exists(ck(g))]
    for g in random.sample(others, min(n_opponents, len(others))):
        pool.append(("rand", g, ck(g)))
        used.add(g)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/amazons_az/live")
    ap.add_argument("--rust-bin", default="target/release/amazons-engine")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--node-budget", type=int, default=400)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--n-opponents", type=int, default=4,
                    help="random opponents sampled from ALL gens (plus current best)")
    ap.add_argument("--prior", type=float, default=2.0)
    # Rate the single latest snapshot then exit (used by the orchestrator, which
    # spawns a fresh evaluator per generation instead of keeping one resident).
    ap.add_argument("--once", action="store_true")
    # SIGSTOP the actor (+learner) during matches. Default OFF: pausing the actor
    # wastes self-play time AND a lone small-batch candle eval-match doesn't keep the
    # GPU boosted; letting the actor keep running holds the boost and it recovers its
    # ~26k rate right after each match (the dip during a match is harmless).
    ap.add_argument("--pause-actor", action="store_true")
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
    state.setdefault("anchor_gen", anchor_gen)
    state.setdefault("prev_best_gen", state.get("best_gen", anchor_gen))

    # --- migrate older elo.json formats to the cumulative-matrix + HoF schema -------
    if "wins" not in state:
        wins: dict = {}
        for h in state.get("history", []):
            ng = h["gen"]
            for p in h.get("pool", []):
                accumulate(wins, ng, p["gen"], p["score"], args.games)
        state["wins"] = wins
    if "best_history" not in state:
        bh = [state["anchor_gen"]]
        for h in state.get("history", []):
            if h.get("promoted") and h["gen"] not in bh:
                bh.append(h["gen"])
        if state["best_gen"] not in bh:
            bh.append(state["best_gen"])
        state["best_history"] = bh

    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    if args.once:
        # The orchestrator spawns us right AFTER a new snapshot is made, to rate it.
        # Resume from the persisted processed_through (do NOT skip to max existing, or
        # we'd skip the very gen we were spawned to evaluate).
        pt = int(state.get("processed_through", anchor_gen))
        skip_note = "resume"
    else:
        # Persistent mode: skip whatever backlog accrued while eval was down.
        pt = max([anchor_gen, int(state.get("processed_through", anchor_gen))] + existing)
        skip_note = "backlog skipped"
    state["processed_through"] = pt
    with open(elo_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"evaluator: anchor gen {state['anchor_gen']}  best gen {state['best_gen']}  "
          f"HoF {state['best_history']}  processed_through gen {pt} ({skip_note})  "
          f"games {args.games}  node-budget {args.node_budget}  "
          f"gauntlet=best+{args.n_opponents}random(all gens)", flush=True)

    while True:
        gens = []
        for d in glob.glob(os.path.join(snaps, "gen*")):
            ck = os.path.join(d, "checkpoint.safetensors")
            if os.path.exists(ck):
                gens.append((int(os.path.basename(d)[3:]), ck))
        # Rate only the LATEST snapshot each pass (skip intermediates) so eval — which
        # pauses the actor — runs sparsely and can't starve training when it lags.
        rated = False
        for gen, ck in sorted(gens, reverse=True):
            if gen <= state["processed_through"]:
                continue
            pool = select_pool(gen, snaps, state["best_gen"], args.n_opponents)
            # Optionally pause the actor+learner for the matches; default off (see
            # --pause-actor). Always resume in finally if we did pause.
            if args.pause_actor:
                set_paused(True)
            try:
                results = []
                for i, (label, og, ock) in enumerate(pool):
                    s = run_match(args.rust_bin, ck, ock, args.games, args.node_budget,
                                  args.threads, args.max_batch, seed=gen * 100 + i)
                    results.append((label, og, s))
            except Exception as e:
                print(f"gen{gen}: eval failed: {e}", flush=True)
                if args.once:
                    return
                time.sleep(args.interval)
                continue
            finally:
                if args.pause_actor:
                    set_paused(False)

            # Fold this pass's games into the cumulative matrix, then re-fit ALL
            # ratings (old models' ratings move too, since they're in the graph).
            for label, og, s in results:
                accumulate(state["wins"], gen, og, s, args.games)
            elos = fit_bt(state["wins"], [gen, state["best_gen"]], state["elo"], args.prior)
            state["elo"] = {str(g): e for g, e in elos.items()}

            new_elo = elos.get(gen, 0.0)
            best_elo = elos.get(state["best_gen"], float("-inf"))
            vs_best = next((s for lbl, _, s in results if lbl == "best"), None)
            promoted = new_elo > best_elo
            if promoted:
                state["prev_best_gen"] = state["best_gen"]
                shutil.copyfile(ck, best_ckpt)
                state["best_gen"] = gen
                state["best_elo"] = new_elo
                if gen not in state["best_history"]:
                    state["best_history"].append(gen)
            state["history"].append({
                "gen": gen, "elo": new_elo, "promoted": promoted,
                "pool": [{"label": l, "gen": g, "score": s} for l, g, s in results],
            })
            state["processed_through"] = gen
            with open(elo_path, "w") as f:
                json.dump(state, f, indent=2)
            pool_str = " ".join(f"{l}{g}:{s:.2f}" for l, g, s in results)
            tag = "PROMOTED" if promoted else f"kept best gen{state['best_gen']}"
            vb = f"{vs_best:.2f}" if vs_best is not None else "n/a"
            print(f"gen{gen}: rating {new_elo:+.0f} (best gen{state['best_gen']} "
                  f"{best_elo:+.0f})  vs_best {vb}  [{pool_str}]  [{tag}]", flush=True)
            rated = True
            break  # rated the latest; resume training, recheck after the interval
        if args.once:
            if not rated:
                print("evaluator --once: no new snapshot to rate", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
