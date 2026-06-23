"""BT-ELO ladder + promotion for the Connect 4 LGS generations.

Each generation snapshots the live checkpoint into `gens/gen_NNNNNN.safetensors`,
plays head-to-head matches (via the engine's `lgs-match`) against the current
champion, a few random previous generations, and the perfect solver, then refits
Bradley-Terry ELOs over *all* players (gens + solver) from the full match history
and promotes the new generation to champion if its ELO exceeds the champion's.

Ladder state (json):
  players : [name, ...]              # "solver" + "gen_NNNNNN"
  ckpt    : {name: path|"solver"}
  matches : [[a, b, wa, d, wb], ...] # a's wins / draws / b's wins
  elo     : {name: rating}           # anchored so solver = 0
  best    : name | null              # champion (max-ELO generation)
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess


def _run_match(engine, a, b, games, sims, cuda, seed):
    cmd = [engine, "lgs-match", "--a", a, "--b", b,
           "--games", str(games), "--sims", str(sims), "--seed", str(seed)]
    if cuda:
        cmd.append("--cuda")
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("MATCH"):
            _, wa, d, wb = line.split()
            return int(wa), int(d), int(wb)
    raise RuntimeError(f"no MATCH line (rc={out.returncode}):\n{out.stdout}\n{out.stderr}")


def load_ladder(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {"players": [], "ckpt": {"solver": "solver"}, "matches": [], "elo": {}, "best": None}


def save_ladder(path, L):
    json.dump(L, open(path, "w"), indent=1)


def fit_bt(players, matches, anchor="solver", iters=2000):
    """Bradley-Terry MLE via MM iteration; draws count as half a win to each.
    Sparse over the pairs that actually played. Returns {name: elo}, solver=0."""
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = [0.0] * n
    pair = [dict() for _ in range(n)]  # i -> {j: games}
    for a, b, wa, d, wb in matches:
        if a not in idx or b not in idx:
            continue
        ia, ib = idx[a], idx[b]
        wins[ia] += wa + 0.5 * d
        wins[ib] += wb + 0.5 * d
        g = wa + wb + d
        pair[ia][ib] = pair[ia].get(ib, 0.0) + g
        pair[ib][ia] = pair[ib].get(ia, 0.0) + g
    gamma = [1.0] * n
    has_games = [len(pair[i]) > 0 for i in range(n)]
    for _ in range(iters):
        new = list(gamma)
        for i in range(n):
            if not has_games[i] or wins[i] <= 0:
                continue
            denom = sum(g / (gamma[i] + gamma[j]) for j, g in pair[i].items())
            if denom > 0:
                new[i] = wins[i] / denom
        active = [new[i] for i in range(n) if has_games[i] and new[i] > 0]
        if active:
            gm = math.exp(sum(math.log(x) for x in active) / len(active))
            for i in range(n):
                if has_games[i]:
                    new[i] /= gm
        delta = max((abs(new[i] - gamma[i]) for i in range(n) if has_games[i]), default=0.0)
        gamma = new
        if delta < 1e-9:
            break
    elo = {}
    for p in players:
        gi = gamma[idx[p]]
        elo[p] = 400.0 * math.log10(gi) if (has_games[idx[p]] and gi > 0) else float("nan")
    if anchor in elo and not math.isnan(elo[anchor]):
        shift = elo[anchor]
        elo = {p: (v - shift if not math.isnan(v) else v) for p, v in elo.items()}
    return elo


def step(engine, live_ckpt, gens_dir, ladder_path, gen, cuda,
         games=40, sims=64, n_random=3, seed=0):
    """Archive the live checkpoint, run matches, refit ELO, maybe promote.
    Returns (ladder, name, promoted, elo). Raises on match failure."""
    os.makedirs(gens_dir, exist_ok=True)
    # Archived checkpoints need their arch so the engine loads the right shapes.
    # Each gen gets a per-checkpoint sidecar (gen_NNNNNN.config.json) — the engine
    # reads that first — so a single gens/ dir can hold mixed archs (e.g. legacy
    # vs grown value heads). A shared config.json is kept as a fallback.
    src_cfg = os.path.join(os.path.dirname(os.path.abspath(live_ckpt)), "config.json")
    dst_cfg = os.path.join(gens_dir, "config.json")
    if os.path.exists(src_cfg) and not os.path.exists(dst_cfg):
        shutil.copy(src_cfg, dst_cfg)
    L = load_ladder(ladder_path)
    name = f"gen_{gen:06d}"
    arch = os.path.join(gens_dir, name + ".safetensors")
    shutil.copy(live_ckpt, arch)
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, os.path.join(gens_dir, name + ".config.json"))
    L["ckpt"][name] = arch
    if "solver" not in L["players"]:
        L["players"].append("solver")
    if name not in L["players"]:
        L["players"].append(name)

    prev_gens = [p for p in L["players"] if p.startswith("gen_") and p != name]
    opps = {"solver"}
    if L["best"]:
        opps.add(L["best"])
    if prev_gens:
        opps.update(random.sample(prev_gens, min(n_random, len(prev_gens))))
    for i, opp in enumerate(sorted(opps)):
        wa, d, wb = _run_match(engine, arch, L["ckpt"][opp], games, sims, cuda, seed + i)
        L["matches"].append([name, opp, wa, d, wb])

    elo = fit_bt(L["players"], L["matches"])
    L["elo"] = elo
    promoted = False
    cur = elo.get(name, float("nan"))
    champ = elo.get(L["best"], float("-inf")) if L["best"] else float("-inf")
    if L["best"] is None or (not math.isnan(cur) and cur > champ):
        L["best"] = name
        promoted = True
    save_ladder(ladder_path, L)
    return L, name, promoted, elo


def elo_table(L, top=8):
    items = [(p, v) for p, v in L["elo"].items() if not math.isnan(v)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    head = items[:top]
    return "  ".join(f"{p}={v:.0f}" for p, v in head)
