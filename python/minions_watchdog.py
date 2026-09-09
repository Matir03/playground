"""Degeneracy watchdog for the Minions run — read the replay buffer, not the losses.

`MAX_TURNS` is 60 side-turns and reaching it is adjudicated on **net worth**
(gold plus every owned unit at cost, on the board and in reinforcements). Under
that rule buying is par, so hoarding earns nothing — but self-play found the
hoarding equilibrium under the *old* raw-gold rule within three generations, and
the failure was invisible in the loss curves, so the metric stays.

What it now watches for is a policy that banks instead of developing: a high
`gold share` of net worth with few units on the board. That is no longer
rewarded by the scoring, but it would still signal that the search is not finding
anything worth buying.

The signature to watch for is money climbing while the board stays empty, i.e. a
high mean gold with low units-on-board and a *negative* money/units correlation.
A healthy run spends: gold stays low and correlates positively with units, since
both grow with a strong position.

Run after each generation::

    PYTHONPATH=python/src:python python python/minions_watchdog.py \\
        --live data/models/minions/live
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "python")

from playground.model_server.minions_net import IN_PLANES, N  # noqa: E402
from minions_az_learner import Buffer  # noqa: E402

# Plane layout, from `Minions::to_features` in crates/games/minions/src/game.rs:
# 0..5 own units (NUM_UNITS=6), 6..11 enemy units, 12 graveyard,
# 13 own money, 14 enemy money, 15..19 own reinforcements, 20..24 enemy.
#
# Dropped since the 31-plane version: `side` (the sampler canonicalises to the
# mover's frame) and damage/moved/attacked/exhausted/phase (the net only sees
# turn boundaries, where `end_turn` has cleared all of them).
P_OWN_UNITS = slice(0, 6)
P_ENEMY_UNITS = slice(6, 12)
P_OWN_MONEY = 13
P_OWN_REINF = slice(15, 20)

# Unit index order (Unit::from_index): Necromancer, Zombie, Skeleton, Serpent,
# Warg, Ghost. Costs must match `UnitStats::cost` in units.rs — `adjudicate`
# values units at cost, so this does too.
UNIT_COST = np.array([0, 2, 4, 4, 4, 3], dtype=np.float32)
# BUYABLE order: Zombie, Skeleton, Serpent, Warg, Ghost.
REINF_COST = np.array([2, 4, 4, 4, 3], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Minions degeneracy watchdog")
    ap.add_argument("--live", required=True)
    ap.add_argument("--hoard-gold", type=float, default=25.0,
                    help="mean gold above which hoarding is suspected")
    ap.add_argument("--idle-share", type=float, default=0.5,
                    help="share of net worth held as gold above which the policy "
                         "is judged to be refusing to build")
    ap.add_argument("--hoard-share", type=float, default=0.45,
                    help="idle share a *rising* trend must also exceed to warn; "
                         "a fresh run legitimately climbs to ~0.36")
    args = ap.parse_args()

    buf = Buffer(f"{args.live}/buffer")
    if not buf.refresh():
        print("no examples in buffer")
        return 1
    d = buf.merged()
    pl = d["planes"].reshape(-1, IN_PLANES, N, N).astype(np.float32)

    # Money and reinforcement planes are constant across cells, so any cell reads
    # the value.
    gold = pl[:, P_OWN_MONEY, 0, 0]
    units = pl[:, P_OWN_UNITS].sum(axis=(1, 2, 3))
    # Net worth, mirroring `net_worth` in game.rs: gold + units at cost, on the
    # board and in reinforcements. This must track the rule; a watchdog measuring
    # a different quantity than the one being optimised is worse than none.
    board_val = (pl[:, P_OWN_UNITS].sum(axis=(2, 3)) * UNIT_COST).sum(axis=1)
    reinf_val = (pl[:, P_OWN_REINF, 0, 0] * REINF_COST).sum(axis=1)
    worth = gold + board_val + reinf_val
    # With adjudication on net worth, buying is par, so hoarding is no longer
    # *rewarded*. What still matters is whether the policy develops at all: the
    # share of net worth sitting idle as gold.
    idle = np.divide(gold, np.maximum(worth, 1e-6))
    corr = float(np.corrcoef(gold, units)[0, 1]) if gold.std() > 0 else 0.0
    v = d["value"]

    print(f"examples          {len(gold)}")
    print(f"net worth         mean {worth.mean():.1f}  median {np.median(worth):.0f}")
    print(f"own gold          mean {gold.mean():.1f}  median {np.median(gold):.0f}  "
          f"p90 {np.percentile(gold, 90):.0f}  max {gold.max():.0f}")
    print(f"gold share        mean {idle.mean():.2f}  (fraction of net worth held idle)")
    print(f"own units         mean {units.mean():.2f}  median {np.median(units):.0f}")
    print(f"corr(gold, units) {corr:+.3f}")
    print(f"outcomes          win {(v > 0).mean():.2f}  loss {(v < 0).mean():.2f}  "
          f"draw {(v == 0).mean():.2f}")
    # Mean terminal score over self-play positions, from each position's mover.
    # Near 0 by construction -- self-play is symmetric, so one seat's score is
    # the other's negated -- which is exactly why the useful reading is its
    # *spread*, not its mean: it says how decisively these games are ending.
    sc = d.get("score")
    has_sc = d.get("has_score")
    if sc is not None and has_sc is not None and has_sc.any():
        sc = sc[has_sc.astype(bool)]
        print(f"game score        mean {sc.mean():+.1f}  |mean| {np.abs(sc).mean():.1f}  "
              f"p10 {np.percentile(sc, 10):+.0f}  p90 {np.percentile(sc, 90):+.0f}")

    # Append to a per-generation history so the *trend* is visible. A single
    # generation cannot distinguish "rich because winning" from "rich because
    # hoarding"; the direction of travel can.
    hist_path = os.path.join(args.live, "watchdog.jsonl")
    gen = 0
    cfg = os.path.join(args.live, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as fh:
            gen = int(json.load(fh).get("generation", 0))
    row = {"generation": gen, "gold_mean": float(gold.mean()),
           "gold_p90": float(np.percentile(gold, 90)), "units_mean": float(units.mean()),
           "corr": corr, "net_worth_mean": float(worth.mean()),
           "gold_share": float(idle.mean())}
    if sc is not None and has_sc is not None and has_sc.any():
        row["score_absmean"] = round(float(np.abs(sc).mean()), 2)
        row["score_p90"] = round(float(np.percentile(sc, 90)), 1)
    # One row per generation. The cycle runs this after every generation and it
    # also gets run by hand, so appending unconditionally makes the trend — the
    # thing the file exists for — hard to read.
    prior = []
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            prior = [json.loads(x) for x in fh if x.strip()]
    if not prior or prior[-1].get("generation") != gen:
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    # Under net-worth adjudication the gold/units *correlation* is no longer the
    # signal — banking and building score the same, so they stop being opposed
    # and the correlation drifts to zero even while the policy refuses to build.
    # What matters is the share of wealth left idle, and whether anything
    # decisive ever happens: with small armies nobody can muster the 7 damage a
    # necromancer's defence needs, so every game runs to the cap and the value
    # target carries almost no information about material.
    decisive = float((d["necro_safety"].max(axis=(1, 2)) > 0).mean())
    print(f"decisive          {100 * decisive:.1f}% of positions have a "
          f"necromancer death within {d['necro_safety'].shape[2]} turns")
    if idle.mean() > args.idle_share and decisive == 0.0:
        print(f"\nSTALLED: {100 * idle.mean():.0f}% of net worth sits idle and no game is\n"
              "decided by combat. Buying is par under net worth, but gold is risk-free\n"
              "while a unit can die — so a policy that loses its trades correctly\n"
              "refuses to build, and then never learns to trade. Consider valuing units\n"
              "slightly above cost, or discounting banked gold, to break the tie.")
        return 1

    if gold.mean() > args.hoard_gold and corr < 0.0:
        print("\nDEGENERATE: gold is accumulating and does not convert into units.\n"
              "Adjudication already scores net worth, so hoarding wins nothing here —\n"
              "buying is par by construction. If this fires, the cause is elsewhere:\n"
              "check that spawns are actually reachable (spawn_cells / realizer\n"
              "fidelity), and that the search is proposing them at all.")
        return 1

    # Warn well before the hard threshold: by the time both fire the run has
    # already spent generations learning to hoard. A negative correlation alone
    # means gold has stopped converting into board presence, which is the first
    # half of the failure and the point at which to start watching closely.
    prev = []
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            prev = [json.loads(x) for x in fh if x.strip()]
    # Fire on a *trend*, not a level.
    #
    # A correlation near zero is the expected equilibrium here, not a warning:
    # under net-worth adjudication banking and building score the same, so the
    # two quantities decouple once the policy stops treating them as opposed.
    # Keying on `corr < 0` therefore fires on noise around zero — it did, on a
    # run whose gold had been flat at 13.2 for three generations. What actually
    # matters is whether the idle share is *still climbing*.
    # A rising idle share is not by itself a problem: a run started from random
    # weights climbs from ~0.28 to its normal operating point over the first few
    # generations, and the increments decay as it gets there (measured
    # 2026-08-28: +0.059, +0.015, +0.010, settling near 0.36 — exactly where the
    # previous run sat stably for dozens of generations). Firing on the trend
    # alone meant warning through every healthy warm-up.
    #
    # So require the trend *and* a level meaningfully above that operating point,
    # *and* that the climb has not already flattened.
    # The level gate is what distinguishes the two cases. Requiring the climb to
    # still be *accelerating* was tempting and wrong: it would also have silenced
    # a slow creep from 0.45 to 0.80, which is the failure this exists to catch.
    recent = [r for r in prev[-4:] if "gold_share" in r]
    climbing = (
        len(recent) >= 3
        and recent[-1]["gold_share"] > recent[0]["gold_share"] + 0.03
        and recent[-1]["units_mean"] < recent[0]["units_mean"]
        and recent[-1]["gold_share"] > args.hoard_share
    )
    if idle.mean() > args.idle_share * 0.7 and climbing:
        print(f"\nWARNING: trending toward the gold-hoarding equilibrium.")
        for r in prev[-4:]:
            print(f"  gen {r['generation']}: gold {r['gold_mean']:5.1f} "
                  f"(p90 {r['gold_p90']:5.1f})  units {r['units_mean']:5.2f}  "
                  f"corr {r['corr']:+.3f}")
        print("Gold is decoupling from board presence. Under net-worth adjudication\n"
              "that is not rewarded, so persistent banking points at the policy not\n"
              "finding spawns worth making rather than at the scoring rule.")
        return 0
    print("\nhealthy: gold is being spent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
