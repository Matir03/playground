#!/usr/bin/env python3
"""Print the freshly-trained generation's losses beside the recent ones.

A single generation's loss line says almost nothing on its own: these numbers
are noisy generation to generation, and several of them have floors that are
properties of the *target* rather than the net (see `energy_kl` and `decl_kl`,
which subtract the target entropy for exactly this reason). What is readable is
the direction of travel, and whether the generations that actually got promoted
look different from the ones that did not.

So the table is rows = generations, columns = losses, with promoted generations
marked and a delta against the previous row. Champions are the rows worth
comparing against, because they are the nets whose games trained everything
after them.

Reads `metrics.json` history out of `scores.json` -- the learner writes its
per-generation losses into every scores entry -- so this needs no state of its
own and can be re-run at any time.
"""

import argparse
import json
import os
import sys

# Column key -> (header, width, decimals). Ordered as displayed.
COLUMNS = [
    ("loss_value", "value", 7, 4),
    ("loss_declaration", "decl", 7, 3),
    ("decl_kl", "decl_kl", 8, 3),
    ("loss_energy", "energy", 7, 3),
    ("energy_kl", "en_kl", 7, 3),
    ("loss_aux", "aux", 7, 3),
    ("loss_score", "score", 7, 4),
    ("score_mae", "s_mae", 7, 1),
    ("loss_tb", "tb", 7, 2),
]


def _fmt(v, width, dec):
    if v is None:
        return " " * (width - 1) + "-"
    return f"{v:>{width}.{dec}f}"


def _delta(cur, prev, width):
    """Signed change from the previous row, blank when either side is missing.

    Deliberately unsigned-coloured and unlabelled as good/bad: `loss_score`
    falling is progress, but `decl_kl` rising can be too when the target got
    sharper. Reading that is the user's job; the table's job is to make it
    visible.
    """
    if cur is None or prev is None:
        return " " * width
    d = cur - prev
    if abs(d) < 1e-9:
        return " " * (width - 1) + "="
    return f"{d:>+{width}.2g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--rows", type=int, default=8,
                    help="recent generations to show (default 8)")
    ap.add_argument("--champions", type=int, default=4,
                    help="most recent promoted generations to always include, "
                         "even if they fall outside --rows (default 4)")
    args = ap.parse_args()

    path = os.path.join(args.live, "scores.json")
    if not os.path.exists(path):
        print("no scores.json yet", file=sys.stderr)
        return 0
    with open(path) as fh:
        scores = json.load(fh)
    hist = scores.get("history", [])
    if not hist:
        print("no history yet", file=sys.stderr)
        return 0

    # Rows with at least one loss recorded. An entry written by a diagnostic-only
    # pass has scores but no losses, and a blank row is worse than no row.
    trained = [r for r in hist if any(k in r for k, *_ in COLUMNS)]
    if not trained:
        print("no trained generations in history yet", file=sys.stderr)
        return 0

    recent = trained[-args.rows:]
    champs = [r for r in trained if r.get("promoted")][-args.champions:]
    shown = {r["gen"]: r for r in recent}
    for r in champs:
        shown.setdefault(r["gen"], r)
    rows = [shown[g] for g in sorted(shown)]

    best_gen = scores.get("best_gen")
    header = f"{'gen':>5} {'':2}" + "".join(
        f" {h:>{w}}" for _, h, w, _ in COLUMNS
    )
    print()
    print(header)
    print("-" * len(header))

    prev = None
    for r in rows:
        gen = r["gen"]
        # * promoted, and > for the reigning champion, which is the row every
        # later generation's training data actually came from.
        mark = "*" if r.get("promoted") else " "
        mark = ">" if gen == best_gen else mark
        line = f"{gen:>5} {mark:>2}"
        for key, _, w, dec in COLUMNS:
            line += " " + _fmt(r.get(key), w, dec)
        print(line)
        # Delta row against the previous *shown* generation. Skipped when the
        # two are not adjacent, since a jump of several generations is not a
        # step and showing it as one invites reading a trend that isn't there.
        if prev is not None and gen == prev["gen"] + 1:
            dline = f"{'':>5} {'':>2}"
            for key, _, w, _dec in COLUMNS:
                dline += " " + _delta(r.get(key), prev.get(key), w)
            print(dline)
        prev = r

    print()
    print("  * promoted    > reigning champion")
    latest = rows[-1]
    if latest.get("score_mae") is not None:
        print(f"  score head is off by {latest['score_mae']:.1f} points on average "
              f"(a turn-19 win with level material scores 10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
