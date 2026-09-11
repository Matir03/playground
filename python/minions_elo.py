#!/usr/bin/env python3
"""Bradley-Terry ratings over the run's recorded matches.

Why BT rather than incremental Elo: incremental Elo depends on the order games
were played and on a K-factor, so the same results fed in a different order give
different numbers, and early ratings stay wrong forever because nothing ever
revisits them. BT fits every rating jointly to the whole match record, so adding
one match today corrects the estimate for a champion from two hundred
generations ago if that is what the evidence says.

The model is the standard one: player `i` beats `j` with probability

    P(i > j) = 1 / (1 + 10 ** ((r_j - r_i) / 400))

fitted by maximum likelihood. Draws count as half a win to each side, which is
the usual treatment and adequate here -- Minions draws only on an exact
net-worth tie at the turn cap and they are rare.

**Greedy is the anchor.** It is the one opponent that never changes, so pinning
it at 0 makes the whole history commensurable: a champion's rating means "this
far above the fixed heuristic", and it cannot drift just because the self-play
ladder as a whole moved. Anchoring on a *generation* instead would make every
rating relative to a net that is itself only defined by other ratings.
"""

import argparse
import json
import math
import os
import sys

GREEDY = "greedy"
# Rating points per factor-of-10 odds, i.e. the usual Elo scale.
SCALE = 400.0 / math.log(10.0)


def load_matches(path):
    """Read the match log. One JSON object per line, appended as games happen."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a crash mid-append. Skip it rather than
                # refusing to report any ratings at all.
                continue
    return out


def fit(matches, iters=10000, tol=1e-9):
    """Minorization-maximization fit of BT strengths.

    MM rather than gradient descent: it has no step size to tune, every update
    is a closed form, and it is monotone in the likelihood, so it cannot
    diverge on a sparse or badly conditioned match graph.

    Returns `{player: elo}` with greedy pinned at 0, or `{}` if there is
    nothing to fit.
    """
    # wins[i][j] = fractional games i won against j.
    wins, opponents = {}, {}
    for m in matches:
        a, b, n = m["a"], m["b"], m["games"]
        if n <= 0 or a == b:
            continue
        sa = m["score_a"] * n           # fractional wins for a
        sb = n - sa
        wins.setdefault(a, {}).setdefault(b, 0.0)
        wins.setdefault(b, {}).setdefault(a, 0.0)
        wins[a][b] += sa
        wins[b][a] += sb
        opponents.setdefault(a, set()).add(b)
        opponents.setdefault(b, set()).add(a)

    players = sorted(wins)
    if not players:
        return {}

    # Keep only the component containing greedy: ratings in a disconnected
    # component are unidentifiable against the anchor, and reporting them would
    # invent a comparison no game supports.
    if GREEDY in opponents:
        seen, stack = {GREEDY}, [GREEDY]
        while stack:
            cur = stack.pop()
            for nxt in opponents.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        players = [p for p in players if p in seen]
    if not players:
        return {}

    # A player who never lost (or never won) has an unbounded MLE. A symmetric
    # half-game prior against a phantom opponent at the field's geometric mean
    # bounds it, at the cost of shrinking that player's rating toward the field.
    #
    # Applied *only* to the players that need it. Giving it to everyone looks
    # harmless -- half a game against hundreds of real ones -- but it is a pull
    # toward the field mean whose strength grows with distance from that mean,
    # so it compresses the whole scale, and the compression *moves* whenever the
    # field does. Three rejected candidates entering the record shifted every
    # champion down 17-36 points that way, on no evidence about those champions
    # at all. Ratings that drift when unrelated players are added cannot be a
    # promotion criterion, and they now are one.
    prior = 0.5
    won = {p: sum(wins[p].values()) for p in players}
    lost = {p: sum(wins[q].get(p, 0.0) for q in players) for p in players}
    reg = {p: prior if won[p] <= 1e-9 or lost[p] <= 1e-9 else 0.0 for p in players}
    total = {p: won[p] + reg[p] for p in players}

    strength = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            denom = 0.0
            for q, _ in wins[p].items():
                if q not in strength:
                    continue
                n_pq = wins[p].get(q, 0.0) + wins[q].get(p, 0.0)
                denom += n_pq / (strength[p] + strength[q])
            # The phantom opponent sits at the geometric mean, which is 1.0 by
            # the normalisation below. Zero for players with a bounded MLE.
            if reg[p]:
                denom += 2.0 * reg[p] / (strength[p] + 1.0)
            new[p] = total[p] / denom if denom > 0 else strength[p]
        # Normalise to geometric mean 1 so the iteration cannot drift as a
        # whole; the anchor is applied after convergence.
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in new.values()) / len(new))
        new = {p: v / gm for p, v in new.items()}
        delta = max(abs(math.log(max(new[p], 1e-12)) - math.log(max(strength[p], 1e-12)))
                    for p in players)
        strength = new
        if delta < tol:
            break

    elo = {p: SCALE * math.log(max(s, 1e-12)) for p, s in strength.items()}
    anchor = elo.get(GREEDY, 0.0)
    return {p: r - anchor for p, r in elo.items()}


def games_played(matches):
    counts = {}
    for m in matches:
        counts[m["a"]] = counts.get(m["a"], 0) + m["games"]
        counts[m["b"]] = counts.get(m["b"], 0) + m["games"]
    return counts


def report(matches, top=12):
    ratings = fit(matches)
    if not ratings:
        return "no rated matches yet"
    counts = games_played(matches)

    def sort_key(p):
        # Generations numerically, greedy last.
        return (1, 0) if p == GREEDY else (0, -int(p.split("gen")[-1]))

    rows = sorted(ratings, key=sort_key)[:top]
    out = [f"{'player':>10} {'elo':>8} {'games':>6}", "-" * 26]
    for p in rows:
        out.append(f"{p:>10} {ratings[p]:>8.0f} {counts.get(p, 0):>6}")
    if GREEDY in ratings and GREEDY not in rows:
        out.append(f"{GREEDY:>10} {ratings[GREEDY]:>8.0f} {counts.get(GREEDY, 0):>6}")
    out.append("")
    out.append("  greedy anchored at 0; BT fit over the whole match record")
    return "\n".join(out)


def backfill(live):
    """Reconstruct the match log from `scores.json`.

    Every generation already played 40 games against the reigning champion, and
    every promotion played 30 against greedy -- roughly 300 matches of real
    pairwise evidence that predate the match log. Throwing that away and
    starting the ratings from the next promotion would leave the whole run's
    history unrated for no reason.

    The champion each candidate faced is not stored per entry, so it is
    recovered by replaying the history in order: the champion is whoever was
    promoted most recently before that generation.

    Only the aggregate score is recoverable, not the win/loss/draw split -- a
    win rate of 0.525 over 40 games could be 21-19 or 20-18-2. BT consumes
    fractional wins, so this costs nothing.
    """
    path = os.path.join(live, "scores.json")
    if not os.path.exists(path):
        return []
    hist = json.load(open(path)).get("history", [])
    out, champion = [], None
    for r in hist:
        gen = r.get("gen")
        if gen is None:
            continue
        if r.get("vs_champion") is not None and champion is not None:
            out.append({
                # Pre-rating-gate generations played 40; entries written since
                # record what they actually played.
                "a": f"gen{gen}", "b": f"gen{champion}",
                "games": int(r.get("gate_games", 40)),
                "score_a": float(r["vs_champion"]), "kind": "gate",
            })
        if r.get("vs_greedy") is not None:
            # Recorded against the *champion* of the moment, which is this
            # generation when it was just promoted.
            who = r.get("greedy_gen", gen)
            out.append({
                "a": f"gen{who}", "b": GREEDY, "games": 30,
                "score_a": float(r["vs_greedy"]), "kind": "greedy",
            })
        if r.get("promoted"):
            champion = gen
    return out


def append(live, rows):
    """Append match rows to the log, atomically enough for an appending writer."""
    if not rows:
        return
    path = os.path.join(live, "matches.jsonl")
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_ratings(live, ratings, counts, n_matches):
    """Write `elo.json` atomically."""
    path = os.path.join(live, "elo.json")
    tmp = path + ".tmp"
    json.dump({"ratings": {p: round(r, 1) for p, r in ratings.items()},
               "games": counts, "matches": n_matches},
              open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--write", action="store_true",
                    help="also write <live>/elo.json")
    ap.add_argument("--backfill", action="store_true",
                    help="rebuild matches.jsonl from scores.json, discarding "
                         "whatever is there. DESTRUCTIVE since the rating "
                         "gate landed: only gate and greedy rows are "
                         "recoverable from scores.json, so every match against "
                         "a non-reigning champion -- now most of the record -- "
                         "is lost. The displaced file is kept as .bak.")
    args = ap.parse_args()

    if args.backfill:
        rows = backfill(args.live)
        path = os.path.join(args.live, "matches.jsonl")
        if os.path.exists(path):
            os.replace(path, path + ".bak")
        append(args.live, rows)
        print(f"backfilled {len(rows)} matches from scores.json", file=sys.stderr)

    matches = load_matches(os.path.join(args.live, "matches.jsonl"))
    if not matches:
        print("no matches recorded yet", file=sys.stderr)
        return 0
    print()
    print(report(matches, args.top))
    if args.write:
        write_ratings(args.live, fit(matches), games_played(matches), len(matches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
