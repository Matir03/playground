"""Bradley-Terry ratings for steps mode, refitted from the whole of cycle.log
(see minions_elo.fit). Players: gN = gen N greedy,
gN+s = gen N best-first search at 2000 expansions, old / old+s = the gen-501
sampled-AZ value head as step evaluator. Anchor: old+s = 0."""
import re, sys, json
sys.path.insert(0, __import__("os").path.dirname(__file__))
import minions_elo as E
rows, gen = {}, None
for line in open(sys.argv[2] if len(sys.argv) > 2 else "data/models/minions/steps/cycle.log"):
    m = re.match(r"=== gen (\d+): matches", line)
    if m: gen = int(m.group(1)); continue
    m = re.search(r"A wins (\d+), B wins (\d+), draws (\d+).*\[(.*)\]\s*$", line)
    if not m or gen is None: continue
    w, l, d, lab = int(m[1]), int(m[2]), int(m[3]), m[4]
    s = lab.startswith("search")
    a = f"g{gen}+s" if s else f"g{gen}"
    if "greedy, same net" in lab: b = f"g{gen}"
    elif "old" in lab: b = "old+s" if s else "old"
    else:
        o = int(re.search(r"gen(\d+)", lab)[1]); b = f"g{o}+s" if s else f"g{o}"
    n = w + l + d
    rows[(gen, lab)] = {"a": a, "b": b, "games": n, "score_a": (w + d / 2) / n}
ms = list(rows.values())
r = E.fit(ms, include_greedy=True)
base = r["old+s"]; r = {k: v - base for k, v in r.items()}
json.dump({"ratings": r, "matches": ms}, open(sys.argv[1], "w"))
gens = sorted({int(re.match(r"g(\d+)", k)[1]) for k in r if k[1].isdigit()})
print(f"{len(ms)} matches, {len(r)} players; old greedy {r.get('old', float('nan')):+.0f}")
for g in gens:
    print(f"gen{g:3d}  search {r.get(f'g{g}+s', float('nan')):+6.0f}   greedy {r.get(f'g{g}', float('nan')):+6.0f}")
