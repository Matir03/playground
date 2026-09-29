"""Read a self-play tree saved by `steps-selfplay --save-trees` (the full tree,
`<run>/<batch>/nodes.bin`; format in crates/games/minions/src/tree_save.rs) and
summarise its shape.

    python/.venv/bin/python python/minions_steps_trees.py \
        data/models/minions/steps/trees/gen108-test/b000

Node ids (gid) are assigned in creation order, so a parent's gid is always
below its children's and depths come out of one ascending pass.
"""

import argparse
import os

import numpy as np

NODE_DTYPE = np.dtype(
    [
        ("gid", "<u4"),
        ("pgid", "<u4"),
        ("visits", "<u4"),
        ("value", "<f4"),
        ("prior", "<f4"),
        ("step", "u1"),
        ("kind", "u1"),
        ("flags", "u1"),
        ("comp", "i1"),
    ]
)
assert NODE_DTYPE.itemsize == 24
KINDS = ["leaf", "expanded", "terminal", "dead"]
NONE = np.uint32(0xFFFFFFFF)


def load(batch_dir):
    """(header dict, nodes indexed by gid)."""
    path = os.path.join(batch_dir, "nodes.bin")
    with open(path, "rb") as f:
        head = f.read(18)
        assert head[:4] == b"MSTT", path
        version = int(np.frombuffer(head[4:8], "<u4")[0])
        salt = int(np.frombuffer(head[8:16], "<u8")[0])
        n = int(np.frombuffer(head[16:18], "<u2")[0])
        start = f.read(n)
        off = 18 + n
    recs = np.memmap(path, NODE_DTYPE, mode="r", offset=off)
    nodes = np.empty(len(recs), NODE_DTYPE)
    nodes[recs["gid"]] = recs  # every gid below the count appears exactly once
    return {"version": version, "salt": salt, "start": start}, nodes


def depths(nodes):
    """Steps and turns (flips) from the root, per gid, by pointer jumping."""
    n = len(nodes)
    p = nodes["pgid"].astype(np.int64)
    root = p == 0xFFFFFFFF
    p[root] = np.nonzero(root)[0]
    flip = (nodes["flags"] & 1).astype(np.int64)
    step_d = (~root).astype(np.int64)
    turn_d = np.where(root, 0, flip)
    anc = p.copy()
    # After k rounds, step_d/turn_d sum over the 2^k nearest ancestors.
    while True:
        done = anc == p[anc]  # reached a root (roots point at themselves)
        if done.all():
            break
        step_d = step_d + np.where(done, 0, step_d[anc])
        turn_d = turn_d + np.where(done, 0, turn_d[anc])
        anc = np.where(done, anc, anc[anc])
    return step_d, turn_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    args = ap.parse_args()
    head, nodes = load(args.batch_dir)
    n = len(nodes)
    kind = nodes["kind"]
    visits = nodes["visits"]
    print(f"{n:,} nodes ({os.path.getsize(os.path.join(args.batch_dir, 'nodes.bin')) / 1e6:.0f} MB)")
    print("kinds:", ", ".join(f"{KINDS[k]} {np.mean(kind == k):.1%}" for k in range(4)))
    v = visits
    print(
        "visits: 0 {:.1%}, 1 {:.1%}, 2-9 {:.1%}, 10-99 {:.2%}, 100+ {:.3%}; root {:,}".format(
            np.mean(v == 0), np.mean(v == 1), np.mean((v >= 2) & (v < 10)),
            np.mean((v >= 10) & (v < 100)), np.mean(v >= 100), int(v[0]),
        )
    )
    exp = kind == 1
    nch = np.bincount(nodes["pgid"][nodes["pgid"] != NONE].astype(np.int64), minlength=n)
    b = nch[exp]
    print(f"expanded nodes {exp.sum():,}: children mean {b.mean():.1f}, median {np.median(b):.0f}, p90 {np.percentile(b, 90):.0f}, max {b.max()}")
    step_d, turn_d = depths(nodes)
    print(f"depth in steps: mean {step_d.mean():.0f}, median {np.median(step_d):.0f}, max {step_d.max()}")
    print(f"depth in turns: mean {turn_d.mean():.1f}, median {np.median(turn_d):.0f}, max {turn_d.max()}")
    vis = v > 0
    print(f"visited nodes by turn depth (from the batch start):")
    hist = np.bincount(turn_d[vis], minlength=turn_d.max() + 1)
    for t in range(0, len(hist), max(1, len(hist) // 20)):
        print(f"  turn {t:3d}: {hist[t]:>11,} visited, {np.sum(turn_d == t):>11,} all")
    term = kind == 2
    print(f"terminal nodes {term.sum():,}: first at turn depth {turn_d[term].min() if term.any() else '-'}, median {np.median(turn_d[term]) if term.any() else '-'}")
    # Where each turn starts: the flip nodes (end of a turn) on visited lines.
    flips = vis & ((nodes["flags"] & 1) == 1)
    starts = [step_d[flips & (turn_d == t)].mean() for t in range(1, 11) if np.any(flips & (turn_d == t))]
    print("steps into the tree at the end of turns 1..10:", ", ".join(f"{x:.0f}" for x in starts))

if __name__ == "__main__":
    main()
