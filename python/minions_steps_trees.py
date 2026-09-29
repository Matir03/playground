"""Read self-play trees saved by `steps-selfplay --save-trees` (formats in
crates/games/minions/src/tree_save.rs) and summarise their shape.

    python/.venv/bin/python python/minions_steps_trees.py \
        data/models/minions/steps/trees/gen111/b000 [--round K]

Format 2 (the cycle's): the whole tree at every round, `r{K}.bin`; the
summary is of one round's tree (default: the largest), depths counted from
the games' positions. Format 1 (the first runs): one `nodes.bin` with the
batch's whole tree, depths from the batch start.

`--index TREES_DIR` instead rewrites TREES_DIR/index.json (the web viewer's
list) from the runs found there; with `--keep N` it first deletes all but
the newest N cycle generations (`genN` directories; other runs are kept).
"""

import argparse
import glob
import json
import os
import re

import numpy as np

KINDS = ["leaf", "expanded", "terminal", "dead"]
NONE = 0xFFFFFFFF

# Format 2: one round's tree, in the tree's own order (parents first,
# siblings contiguous).
SNAP_DTYPE = np.dtype(
    [
        ("gid", "<u4"),
        ("first", "<u4"),
        ("visits", "<u4"),
        ("value", "<f4"),
        ("prior", "<f4"),
        ("nch", "<u2"),
        ("step", "u1"),
        ("kind", "u1"),
        ("flags", "u1"),
        ("comp", "i1"),
    ]
)
assert SNAP_DTYPE.itemsize == 26

# Format 1: the batch's whole tree, by global id.
NODE_V1_DTYPE = np.dtype(
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
assert NODE_V1_DTYPE.itemsize == 24


def load_snapshot(path):
    """A round's nodes and each node's parent index (NONE at forest roots)."""
    with open(path, "rb") as f:
        head = f.read(12)
    assert head[:4] == b"MSTS", path
    n = int(np.frombuffer(head[8:12], "<u4")[0])
    nodes = np.array(np.memmap(path, SNAP_DTYPE, mode="r", offset=12, shape=(n,)))
    parent = np.full(n, NONE, np.int64)
    has = nodes["nch"] > 0
    counts = nodes["nch"][has].astype(np.int64)
    owners = np.repeat(np.nonzero(has)[0], counts)
    # Children of each owner are first, first+1, ..., first+nch-1.
    starts = np.repeat(nodes["first"][has].astype(np.int64), counts)
    offsets = np.arange(len(owners)) - np.repeat(np.cumsum(counts) - counts, counts)
    parent[starts + offsets] = owners
    return nodes, parent


def load_v1(batch_dir):
    """Format 1: the nodes by global id and each one's parent id."""
    path = os.path.join(batch_dir, "nodes.bin")
    with open(path, "rb") as f:
        head = f.read(18)
        assert head[:4] == b"MSTT", path
        off = 18 + int(np.frombuffer(head[16:18], "<u2")[0])
    recs = np.memmap(path, NODE_V1_DTYPE, mode="r", offset=off)
    nodes = np.empty(len(recs), NODE_V1_DTYPE)
    nodes[recs["gid"]] = recs  # every gid below the count appears exactly once
    return nodes, nodes["pgid"].astype(np.int64)


def depths(parent, flip):
    """Steps and turns (flips) below each node's root, by pointer jumping."""
    n = len(parent)
    root = parent == NONE
    p = np.where(root, np.arange(n), parent)
    step_d = (~root).astype(np.int64)
    turn_d = np.where(root, 0, flip.astype(np.int64))
    anc = p.copy()
    while True:
        done = anc == p[anc]  # reached a root (roots point at themselves)
        if done.all():
            break
        step_d = step_d + np.where(done, 0, step_d[anc])
        turn_d = turn_d + np.where(done, 0, turn_d[anc])
        anc = np.where(done, anc, anc[anc])
    return step_d, turn_d


def summarise(nodes, parent, what):
    n = len(nodes)
    kind, v = nodes["kind"], nodes["visits"]
    print(f"{what}: {n:,} nodes, {np.sum(parent == NONE)} root(s)")
    print("kinds:", ", ".join(f"{KINDS[k]} {np.mean(kind == k):.1%}" for k in range(4)))
    print(
        "visits: 0 {:.1%}, 1 {:.1%}, 2-9 {:.1%}, 10-99 {:.2%}, 100+ {:.3%}; most {:,}".format(
            np.mean(v == 0), np.mean(v == 1), np.mean((v >= 2) & (v < 10)),
            np.mean((v >= 10) & (v < 100)), np.mean(v >= 100), int(v.max()),
        )
    )
    nch = np.bincount(parent[parent != NONE], minlength=n)
    b = nch[kind == 1]
    print(f"expanded nodes {len(b):,}: children mean {b.mean():.1f}, median {np.median(b):.0f}, p90 {np.percentile(b, 90):.0f}, max {b.max()}")
    step_d, turn_d = depths(parent, nodes["flags"] & 1)
    print(f"depth in steps: mean {step_d.mean():.0f}, median {np.median(step_d):.0f}, max {step_d.max()}")
    print(f"depth in turns: mean {turn_d.mean():.1f}, median {np.median(turn_d):.0f}, max {turn_d.max()}")
    vis = v > 0
    print("nodes by turn depth:")
    hist = np.bincount(turn_d[vis], minlength=turn_d.max() + 1)
    for t in range(0, len(hist), max(1, len(hist) // 20)):
        print(f"  turn {t:3d}: {hist[t]:>11,} visited, {np.sum(turn_d == t):>11,} all")
    term = kind == 2
    if term.any():
        print(f"terminal nodes {term.sum():,}: first at turn depth {turn_d[term].min()}, median {np.median(turn_d[term]):.0f}")
    # Where each turn ends: the flip nodes on visited lines.
    flips = vis & ((nodes["flags"] & 1) == 1)
    ends = [step_d[flips & (turn_d == t)].mean() for t in range(1, 11) if np.any(flips & (turn_d == t))]
    print("steps down at the end of turns 1..10:", ", ".join(f"{x:.0f}" for x in ends))


def prune(trees_dir, keep):
    """Delete all but the newest `keep` cycle generations (`genN`)."""
    import shutil

    gens = sorted(
        (int(d[3:]), d)
        for d in os.listdir(trees_dir)
        if re.fullmatch(r"gen\d+", d) and os.path.isdir(os.path.join(trees_dir, d))
    )
    for _, d in gens[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(trees_dir, d))
        print(f"deleted {d}")


def write_index(trees_dir):
    """Rewrite `trees_dir/index.json` from the runs in it: cycle generations
    (`genN`) newest first, then the rest; other runs keep a hand-set label."""
    path = os.path.join(trees_dir, "index.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = {e["run"]: e for e in json.load(f)}
    runs = []
    for run in sorted(os.listdir(trees_dir)):
        metas = sorted(glob.glob(os.path.join(trees_dir, run, "b*", "meta.json")))
        if not metas:
            continue
        cycle = re.fullmatch(r"gen\d+", run) is not None
        entry = {} if cycle else dict(old.get(run, {}))
        entry["run"] = run
        entry["batches"] = [os.path.basename(os.path.dirname(m)) for m in metas]
        if "label" not in entry:
            ms = [json.load(open(m)) for m in metas]
            if ms[0].get("version") == 2:
                rounds = sum(sum(1 for r in m["rounds"] if r["games"]) for m in ms)
                nodes = sum(m["distinct"] for m in ms)
            else:
                rounds = sum(len(m["moves"]) for m in ms)
                nodes = sum(m["nodes"] for m in ms)
            entry["label"] = (
                f"{run} self-play: {len(ms)} tree(s), {ms[0]['games']} games, "
                f"{rounds} rounds, {nodes / 1e6:.0f}M nodes"
            )
        runs.append(entry)
    gen = lambda e: int(e["run"][3:]) if re.fullmatch(r"gen\d+", e["run"]) else -1
    runs.sort(key=lambda e: (gen(e) < 0, -gen(e), e["run"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(runs, f, indent=2)
    os.replace(tmp, path)
    print(f"{path}: {len(runs)} runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", nargs="?")
    ap.add_argument("--round", type=int, help="format 2: the round to summarise (default: the largest tree)")
    ap.add_argument("--index", metavar="TREES_DIR")
    ap.add_argument("--keep", type=int, default=0, help="with --index: keep only the newest N cycle generations (0 = all)")
    args = ap.parse_args()
    if args.index:
        if args.keep:
            prune(args.index, args.keep)
        write_index(args.index)
        return
    meta = json.load(open(os.path.join(args.batch_dir, "meta.json")))
    if meta.get("version") == 2:
        rounds = [r for r in meta["rounds"] if r["games"]]
        k = args.round if args.round is not None else max(range(len(rounds)), key=lambda i: rounds[i]["nodes"])
        nodes, parent = load_snapshot(os.path.join(args.batch_dir, rounds[k]["file"]))
        summarise(nodes, parent, f"round {k} of {len(rounds)} ({meta['distinct']:,} distinct nodes in the batch)")
    else:
        nodes, parent = load_v1(args.batch_dir)
        summarise(nodes, parent, "whole batch (format 1)")


if __name__ == "__main__":
    main()
