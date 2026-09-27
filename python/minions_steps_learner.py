"""Value nets for the best-first step search (docs/minions-steps.md).

Reads the `MSTP` record chunks written by `minions-engine steps-selfplay`
(layout in crates/games/minions/src/stepsp.rs), trains a value net on the
backed-up search values, and exports it to ONNX with input `x` [B, 849] --
the 833 sparse 0/1 features (`plane * 49 + cell`) followed by 16 scalars,
exactly `bfs::dense_features` -- and output `v` [B], a win probability.

Architectures (`--arch`):
  mlp:<h1>,<h2>,...   dense MLP; the first layer is the NNUE accumulator
  res:<ch>x<blocks>   conv ResNet over the 17 feature planes + 16 broadcast scalars
  fcres:<ch>x<blocks> a fully connected first layer 849 -> 49*ch, then the ResNet

Example:
  python minions_steps_learner.py --data data/models/minions/steps/data/gen0 \
      --arch res:64x6 --out data/models/minions/steps/nets/res64x6
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 7
CELLS = 49
SPARSE_PLANES = 17
SPARSE_DIM = SPARSE_PLANES * CELLS  # 833
N_SCALARS = 16
DENSE_DIM = SPARSE_DIM + N_SCALARS  # 849
MAX_ACTIVE = 96
REC_BYTES = 272

REC_DTYPE = np.dtype(
    [
        ("n_active", "u1"),
        ("stage", "u1"),
        ("visits", "<u2"),
        ("target", "<f4"),
        ("prior", "<f4"),
        ("turn", "<u2"),
        ("pad", "<u2"),
        ("active", "<u2", (MAX_ACTIVE,)),
        ("scalars", "<f4", (N_SCALARS,)),
    ]
)
assert REC_DTYPE.itemsize == REC_BYTES


def load_chunks(dirs, max_records=None, holdout_every=0):
    """Records from every chunk; with `holdout_every` > 0, every that-many-th
    file goes to a separate holdout set (whole files, so held-out positions
    come from other games than the training ones)."""
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "chunk_*.bin")))
    parts, hold = [], []
    total = 0
    for fi, f in enumerate(files):
        with open(f, "rb") as fh:
            head = fh.read(16)
            assert head[:4] == b"MSTP", f
            ver, count, rb = np.frombuffer(head[4:], "<u4")
            assert rb == REC_BYTES, (f, rb)
            arr = np.frombuffer(fh.read(), REC_DTYPE, count=count)
        if holdout_every and fi % holdout_every == holdout_every - 1:
            hold.append(arr)
            continue
        parts.append(arr)
        total += len(arr)
        if max_records and total >= max_records:
            break
    recs = np.concatenate(parts)
    if max_records:
        recs = recs[:max_records]
    held = np.concatenate(hold) if hold else recs[:0]
    return recs, held, len(files)


def to_dense(recs):
    """[n, 849] float32 dense inputs, as `bfs::dense_features` builds them."""
    n = len(recs)
    x = np.zeros((n, DENSE_DIM), np.float32)
    act = recs["active"].astype(np.int64)
    cnt = recs["n_active"].astype(np.int64)
    mask = np.arange(MAX_ACTIVE)[None, :] < cnt[:, None]
    rows = np.repeat(np.arange(n), MAX_ACTIVE).reshape(n, MAX_ACTIVE)
    x[rows[mask], act[mask]] = 1.0
    x[:, SPARSE_DIM:] = recs["scalars"]
    return x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ValueHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 8, 1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * CELLS, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, h):
        y = F.relu(self.bn(self.conv(h))).flatten(1)
        return self.fc2(F.relu(self.fc1(y))).squeeze(1)


class ResNet(nn.Module):
    """Conv trunk over 17 feature planes + 16 broadcast scalar planes."""

    def __init__(self, ch, blocks):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(SPARSE_PLANES + N_SCALARS, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.head = ValueHead(ch)

    def forward(self, x):
        b = x.shape[0]
        planes = x[:, :SPARSE_DIM].reshape(b, SPARSE_PLANES, N, N)
        sc = x[:, SPARSE_DIM:].reshape(b, N_SCALARS, 1, 1).expand(b, N_SCALARS, N, N)
        h = self.blocks(self.stem(torch.cat([planes, sc], 1)))
        return self.head(h)


class FcResNet(nn.Module):
    """A fully connected first layer 849 -> 49*ch (the NNUE-style accumulator,
    reshaped onto the board), then a conv ResNet."""

    def __init__(self, ch, blocks):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(DENSE_DIM, CELLS * ch)
        self.bn = nn.BatchNorm2d(ch)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.head = ValueHead(ch)

    def forward(self, x):
        b = x.shape[0]
        h = F.relu(self.bn(self.fc(x).reshape(b, self.ch, N, N)))
        return self.head(self.blocks(h))


class Mlp(nn.Module):
    """NNUE-shaped: the first layer is a (sparse-input) accumulator, clipped."""

    def __init__(self, widths):
        super().__init__()
        dims = [DENSE_DIM] + widths
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for l in self.layers:
            x = torch.clamp(l(x), 0.0, 1.0)
        return self.out(x).squeeze(1)


def build(arch):
    kind, spec = arch.split(":")
    if kind == "mlp":
        return Mlp([int(w) for w in spec.split(",")])
    ch, blocks = (int(v) for v in spec.split("x"))
    if kind == "res":
        return ResNet(ch, blocks)
    if kind == "fcres":
        return FcResNet(ch, blocks)
    raise ValueError(arch)


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def evaluate(net, x, y, dev, bs=8192):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).to(dev)
            out.append(torch.sigmoid(net(xb)).float().cpu().numpy())
    net.train()
    p = np.concatenate(out)
    err = p - y
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 else float("nan"),
        # BCE against the soft targets, minus its floor (the targets' own entropy).
        "bce_excess": float(
            np.mean(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
            - np.mean(-(y * np.log(np.clip(y, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - y, 1e-6, 1))))
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="checkpoint (.pt) to start from")
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--min-visits", type=int, default=0, help="train only on records with at least this many descents")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    every = max(2, int(round(1 / args.holdout))) if args.holdout > 0 else 0
    train, hold, nfiles = load_chunks(args.data, args.max_records or None, every)
    if len(hold) == 0:  # too few files: fall back to a random split
        perm = rng.permutation(len(train))
        n_hold = max(1, int(len(train) * args.holdout))
        hold, train = train[perm[:n_hold]], train[perm[n_hold:]]
    if args.min_visits:
        train = train[train["visits"] >= args.min_visits]
        hold = hold[hold["visits"] >= args.min_visits]
    recs = train
    x_tr, y_tr = to_dense(train), train["target"].astype(np.float32)
    x_ho, y_ho = to_dense(hold), hold["target"].astype(np.float32)
    print(f"loaded {len(train) + len(hold)} records from {nfiles} chunks in {time.time() - t0:.1f}s; train {len(train)} holdout {len(hold)} (whole files)")
    print(f"target mean {y_tr.mean():.3f} std {y_tr.std():.3f}; stage mix {np.bincount(train['stage'], minlength=4).tolist()}")

    net = build(args.arch).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"arch {args.arch}: {n_params} params")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = int(math.ceil(args.epochs * len(train) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    base = evaluate(net, x_ho, y_ho, dev)
    # The static evaluation stored with each record: how far the search moved it.
    prior_err = hold["prior"] - y_ho
    print(f"holdout before: {base} | prior-vs-target mse {np.mean(prior_err**2):.4f} mae {np.mean(np.abs(prior_err)):.4f}")

    x_gpu = torch.from_numpy(x_tr).to(dev) if x_tr.nbytes < 3e9 else None
    y_gpu = torch.from_numpy(y_tr).to(dev)
    t0 = time.time()
    log_every = max(1, steps // 20)
    for step in range(steps):
        idx = torch.randint(0, len(train), (args.batch,), device=dev)
        xb = x_gpu[idx] if x_gpu is not None else torch.from_numpy(x_tr[idx.cpu().numpy()]).to(dev)
        yb = y_gpu[idx]
        loss = F.binary_cross_entropy_with_logits(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            m = evaluate(net, x_ho, y_ho, dev)
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} holdout mse {m['mse']:.5f} mae {m['mae']:.4f} r {m['r']:.4f} bce+ {m['bce_excess']:.5f} ({time.time() - t0:.0f}s)", flush=True)

    final = evaluate(net, x_ho, y_ho, dev)
    torch.save(net.state_dict(), os.path.join(args.out, "net.pt"))
    net.eval()
    exp = Export(net).to(dev).eval()
    dummy = torch.zeros(64, DENSE_DIM, device=dev)
    onnx_path = os.path.join(args.out, "value.onnx")
    torch.onnx.export(
        exp,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["v"],
        dynamic_axes={"x": {0: "b"}, "v": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    if isinstance(net, Mlp):
        # Raw weights for the Rust CPU evaluator (bfs::CpuMlpEval): u32 layer
        # count, u32 dims (in, h1, ..., 1), then per layer W as [in][out] f32
        # (transposed, so a sparse input adds one contiguous row) and b [out].
        layers = list(net.layers) + [net.out]
        with open(os.path.join(args.out, "mlp.bin"), "wb") as f:
            f.write(np.array([len(layers)], "<u4").tobytes())
            f.write(np.array([DENSE_DIM] + [l.out_features for l in layers], "<u4").tobytes())
            for l in layers:
                f.write(l.weight.detach().float().cpu().numpy().T.astype("<f4").tobytes())
                f.write(l.bias.detach().float().cpu().numpy().astype("<f4").tobytes())
    meta = {
        "arch": args.arch,
        "params": n_params,
        "data": args.data,
        "records": int(len(recs)),
        "epochs": args.epochs,
        "holdout": final,
        "holdout_before": base,
        "prior_mse": float(np.mean(prior_err**2)),
        "train_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("final", json.dumps(meta))


if __name__ == "__main__":
    main()
