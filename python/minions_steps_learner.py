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
import copy
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
        # Game result for the record's mover: bits 0-1 (1 loss, 2 draw, 3 win,
        # 0 unknown), bits 2-15 turns to the game's end (stepsp::outcome_code).
        ("outcome", "<u2"),
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


class SparseSet:
    """Records kept as active-index lists on the device, densified per batch:
    ~200 B per record instead of 3.4 KB dense (6M records would be 20 GB)."""

    def __init__(self, recs, dev):
        n = len(recs)
        act = recs["active"].astype(np.int64)
        cnt = recs["n_active"].astype(np.int64)
        # Inactive slots point at a spare column that is dropped after the scatter.
        act[np.arange(MAX_ACTIVE)[None, :] >= cnt[:, None]] = DENSE_DIM
        self.act = torch.from_numpy(act.astype(np.int16)).to(dev)
        self.scal = torch.from_numpy(recs["scalars"].astype(np.float32)).to(dev)
        self.n = n

    def __len__(self):
        return self.n

    def dense(self, idx):
        a = self.act[idx].long()
        x = torch.zeros(len(a), DENSE_DIM + 1, device=a.device)
        x.scatter_(1, a, 1.0)
        x = x[:, :DENSE_DIM]
        x[:, SPARSE_DIM:] = self.scal[idx]
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


class SparseExport(nn.Module):
    """Same net, sparse inputs: `idx` [B, MAX_ACTIVE] int32 active features
    (unused slots = SPARSE_DIM, a spare column dropped after the scatter) and
    `scal` [B, N_SCALARS] f32. ~9x fewer bytes per row to the GPU than `x`."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, idx, scal):
        b = idx.shape[0]
        ones = torch.ones(b, MAX_ACTIVE, device=scal.device, dtype=scal.dtype)
        x = torch.zeros(b, SPARSE_DIM + 1, device=scal.device, dtype=scal.dtype)
        x = x.scatter(1, idx.long(), ones)[:, :SPARSE_DIM]
        return torch.sigmoid(self.net(torch.cat([x, scal], 1)))


def export_onnx(net, out, dev):
    """`value.onnx` (dense input `x`) and `value_sparse.onnx` (`idx`, `scal`)."""
    net.eval()
    torch.onnx.export(
        Export(net).to(dev).eval(),
        torch.zeros(64, DENSE_DIM, device=dev),
        os.path.join(out, "value.onnx"),
        input_names=["x"],
        output_names=["v"],
        dynamic_axes={"x": {0: "b"}, "v": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        SparseExport(net).to(dev).eval(),
        (
            torch.full((64, MAX_ACTIVE), SPARSE_DIM, dtype=torch.int32, device=dev),
            torch.zeros(64, DENSE_DIM - SPARSE_DIM, device=dev),
        ),
        os.path.join(out, "value_sparse.onnx"),
        input_names=["idx", "scal"],
        output_names=["v"],
        dynamic_axes={"idx": {0: "b"}, "scal": {0: "b"}, "v": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )


def outcome_value(recs):
    """(z, known): the mover's game result as 0 / 0.5 / 1, and whether known."""
    code = recs["outcome"] & 3
    z = np.where(code == 3, 1.0, np.where(code == 2, 0.5, 0.0)).astype(np.float32)
    return z, code != 0


def evaluate(net, x, y, dev, bs=8192, z=None):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x.dense(torch.arange(i, min(i + bs, len(x)), device=dev))
            out.append(torch.sigmoid(net(xb)).float().cpu().numpy())
    net.train()
    p = np.concatenate(out)
    err = p - y
    extra = {}
    if z is not None and z[1].any():
        # How well the values predict the games' actual results.
        zz, known = z
        extra["outcome_mse"] = float(np.mean((p[known] - zz[known]) ** 2))
    return extra | {
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
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--export-only", action="store_true", help="just export --init's weights (ONNX graphs) into --out")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="checkpoint (.pt) to start from")
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--min-visits", type=int, default=0, help="train only on records with at least this many descents (the holdout keeps all)")
    ap.add_argument("--outcome-weight", type=float, default=0.0, help="train on (1-w)*search value + w*game result where the result is known")
    ap.add_argument("--outcome-decay", type=float, default=1.0, help="scale the outcome weight by this per turn to the game's end")
    ap.add_argument("--ema", type=float, default=0.0, help="evaluate and export an exponential moving average of the weights with this decay (0 = off)")
    ap.add_argument("--keep-best", type=int, default=1, help="export the checkpoint with the best holdout MSE seen at the log points, not the last")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if args.export_only:
        net = build(args.arch).to(dev)
        net.load_state_dict(torch.load(args.init, map_location=dev))
        os.makedirs(args.out, exist_ok=True)
        export_onnx(net, args.out, dev)
        print(f"exported {args.init} to {args.out}")
        return
    t0 = time.time()
    every = max(2, int(round(1 / args.holdout))) if args.holdout > 0 else 0
    train, hold, nfiles = load_chunks(args.data, args.max_records or None, every)
    if len(hold) == 0:  # too few files: fall back to a random split
        perm = rng.permutation(len(train))
        n_hold = max(1, int(len(train) * args.holdout))
        hold, train = train[perm[:n_hold]], train[perm[n_hold:]]
    if args.min_visits:
        train = train[train["visits"] >= args.min_visits]
    recs = train
    x_tr, y_tr = SparseSet(train, dev), train["target"].astype(np.float32)
    x_ho, y_ho = SparseSet(hold, dev), hold["target"].astype(np.float32)
    z_ho = outcome_value(hold)
    z_tr, known_tr = outcome_value(train)
    print(f"game results known for {known_tr.mean():.3f} of training records")
    if args.outcome_weight > 0:
        left = (train["outcome"] >> 2).astype(np.float32)
        w = args.outcome_weight * args.outcome_decay**left * known_tr
        y_tr = ((1 - w) * y_tr + w * z_tr).astype(np.float32)
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

    base = evaluate(net, x_ho, y_ho, dev, z=z_ho)
    # The static evaluation stored with each record: how far the search moved it.
    prior_err = hold["prior"] - y_ho
    print(f"holdout before: {base} | prior-vs-target mse {np.mean(prior_err**2):.4f} mae {np.mean(np.abs(prior_err)):.4f}")

    y_gpu = torch.from_numpy(y_tr).to(dev)
    t0 = time.time()
    log_every = max(1, steps // 20)
    best = (base["mse"], {k: v.clone() for k, v in net.state_dict().items()}, 0)
    # EMA of parameters and BN buffers; `judged` is what gets evaluated/kept.
    ema = copy.deepcopy(net) if args.ema > 0 else None
    judged = ema if ema is not None else net
    for step in range(steps):
        idx = torch.randint(0, len(train), (args.batch,), device=dev)
        xb = x_tr.dense(idx)
        yb = y_gpu[idx]
        loss = F.binary_cross_entropy_with_logits(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.state_dict().values(), net.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(args.ema).add_(pn, alpha=1 - args.ema)
                    else:
                        pe.copy_(pn)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            m = evaluate(judged, x_ho, y_ho, dev, z=z_ho)
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} holdout mse {m['mse']:.5f} mae {m['mae']:.4f} r {m['r']:.4f} bce+ {m['bce_excess']:.5f} z-mse {m.get('outcome_mse', float('nan')):.4f} ({time.time() - t0:.0f}s)", flush=True)
            if m["mse"] < best[0]:
                best = (m["mse"], {k: v.clone() for k, v in judged.state_dict().items()}, step + 1)

    if ema is not None:
        net.load_state_dict(ema.state_dict())
    if args.keep_best:
        net.load_state_dict(best[1])
        print(f"keeping step {best[2]} (holdout mse {best[0]:.5f})")
    final = evaluate(net, x_ho, y_ho, dev, z=z_ho)
    torch.save(net.state_dict(), os.path.join(args.out, "net.pt"))
    export_onnx(net, args.out, dev)
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
