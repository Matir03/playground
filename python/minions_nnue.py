"""A small NNUE-style value net for Minions, distilled from the gen405 champion.

The target is ~100k evaluations/second, which rules out the full net (2.25M
parameters, 8 residual blocks over a 7x7 board) and dictates the architecture
here.

Why this shape. `Minions::to_features` emits 26 planes, but only three kinds:
12 binary unit planes (6 types x 2 sides), a constant graveyard mask, and 13
scalars broadcast across all cells. So a position is 588 binary features plus 13
numbers -- and 588 sparse binary features feeding a wide first layer is exactly
NNUE's HalfKP shape. That matters for speed, not elegance: with at most a
couple of dozen units on the board the accumulator touches ~20 of 588 columns,
so an incremental update is ~20*H adds rather than a 588xH matmul, and the rest
of the net is a few hundred MACs. The dense path is implemented here (training
wants dense batches); the sparse/incremental path is what a Rust inference port
would use, and `bench_sparse` measures that cost model directly.

The graveyard plane is dropped: it is identical in every position, so it is a
constant the student absorbs into its biases.

Validation is deliberately on the *replay buffer*, not a held-out slice of the
generated set. Random-playout positions and real self-play positions are
different distributions, and the question worth answering is how well the small
net tracks the teacher on positions the search actually visits.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from playground.model_server.minions_net import IN_PLANES, MinionsNet  # noqa: E402

CELLS = 49
N = 7
N_UNIT_PLANES = 12
N_FEATURES = N_UNIT_PLANES * CELLS  # 588
N_SCALARS = 13
I_GRAVEYARD = 12
I_SCALARS = 13
BITS_BYTES = (N_FEATURES + 7) // 8  # 74
PACK_RECORD = BITS_BYTES + N_SCALARS + 4 + 2  # 93
PACK_MAGIC = 0x4D4E5544
PACK_VERSION = 1

# Scalars are money x2 (0..255), reinforcements x10 (small counts) and turns
# remaining (0..MAX_TURNS). Fixed divisors rather than dataset statistics, so
# the same normalisation applies to any later data without a second pass.
SCALE = np.array([30.0, 30.0] + [4.0] * 10 + [30.0], dtype=np.float32)


# ── data ──────────────────────────────────────────────────────────────────────


def read_labelled(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (bits uint8 (n, BITS_BYTES), scalars uint8 (n, 13), value float32 (n,))."""
    raw = np.fromfile(path, dtype=np.uint8)
    magic, version = struct.unpack("<II", raw[:8].tobytes())
    (count,) = struct.unpack("<Q", raw[8:16].tobytes())
    if magic != PACK_MAGIC or version != PACK_VERSION:
        raise ValueError(f"{path}: magic {magic:#x} v{version}")
    rec = raw[16:].reshape(count, PACK_RECORD)
    bits = rec[:, :BITS_BYTES]
    scal = rec[:, BITS_BYTES:BITS_BYTES + N_SCALARS]
    val = rec[:, BITS_BYTES + N_SCALARS:BITS_BYTES + N_SCALARS + 4].copy().view("<f4").reshape(count)
    return bits, scal, val


def unpack_bits(bits: np.ndarray) -> np.ndarray:
    """(n, 74) packed -> (n, 588) float32 dense."""
    out = np.unpackbits(bits, axis=1)[:, :N_FEATURES]
    return out.astype(np.float32)


def planes_to_inputs(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replay-buffer planes (n, 1274) uint8 -> (features (n,588), scalars (n,13)).

    The same split the packer applies, so buffer positions and generated ones
    reach the student through identical arithmetic.
    """
    p = planes.reshape(-1, IN_PLANES, CELLS)
    feats = p[:, :N_UNIT_PLANES].reshape(-1, N_FEATURES).astype(np.float32)
    scal = p[:, I_SCALARS:I_SCALARS + N_SCALARS, 0].astype(np.float32)
    return feats, scal


# ── model ─────────────────────────────────────────────────────────────────────


class ClippedReLU(nn.Module):
    """NNUE's activation: clamping to [0,1] is what keeps an int8 quantisation
    of this net faithful. Used here even though training is float, so the
    float and quantised nets cannot diverge in shape later."""

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class MinionsNNUE(nn.Module):
    """588 sparse features -> 512 accumulator -> 32 -> 64 -> 32 -> 1.

    The 13 scalars are concatenated raw (already normalised) onto the
    accumulator output rather than passed through their own embedding, so the
    post-input layer widths are exactly the ones specified. They are dense and
    tiny; an embedding for 13 numbers bought nothing but a layer to quantise.
    """

    def __init__(self, hidden: int = 512, l1: int = 32, l2: int = 64, l3: int = 32):
        super().__init__()
        # The feature transformer. Sparse in deployment: one column per active
        # feature, summed. Dense here because training batches are dense.
        self.ft = nn.Linear(N_FEATURES, hidden)
        self.act = ClippedReLU()
        self.fc1 = nn.Linear(hidden + N_SCALARS, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.out = nn.Linear(l3, 1)
        self.hidden, self.l1, self.l2, self.l3 = hidden, l1, l2, l3

    def forward(self, feats, scalars):
        a = self.act(self.ft(feats))
        x = torch.cat([a, scalars], dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        # tanh: the teacher's value head is bounded, so the student should be.
        return torch.tanh(self.out(x)).squeeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── teacher ───────────────────────────────────────────────────────────────────


def load_teacher(ckpt: str, config: str, device: str) -> MinionsNet:
    cfg = json.load(open(config))["arch"]
    net = MinionsNet(channels=cfg["channels"], blocks=cfg["blocks"], energy_blocks=cfg["energy_blocks"])
    missing, unexpected = net.load_state_dict(
        __import__("safetensors.torch", fromlist=["load_file"]).load_file(ckpt), strict=False
    )
    if missing or unexpected:
        raise SystemExit(f"teacher mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return net.eval().to(device)


@torch.no_grad()
def teacher_values(net: MinionsNet, planes: np.ndarray, batch: int, device: str) -> np.ndarray:
    out = np.empty(planes.shape[0], dtype=np.float32)
    for i in range(0, planes.shape[0], batch):
        x = torch.from_numpy(planes[i:i + batch]).to(device).view(-1, IN_PLANES, N, N).float()
        out[i:i + batch] = net.value_head(net.trunk(x)).float().cpu().numpy()
    return out


# ── validation on the replay buffer ───────────────────────────────────────────


def load_buffer(buffer_dir: str, max_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (planes (n,1274) uint8, stored value (n,)) from real self-play."""
    import minions_az_learner as L

    planes, values = [], []
    for path in sorted(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))[-max_chunks:]:
        d = L.read_chunk(path)
        if d is None:
            continue
        planes.append(d["planes"])
        values.append(d["value"])
    if not planes:
        raise SystemExit(f"no readable chunks in {buffer_dir}")
    return np.concatenate(planes), np.concatenate(values)


@torch.no_grad()
def evaluate(student, teacher, planes, stored, device, batch=8192) -> dict:
    feats, scal = planes_to_inputs(planes)
    scal = scal / SCALE
    sv = np.empty(len(planes), dtype=np.float32)
    student.eval()
    for i in range(0, len(planes), batch):
        f = torch.from_numpy(feats[i:i + batch]).to(device)
        s = torch.from_numpy(scal[i:i + batch]).to(device)
        sv[i:i + batch] = student(f, s).float().cpu().numpy()
    tv = teacher_values(teacher, planes, batch, device)
    d_st = sv - tv
    return {
        "n": int(len(planes)),
        # The headline number: how well the student reproduces the teacher on
        # positions the search actually visits.
        "student_vs_teacher_mae": float(np.abs(d_st).mean()),
        "student_vs_teacher_rmse": float(np.sqrt((d_st ** 2).mean())),
        "student_vs_teacher_r": float(np.corrcoef(sv, tv)[0, 1]),
        "student_vs_teacher_p99": float(np.quantile(np.abs(d_st), 0.99)),
        # Context: both against the buffer's own stored target. The teacher's
        # number is the floor -- the student cannot beat its own teacher.
        "teacher_vs_stored_mae": float(np.abs(tv - stored).mean()),
        "student_vs_stored_mae": float(np.abs(sv - stored).mean()),
        "teacher_mean": float(tv.mean()),
        "teacher_std": float(tv.std()),
        "student_mean": float(sv.mean()),
        "student_std": float(sv.std()),
    }


# ── throughput ────────────────────────────────────────────────────────────────


def bench_sparse(model: MinionsNNUE, reps: int = 20000, active: int = 27) -> dict:
    """Cost model for the deployed sparse path, on CPU, single thread.

    Pure numpy on purpose. An earlier version of this called torch ops per
    evaluation and reported ~11k/s *flat* across hidden 128/256/512 -- a 4x
    change in the only thing that should matter. That flatness proved it was
    timing torch's per-op dispatch (~90us) rather than the network, so the
    number was meaningless. Anything that measures a cost model must scale with
    the cost, or it is measuring the harness.

    Even this is an upper bound on cost: it is interpreted numpy, one BLAS call
    per layer on tiny matrices. A Rust/SIMD port with int16 accumulators is the
    deployable number, and it is strictly faster than this. Read this as a
    conservative floor.
    """
    m = model.cpu().eval()
    W = m.ft.weight.detach().numpy().T.astype(np.float32).copy()  # (588, hidden)
    bft = m.ft.bias.detach().numpy().astype(np.float32).copy()
    W1 = m.fc1.weight.detach().numpy().T.astype(np.float32).copy()
    b1 = m.fc1.bias.detach().numpy().astype(np.float32).copy()
    W2 = m.fc2.weight.detach().numpy().T.astype(np.float32).copy()
    b2 = m.fc2.bias.detach().numpy().astype(np.float32).copy()
    W3 = m.fc3.weight.detach().numpy().T.astype(np.float32).copy()
    b3 = m.fc3.bias.detach().numpy().astype(np.float32).copy()
    Wo = m.out.weight.detach().numpy().T.astype(np.float32).copy()
    bo = m.out.bias.detach().numpy().astype(np.float32).copy()

    rng = np.random.default_rng(0)
    idx = rng.choice(N_FEATURES, size=active, replace=False)
    scal = np.zeros(N_SCALARS, dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(reps):
        acc = np.clip(bft + W[idx].sum(axis=0), 0.0, 1.0)
        x = np.concatenate([acc, scal])
        x = np.clip(x @ W1 + b1, 0.0, 1.0)
        x = np.clip(x @ W2 + b2, 0.0, 1.0)
        x = np.clip(x @ W3 + b3, 0.0, 1.0)
        _ = np.tanh(x @ Wo + bo)
    el = time.perf_counter() - t0
    return {
        "evals": reps,
        "seconds": el,
        "evals_per_sec": reps / el,
        "us_per_eval": el / reps * 1e6,
        "active_features": active,
        "note": "interpreted numpy, single thread; a Rust/SIMD port is strictly faster",
    }


# ── training ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="data/minions_nnue/labelled")
    ap.add_argument("--buffer", default="data/models/minions/retrain-tb/buffer")
    ap.add_argument("--val-chunks", type=int, default=40)
    ap.add_argument("--ckpt", default="data/models/minions/live/snapshots/gen405/checkpoint.safetensors")
    ap.add_argument("--config", default="data/models/minions/live/best/config.json")
    ap.add_argument("--out", default="data/minions_nnue/student.pt")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--l1", type=int, default=32)
    ap.add_argument("--l2", type=int, default=64)
    ap.add_argument("--l3", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = MinionsNNUE(hidden=args.hidden, l1=args.l1, l2=args.l2, l3=args.l3).to(device)
    print(
        f"student: {student.n_params()} params, "
        f"{N_FEATURES}+{N_SCALARS} -> {args.hidden} -> {args.l1} -> {args.l2} -> {args.l3} -> 1"
    )

    if args.bench_only:
        print(json.dumps(bench_sparse(student), indent=2))
        return

    shards = sorted(glob.glob(os.path.join(args.labelled, "labelled_*.bin")))
    if args.max_shards:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"no labelled shards in {args.labelled}")
    print(f"loading {len(shards)} shards ...")
    bits, scal, val = [], [], []
    for p in shards:
        b, s, v = read_labelled(p)
        bits.append(b)
        scal.append(s)
        val.append(v)
    bits = np.concatenate(bits)
    scal = (np.concatenate(scal).astype(np.float32) / SCALE)
    val = np.concatenate(val)
    n = len(val)
    print(f"  {n} positions, teacher value mean {val.mean():+.4f} std {val.std():.4f}")

    # Features stay *packed* in host memory and are unpacked per batch. Dense
    # they would be (n, 588) float32 -- 23.5 GB at 10M positions, against ~23 GB
    # available on this box. Packed they are 74 bytes a row (740 MB), and a
    # 16k batch unpacks to 38 MB. This is the difference between the run
    # completing and the machine swapping itself to death.
    st = torch.from_numpy(scal)
    vt = torch.from_numpy(val)
    print(
        f"  host memory: bits {bits.nbytes / 1e9:.2f} GB packed, "
        f"scalars {scal.nbytes / 1e9:.2f} GB, values {val.nbytes / 1e9:.2f} GB"
    )

    # Hold out a slice of the *generated* set as well as validating on the
    # buffer. In-distribution error needs no teacher forward pass -- the packed
    # records already carry the teacher's value -- and it is the number that
    # separates "undertrained" from "does not transfer". The two distributions
    # are measurably different (generated: teacher mean +0.37 std 0.75; buffer:
    # +0.02 std 0.93), so transfer is a real question. Without this number a bad
    # buffer result is ambiguous and costs a whole re-run to disambiguate.
    holdout = min(200_000, n // 10)
    n_train = n - holdout
    print(f"  train {n_train}, in-distribution holdout {holdout}")

    @torch.no_grad()
    def indist() -> tuple[float, float]:
        student.eval()
        errs = []
        for i in range(n_train, n, 16384):
            f = torch.from_numpy(unpack_bits(bits[i:i + 16384])).to(device)
            s = st[i:i + 16384].to(device)
            p = student(f, s).float().cpu().numpy()
            errs.append(p - val[i:i + 16384])
        student.train()
        e = np.concatenate(errs)
        return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))

    teacher = load_teacher(args.ckpt, args.config, device)
    vplanes, vstored = load_buffer(args.buffer, args.val_chunks)
    print(f"validation: {len(vplanes)} buffer positions from {args.val_chunks} chunks")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * ((n + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    step = 0
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(n_train)
        tot, seen = 0.0, 0
        t0 = time.time()
        for i in range(0, n_train, args.batch):
            j = perm[i:i + args.batch]
            # Unpack just this batch: 38 MB, against 23.5 GB for the whole set.
            f = torch.from_numpy(unpack_bits(bits[j.numpy()])).to(device, non_blocking=True)
            s = st[j].to(device, non_blocking=True)
            y = vt[j].to(device, non_blocking=True)
            pred = student(f, s)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += loss.item() * len(j)
            seen += len(j)
        m = evaluate(student, teacher, vplanes, vstored, device)
        id_mae, id_rmse = indist()
        print(
            f"epoch {ep + 1}/{args.epochs}  train_mse {tot / seen:.5f}  "
            f"[in-dist] MAE {id_mae:.4f} RMSE {id_rmse:.4f}  "
            f"[buffer] vs-teacher MAE {m['student_vs_teacher_mae']:.4f} "
            f"RMSE {m['student_vs_teacher_rmse']:.4f} r {m['student_vs_teacher_r']:.4f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    m = evaluate(student, teacher, vplanes, vstored, device)
    bench = bench_sparse(MinionsNNUE(hidden=args.hidden, l1=args.l1, l2=args.l2, l3=args.l3))
    print("\n=== final ===")
    print(json.dumps({"validation": m, "sparse_cpu_bench": bench}, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "hidden": args.hidden,
            "l1": args.l1,
            "l2": args.l2,
            "l3": args.l3,
            "scale": SCALE,
            "validation": m,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
