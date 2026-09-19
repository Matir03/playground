"""Label generated positions with the gen405 teacher and pack them for training.

Reads the shards `minions-engine gen-positions` writes, runs the full net's
value head over them, and writes a compact record the student can train on
without re-deriving anything.

Why pack at all: the raw planes are 1274 bytes per position (26 planes x 49
cells, uint8), so 10M positions is ~12.8 GB. But `Minions::to_features` only
ever emits three kinds of plane:

  * 12 binary planes  - 6 unit types x 2 sides, the only per-cell signal
  * 1 constant plane  - the graveyard mask, identical in every position
  * 13 scalar planes  - 2 money, 10 reinforcements, 1 turns-remaining, each
                        written as one value broadcast across all 49 cells

So a position is really 588 bits + 13 bytes, not 1274 bytes. Packed that way a
record is 93 bytes and the whole set is under a gigabyte, which fits in RAM and
makes training IO-free. The graveyard plane is dropped outright: it carries no
per-position information, and the student can learn the board geometry from its
own weights.

The "scalar planes are constant across cells" claim is *checked*, not assumed --
see `--verify`. If it were ever false the packing would silently discard real
signal, which is exactly the class of bug that hides for weeks.
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
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from playground.model_server.minions_net import IN_PLANES, MinionsNet  # noqa: E402

CELLS = 49
N = 7
PLANE_BYTES = IN_PLANES * CELLS
GEN_MAGIC = 0x4D4E5053
GEN_VERSION = 1
GEN_HEADER = 4 + 4 + 8 + 4 + 4
GEN_RECORD = PLANE_BYTES + 2

# Plane groups, in `to_features` order: own units, enemy units, graveyard mask,
# money x2, reinforcements x10, turns remaining.
N_UNIT_PLANES = 12
I_GRAVEYARD = 12
I_SCALARS = 13  # first scalar plane index
N_SCALARS = 13

# Packed student record: 588 bits of occupancy, 13 scalar bytes, teacher value,
# then the two provenance bytes carried through from generation.
BITS_BYTES = (N_UNIT_PLANES * CELLS + 7) // 8  # 74
PACK_RECORD = BITS_BYTES + N_SCALARS + 4 + 2  # 93
PACK_MAGIC = 0x4D4E5544  # "MNUD"
PACK_VERSION = 1


def read_shard(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (planes uint8 (n, IN_PLANES, CELLS), turns uint8 (n,), bid int8 (n,))."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < GEN_HEADER:
        raise ValueError(f"{path}: {raw.size} bytes, shorter than a header")
    magic, version = struct.unpack("<II", raw[:8].tobytes())
    (count,) = struct.unpack("<Q", raw[8:16].tobytes())
    planes, cells = struct.unpack("<II", raw[16:24].tobytes())
    if magic != GEN_MAGIC or version != GEN_VERSION:
        raise ValueError(f"{path}: magic {magic:#x} v{version}, expected {GEN_MAGIC:#x} v{GEN_VERSION}")
    if planes != IN_PLANES or cells != CELLS:
        raise ValueError(f"{path}: {planes}x{cells} planes, expected {IN_PLANES}x{CELLS}")
    body = raw[GEN_HEADER:]
    want = count * GEN_RECORD
    if body.size != want:
        raise ValueError(f"{path}: {body.size} body bytes, expected {want} for {count} records")
    rec = body.reshape(count, GEN_RECORD)
    p = rec[:, :PLANE_BYTES].reshape(count, IN_PLANES, CELLS)
    return p, rec[:, PLANE_BYTES].copy(), rec[:, PLANE_BYTES + 1].copy().view(np.int8)


def verify_layout(p: np.ndarray) -> None:
    """Assert the plane-group structure the packing depends on."""
    units = p[:, :N_UNIT_PLANES]
    if units.max() > 1:
        raise ValueError(f"unit planes are not binary (max {units.max()})")
    # At most one unit occupies a cell, so the 12 unit planes are disjoint.
    if units.sum(axis=1).max() > 1:
        raise ValueError("two unit planes set on one cell")
    scal = p[:, I_SCALARS:I_SCALARS + N_SCALARS]
    spread = scal.max(axis=2).astype(np.int32) - scal.min(axis=2).astype(np.int32)
    if spread.max() != 0:
        bad = int(np.argmax(spread.max(axis=0)))
        raise ValueError(
            f"scalar plane {I_SCALARS + bad} varies across cells by {spread.max()} "
            "-- packing it to one byte would discard real signal"
        )
    grave = p[:, I_GRAVEYARD]
    if not np.array_equal(grave, np.broadcast_to(grave[0], grave.shape)):
        raise ValueError("graveyard plane is not constant across positions")


def pack(p: np.ndarray, value: np.ndarray, turns: np.ndarray, bid: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    out = np.empty((n, PACK_RECORD), dtype=np.uint8)
    bits = np.packbits(p[:, :N_UNIT_PLANES].reshape(n, -1), axis=1)
    out[:, :BITS_BYTES] = bits
    out[:, BITS_BYTES:BITS_BYTES + N_SCALARS] = p[:, I_SCALARS:I_SCALARS + N_SCALARS, 0]
    out[:, BITS_BYTES + N_SCALARS:BITS_BYTES + N_SCALARS + 4] = (
        value.astype("<f4").view(np.uint8).reshape(n, 4)
    )
    out[:, -2] = turns
    out[:, -1] = bid.view(np.uint8)
    return out


def load_teacher(ckpt: str, config: str, device: str) -> MinionsNet:
    cfg = json.load(open(config))["arch"]
    net = MinionsNet(
        channels=cfg["channels"], blocks=cfg["blocks"], energy_blocks=cfg["energy_blocks"]
    )
    missing, unexpected = net.load_state_dict(load_file(ckpt), strict=False)
    if missing or unexpected:
        raise SystemExit(f"teacher load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return net.eval().to(device)


@torch.no_grad()
def label(net: MinionsNet, p: np.ndarray, batch: int, device: str, half: bool) -> np.ndarray:
    out = np.empty(p.shape[0], dtype=np.float32)
    for i in range(0, p.shape[0], batch):
        x = torch.from_numpy(p[i:i + batch]).to(device, non_blocking=True)
        x = x.view(-1, IN_PLANES, N, N).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=half and device == "cuda"):
            v = net.value_head(net.trunk(x))
        out[i:i + batch] = v.float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="data/minions_nnue/positions")
    ap.add_argument("--out", default="data/minions_nnue/labelled")
    ap.add_argument("--ckpt", default="data/models/minions/live/snapshots/gen405/checkpoint.safetensors")
    ap.add_argument("--config", default="data/models/minions/live/best/config.json")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--fp16", action="store_true", help="autocast the teacher (value is still stored fp32)")
    ap.add_argument("--verify", type=int, default=1, help="verify plane layout on the first N shards")
    ap.add_argument("--delete-raw", action="store_true", help="remove each shard once labelled")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_teacher(args.ckpt, args.config, device)
    os.makedirs(args.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.positions, "positions_*.bin")))
    if not shards:
        raise SystemExit(f"no shards in {args.positions}")
    print(f"{len(shards)} shards, teacher on {device}, batch {args.batch}, fp16={args.fp16}")

    t0 = time.time()
    total = 0
    for k, path in enumerate(shards):
        dst = os.path.join(args.out, os.path.basename(path).replace("positions_", "labelled_"))
        if os.path.exists(dst):
            continue
        p, turns, bid = read_shard(path)
        if k < args.verify:
            verify_layout(p)
        v = label(net, p, args.batch, device, args.fp16)
        rec = pack(p, v, turns, bid)
        with open(dst + ".tmp", "wb") as f:
            f.write(struct.pack("<IIQ", PACK_MAGIC, PACK_VERSION, rec.shape[0]))
            f.write(rec.tobytes())
        os.replace(dst + ".tmp", dst)
        if args.delete_raw:
            os.remove(path)
        total += rec.shape[0]
        el = time.time() - t0
        print(
            f"  {os.path.basename(dst)}: {rec.shape[0]} positions, "
            f"value mean {v.mean():+.4f} std {v.std():.4f}, "
            f"{total / max(el, 1e-9):.0f}/s, {el / 60:.1f} min",
            flush=True,
        )
    print(f"done: {total} positions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
