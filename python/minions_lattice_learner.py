"""Nets for the turn-lattice search (docs/minions-lattice.md).

Tier 1 is the steps-mode ResNet (minions_steps_learner.py, `res:CxB`): its
stem + blocks are the trunk, emitting a C x 7 x 7 image of the turn start,
and its value head is the tier-1 value. Tier-1 records (`lattice-gen` output
`t1/`) are steps-format, so the steps learner trains it unchanged.

Tier 2 (this script) is a small net reading the trunk's image of the turn
start plus the lattice node's own feature planes (the steps view without the
`next` plane). Per cell it emits prior logits and value logits over 37
slots -- the choice of whatever variable sits on that cell: a fate code, a
move offset, a spawn's unit code -- and one scalar value for the node.

Training (`--data`: `lattice-gen` output dirs, reading `t2/`):
  prior   cross-entropy, per record and cell, between the softmax over that
          cell's offered slots and its children's visit shares
  values  BCE of each child's value against its mean value in the search
  node    BCE of the node value against the node's mean value
The tier-1 trunk is frozen (taken from `--t1`, a steps net.pt).

Export (`--out`), two graphs so the trunk runs once per turn start:
  tier1.onnx  ridx [B,96] int32, rscal [B,16] (the turn start)
              -> v1 [B] tier-1 value, h [B,C,7,7] fp16 trunk image
  tier2.onnx  h [B,C,7,7] fp16, nidx [B,96] int32, nscal [B,16] (the node),
              kid [T] int32 (row * 49*37 + cell * 37 + slot, all rows' children)
              -> logit [T] prior logits, value [T] child values, v2 [B]
  lattice.json  {"c_trunk": C}
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minions_steps_learner import (
    CELLS,
    MAX_ACTIVE,
    N,
    N_SCALARS,
    SPARSE_DIM,
    SPARSE_PLANES,
    ResBlock,
    build,
)

SLOTS = 37
T2_REC_BYTES = 528
T2_KID_BYTES = 12

T2_DTYPE = np.dtype(
    [
        ("rn", "u1"),
        ("nn", "u1"),
        ("stage", "u1"),
        ("pad", "u1"),
        ("kids", "<u2"),
        ("turn", "<u2"),
        ("n", "<f4"),
        ("q", "<f4"),
        ("ract", "<u2", (MAX_ACTIVE,)),
        ("rscal", "<f4", (N_SCALARS,)),
        ("nact", "<u2", (MAX_ACTIVE,)),
        ("nscal", "<f4", (N_SCALARS,)),
    ]
)
assert T2_DTYPE.itemsize == T2_REC_BYTES
KID_DTYPE = np.dtype([("cell", "u1"), ("slot", "u1"), ("pad", "<u2"), ("n", "<f4"), ("q", "<f4")])
assert KID_DTYPE.itemsize == T2_KID_BYTES


def read_chunk(path, magic, dtype):
    with open(path, "rb") as f:
        head = f.read(16)
        assert head[:4] == magic, (path, head[:4])
        count, rec = np.frombuffer(head[8:16], "<u4")
        assert rec == dtype.itemsize, (path, rec)
        return np.frombuffer(f.read(), dtype, count=count)


def load_t2(dirs):
    recs, kids = [], []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "t2", "chunk_*.bin"))):
            r = read_chunk(p, b"MLT2", T2_DTYPE)
            k = read_chunk(p[:-4] + ".kids", b"MLTK", KID_DTYPE)
            assert int(r["kids"].sum()) == len(k), p
            recs.append(r)
            kids.append(k)
    return np.concatenate(recs), np.concatenate(kids)


def dense(act, cnt, scal):
    """Sparse feature rows -> dense [B, 1241] on act's device."""
    b = act.shape[0]
    slots = torch.arange(MAX_ACTIVE, device=act.device)[None, :]
    a = act.long().masked_fill(slots >= cnt[:, None].long(), SPARSE_DIM)
    x = torch.zeros(b, SPARSE_DIM + 1, device=act.device)
    x.scatter_(1, a, 1.0)
    return torch.cat([x[:, :SPARSE_DIM], scal], 1)


def planes_of(x):
    b = x.shape[0]
    p = x[:, :SPARSE_DIM].reshape(b, SPARSE_PLANES, N, N)
    s = x[:, SPARSE_DIM:].reshape(b, N_SCALARS, 1, 1).expand(b, N_SCALARS, N, N)
    return torch.cat([p, s], 1)


class Trunk(nn.Module):
    """The steps ResNet split: trunk image and value logit."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        h = self.net.blocks(self.net.stem(planes_of(x)))
        return h, self.net.head(h)


class T2Net(nn.Module):
    def __init__(self, c_trunk, ch, blocks):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(c_trunk + SPARSE_PLANES + N_SCALARS, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.kid = nn.Conv2d(ch, 2 * SLOTS, 1)
        self.vconv = nn.Sequential(nn.Conv2d(ch, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(4 * CELLS, 64), nn.ReLU(), nn.Linear(64, 1))
        # Start from uniform priors.
        nn.init.zeros_(self.kid.weight[:SLOTS])
        nn.init.zeros_(self.kid.bias[:SLOTS])

    def forward(self, h, x):
        """h: trunk image [B, C, 7, 7]; x: the node's dense features. Returns
        per-(cell, slot) logits and value logits, flattened as cell * 37 +
        slot, and the node's value logit."""
        y = self.blocks(self.inp(torch.cat([h, planes_of(x)], 1)))
        out = self.kid(y)  # [B, 74, 7, 7]
        b = out.shape[0]
        logit = out[:, :SLOTS].reshape(b, SLOTS, CELLS).transpose(1, 2).reshape(b, CELLS * SLOTS)
        vlog = out[:, SLOTS:].reshape(b, SLOTS, CELLS).transpose(1, 2).reshape(b, CELLS * SLOTS)
        v = self.vfc(self.vconv(y).flatten(1)).squeeze(1)
        return logit, vlog, v


def dense_x(idx, scal):
    b = idx.shape[0]
    ones = torch.ones(b, MAX_ACTIVE, device=scal.device, dtype=scal.dtype)
    x = torch.zeros(b, SPARSE_DIM + 1, device=scal.device, dtype=scal.dtype)
    x = x.scatter(1, idx.long(), ones)[:, :SPARSE_DIM]
    return torch.cat([x, scal], 1)


class Tier1Export(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, ridx, rscal):
        h, v1 = self.trunk(dense_x(ridx, rscal))
        return torch.sigmoid(v1), h.half()


class Tier2Export(nn.Module):
    def __init__(self, t2):
        super().__init__()
        self.t2 = t2

    def forward(self, h, nidx, nscal, kid):
        # kid: flat [T] indices row * 49*37 + cell * 37 + slot (no padding).
        logit, vlog, v2 = self.t2(h.float(), dense_x(nidx, nscal))
        k = kid.long()
        return logit.reshape(-1).gather(0, k), torch.sigmoid(vlog.reshape(-1).gather(0, k)), torch.sigmoid(v2)


def export_onnx(trunk, t2, out, dev):
    """`tier1.onnx` (turn start -> value, fp16 trunk image), `tier2.onnx`
    (trunk image + node + children -> children's logits and values, node
    value) and `lattice.json` (the trunk's width)."""
    trunk.eval()
    t2.eval()
    b, k = 64, 24
    c = trunk.net.stem[0].out_channels
    pad = torch.full((b, MAX_ACTIVE), SPARSE_DIM, dtype=torch.int32, device=dev)
    sc = torch.zeros(b, N_SCALARS, device=dev)
    torch.onnx.export(
        Tier1Export(trunk).to(dev).eval(),
        (pad, sc),
        os.path.join(out, "tier1.onnx"),
        input_names=["ridx", "rscal"],
        output_names=["v1", "h"],
        dynamic_axes={"ridx": {0: "b"}, "rscal": {0: "b"}, "v1": {0: "b"}, "h": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        Tier2Export(t2).to(dev).eval(),
        (torch.zeros(b, c, N, N, dtype=torch.float16, device=dev), pad.clone(), sc.clone(), torch.zeros(b * k, dtype=torch.int32, device=dev)),
        os.path.join(out, "tier2.onnx"),
        input_names=["h", "nidx", "nscal", "kid"],
        output_names=["logit", "value", "v2"],
        dynamic_axes={
            "h": {0: "b"},
            "nidx": {0: "b"},
            "nscal": {0: "b"},
            "kid": {0: "t"},
            "logit": {0: "t"},
            "value": {0: "t"},
            "v2": {0: "b"},
        },
        opset_version=17,
        dynamo=False,
    )
    with open(os.path.join(out, "lattice.json"), "w") as f:
        json.dump({"c_trunk": c}, f)


def group_log_softmax(logit, group, n_groups):
    """log-softmax of `logit` within each group id."""
    m = torch.full((n_groups,), -1e30, device=logit.device).scatter_reduce(0, group, logit, "amax")
    e = torch.exp(logit - m[group])
    s = torch.zeros(n_groups, device=logit.device).index_add_(0, group, e)
    return logit - m[group] - torch.log(s[group])


class T2Set:
    """Records and kids on the device, with CSR offsets."""

    def __init__(self, recs, kids, dev):
        t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a if dt is None else a.astype(dt))).to(dev)
        self.ract = t(recs["ract"].view(np.int16))
        self.rn = t(recs["rn"])
        self.rscal = t(recs["rscal"], np.float32)
        self.nact = t(recs["nact"].view(np.int16))
        self.nn = t(recs["nn"])
        self.nscal = t(recs["nscal"], np.float32)
        self.q = t(recs["q"], np.float32)
        self.nkids = t(recs["kids"], np.int64)
        self.off = torch.cumsum(self.nkids, 0) - self.nkids
        self.kflat = t(kids["cell"], np.int64) * SLOTS + t(kids["slot"], np.int64)
        self.kcell = t(kids["cell"], np.int64)
        self.kn = t(kids["n"], np.float32)
        self.kq = t(kids["q"], np.float32)
        self.stage = recs["stage"]
        self.n = len(recs)

    def batch(self, idx):
        cnt = self.nkids[idx]
        rec_of = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), cnt)
        start = torch.repeat_interleave(self.off[idx], cnt)
        first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        kid = start + torch.arange(len(rec_of), device=idx.device) - first
        return rec_of, kid


def losses(trunk, t2, ds, idx):
    with torch.no_grad():
        h, _ = trunk(dense(ds.ract[idx], ds.rn[idx], ds.rscal[idx]))
    logit, vlog, v = t2(h, dense(ds.nact[idx], ds.nn[idx], ds.nscal[idx]))
    rec_of, kid = ds.batch(idx)
    flat = ds.kflat[kid]
    lg = logit[rec_of, flat]
    vl = vlog[rec_of, flat]
    # Groups: (record in batch, cell).
    group = rec_of * CELLS + ds.kcell[kid]
    ng = len(idx) * CELLS
    lp = group_log_softmax(lg, group, ng)
    tot = torch.zeros(ng, device=lg.device).index_add_(0, group, ds.kn[kid])
    tgt = ds.kn[kid] / tot[group].clamp(min=1e-9)
    n_cells = (tot > 0).sum().clamp(min=1)
    l_prior = -(tgt * lp).sum() / n_cells
    # Its floor: the targets' own entropy.
    h_tgt = -(tgt * torch.log(tgt.clamp(min=1e-9))).sum() / n_cells
    l_val = F.binary_cross_entropy_with_logits(vl, ds.kq[kid])
    l_node = F.binary_cross_entropy_with_logits(v, ds.q[idx])
    return l_prior, l_val, l_node, h_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--t1", required=True, help="steps net.pt for the tier-1 trunk + value head")
    ap.add_argument("--t1-arch", default="res:32x4")
    ap.add_argument("--arch", default="32x2", help="tier-2 CHANNELSxBLOCKS")
    ap.add_argument("--init", help="tier-2 weights to start from (t2.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-every", type=int, default=20)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    net1 = build(args.t1_arch).to(dev)
    net1.load_state_dict(torch.load(args.t1, map_location=dev))
    trunk = Trunk(net1).to(dev).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    c_trunk = net1.stem[0].out_channels
    ch, blocks = (int(v) for v in args.arch.split("x"))
    t2 = T2Net(c_trunk, ch, blocks).to(dev)
    if args.init:
        t2.load_state_dict(torch.load(args.init, map_location=dev))
    if args.export_only:
        torch.save(t2.state_dict(), os.path.join(args.out, "t2.pt"))
        export_onnx(trunk, t2, args.out, dev)
        return
    t0 = time.time()
    recs, kids = load_t2(args.data)
    ds = T2Set(recs, kids, dev)
    all_idx = torch.arange(ds.n, device=dev)
    ho_mask = (all_idx % args.holdout_every) == 0
    ho, tr = all_idx[ho_mask], all_idx[~ho_mask]
    print(f"{ds.n} records ({len(kids)} kids, {len(kids) / max(ds.n, 1):.1f}/record), {len(tr)} train, load {time.time() - t0:.0f}s")
    print("stages:", np.bincount(ds.stage, minlength=4).tolist())

    def holdout():
        t2.eval()
        out = np.zeros(4)
        with torch.no_grad():
            for i in range(0, len(ho), 4096):
                out += np.array([x.item() for x in losses(trunk, t2, ds, ho[i : i + 4096])]) * len(ho[i : i + 4096])
        t2.train()
        o = out / max(len(ho), 1)
        return {"prior": o[0], "prior_excess": o[0] - o[3], "value": o[1], "node": o[2]}

    base = holdout()
    print("holdout before:", json.dumps(base))
    opt = torch.optim.AdamW(t2.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = int(args.epochs * len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1))
    t2.train()
    for step in range(steps):
        idx = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        lp, lv, ln, _ = losses(trunk, t2, ds, idx)
        loss = lp + lv + ln
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"step {step}/{steps} prior {lp.item():.4f} value {lv.item():.4f} node {ln.item():.4f}", flush=True)
    final = holdout()
    print("holdout after:", json.dumps(final))
    torch.save(t2.state_dict(), os.path.join(args.out, "t2.pt"))
    export_onnx(trunk, t2, args.out, dev)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {"t1": args.t1, "arch": args.arch, "data": args.data, "records": ds.n, "holdout_before": base, "holdout": final, "steps": steps},
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
