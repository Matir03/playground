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
  t1      BCE of the tier-1 value against the tier-1 search value (`t1/`)
trained jointly: the tier-2 losses train the trunk too, so the trunk learns
from the (many more) tier-2 records. `--freeze-trunk` trains tier 2 alone on
a fixed tier 1. Tier 1 starts from `--t1` (a steps net.pt) or random weights.

Export (`--out`), two graphs so the trunk runs once per turn start:
  tier1.onnx  ridx [B,96] int32, rscal [B,16] (the turn start)
              -> v1 [B] tier-1 value, h [B,C,7,7] fp16 trunk image
  tier2.onnx  h [B,C,7,7] fp16, nidx [B,96] int32, nscal [B,16] (the node),
              kid [T] int32 (row * 49*37 + cell * 37 + slot, all rows' children)
              -> logit [T] prior logits, value [T] child values, v2 [B]
  lattice.json  {"c_trunk": C}
plus net.pt / t2.pt and the tier-1 net alone as a steps value net
(value.onnx, value_sparse.onnx).
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
    ResNet,
    SparseSet,
    build,
    load_chunks,
)
from minions_steps_learner import export_onnx as steps_export_onnx

SLOTS = 37
# The lattice view (lview.rs, `--lview`): original position plus decisions,
# and per-cell output slots in separate ranges per kind of variable.
LV_FATE_OFF, LV_MOVE_OFF, LV_SPAWN_OFF, LV_SLOTS = 0, 3, 40, 46
E_DTYPE = np.dtype(
    [
        ("kind", "u1"),  # 0: a turn start (opens a group), 1: one of its tier-1 children
        ("n_active", "u1"),
        ("pad", "<u2"),
        ("visits", "<f4"),
        ("active", "<u2", (MAX_ACTIVE,)),
        ("scalars", "<f4", (N_SCALARS,)),
    ]
)
T2_REC_BYTES = 528
T2_KID_BYTES = 12

T2_DTYPE = np.dtype(
    [
        ("rn", "u1"),
        ("nn", "u1"),
        ("stage", "u1"),
        # 1: the values average sink returns only (-1 = no target); 0: older
        # records, whose values averaged every visit (intermediate ones too).
        ("sink", "u1"),
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


OLD_DTYPE = np.dtype(
    [
        ("rn", "u1"),
        ("nn", "u1"),
        ("pad", "<u2"),
        ("ract", "<u2", (MAX_ACTIVE,)),
        ("rscal", "<f4", (N_SCALARS,)),
        ("nact", "<u2", (MAX_ACTIVE,)),
        ("nscal", "<f4", (N_SCALARS,)),
    ]
)


def read_chunk(path, magic, dtype):
    with open(path, "rb") as f:
        head = f.read(16)
        assert head[:4] == magic, (path, head[:4])
        count, rec = np.frombuffer(head[8:16], "<u4")
        assert rec == dtype.itemsize, (path, rec)
        return np.frombuffer(f.read(), dtype, count=count)


def load_t2(dirs, old=False):
    """Records and kids; with `old`, only chunks with `.old` side files
    (--teacher-view) and those too."""
    recs, kids, olds = [], [], []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "t2", "chunk_*.bin"))):
            if old and not os.path.exists(p[:-4] + ".old"):
                continue
            r = read_chunk(p, b"MLT2", T2_DTYPE)
            k = read_chunk(p[:-4] + ".kids", b"MLTK", KID_DTYPE)
            assert int(r["kids"].sum()) == len(k), p
            recs.append(r)
            kids.append(k)
            if old:
                o = read_chunk(p[:-4] + ".old", b"MLTO", OLD_DTYPE)
                assert len(o) == len(r), p
                olds.append(o)
    if old:
        return np.concatenate(recs), np.concatenate(kids), np.concatenate(olds)
    return np.concatenate(recs), np.concatenate(kids)


# The trunk's input is a full-turn position only (2026-10-04, the user's
# call): unit and graveyard planes (steps.rs P_UNITS..P_GRAVE) and the
# position scalars (money, reinforcements, turns left: scalars 0..12), plus
# one constant plane. Decision planes and stage flags -- partial turns --
# are not part of it; they reach tier 2 only, on top of the trunk.
POS_PLANES = 13
POS_SCALARS = 13
P_NEXT = 16


class PosNet(ResNet):
    """The steps ResNet on full-turn positions: reads the steps feature
    layout but only its position channels."""

    def __init__(self, ch, blocks):
        super().__init__(ch, blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(POS_PLANES + POS_SCALARS + 1, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()
        )

    def inputs(self, x):
        b = x.shape[0]
        planes = x[:, :SPARSE_DIM].reshape(b, SPARSE_PLANES, N, N)[:, :POS_PLANES]
        sc = x[:, SPARSE_DIM : SPARSE_DIM + POS_SCALARS].reshape(b, POS_SCALARS, 1, 1).expand(b, POS_SCALARS, N, N)
        return torch.cat([planes, sc, torch.ones(b, 1, N, N, device=x.device, dtype=x.dtype)], 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(self.inputs(x))))


def pos_from_res(sd):
    """A steps ResNet's weights for a PosNet: the stem keeps the position
    channels, and the constant plane takes the turn-start stage flag
    (fates, scalar 13), so turn starts read as before but for the `next`
    marker, which is dropped."""
    w = sd["stem.0.weight"]
    if w.shape[1] == POS_PLANES + POS_SCALARS + 1:
        return sd
    assert w.shape[1] == SPARSE_PLANES + N_SCALARS, w.shape
    sd = dict(sd)
    sd["stem.0.weight"] = torch.cat(
        [w[:, :POS_PLANES], w[:, SPARSE_PLANES : SPARSE_PLANES + POS_SCALARS], w[:, SPARSE_PLANES + 13 : SPARSE_PLANES + 14]], 1
    )
    return sd


def build_t1(arch):
    if arch.startswith("pos:"):
        ch, blocks = (int(v) for v in arch[4:].split("x"))
        return PosNet(ch, blocks)
    return build(arch)


def turn_records(r1):
    """Tier-1 records of turn starts only: older generations also wrote
    mid-turn records (decision planes or a later stage)."""
    plane = r1["active"].astype(np.int64) // CELLS
    live = np.arange(MAX_ACTIVE)[None, :] < r1["n_active"][:, None]
    dec = live & (plane >= POS_PLANES) & (plane != P_NEXT)
    return (r1["stage"] == 0) & ~dec.any(1)


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
        inp = self.net.inputs(x) if hasattr(self.net, "inputs") else planes_of(x)
        h = self.net.blocks(self.net.stem(inp))
        return h, self.net.head(h)


class T2Net(nn.Module):
    def __init__(self, c_trunk, ch, blocks, lview=False):
        super().__init__()
        self.lview = lview
        self.slots = SLOTS = LV_SLOTS if lview else globals()["SLOTS"]
        # Scalar outputs: the node value; with --lview also the input
        # position's value (tier 1) and the energy of a complete turn.
        self.n_scalar = 3 if lview else 1
        self.inp = nn.Sequential(
            nn.Conv2d(c_trunk + SPARSE_PLANES + N_SCALARS, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.kid = nn.Conv2d(ch, 2 * SLOTS, 1)
        self.vconv = nn.Sequential(nn.Conv2d(ch, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(4 * CELLS, 64), nn.ReLU(), nn.Linear(64, self.n_scalar))
        # Start from uniform priors.
        nn.init.zeros_(self.kid.weight[:SLOTS])
        nn.init.zeros_(self.kid.bias[:SLOTS])

    def forward(self, h, x):
        """h: trunk image [B, C, 7, 7]; x: the node's dense features. Returns
        per-(cell, slot) logits and value logits, flattened as cell * slots +
        slot, the node's value logit and the other scalar outputs [B, 0 or 2]
        (position value logit, energy)."""
        SLOTS = self.slots
        y = self.blocks(self.inp(torch.cat([h, planes_of(x)], 1)))
        out = self.kid(y)  # [B, 2 * slots, 7, 7]
        b = out.shape[0]
        logit = out[:, :SLOTS].reshape(b, SLOTS, CELLS).transpose(1, 2).reshape(b, CELLS * SLOTS)
        vlog = out[:, SLOTS:].reshape(b, SLOTS, CELLS).transpose(1, 2).reshape(b, CELLS * SLOTS)
        sv = self.vfc(self.vconv(y).flatten(1))
        return logit, vlog, sv[:, 0], sv[:, 1:]


def translate_t2(sd, lview):
    """Old (37 shared slots, one scalar) tier-2 weights as an --lview head's
    starting point: each kind's slot range takes the old slots, the position
    value starts as the old node value (what tier 1 read with --t1-from-t2),
    the energy at zero, and the input planes as near their old meaning as
    there is (the decision planes differ; distillation retrains them)."""
    if not lview or sd["kid.weight"].shape[0] == 2 * LV_SLOTS:
        return sd
    sd = dict(sd)
    for k in ("kid.weight", "kid.bias"):
        w = sd[k]
        new = torch.zeros((2 * LV_SLOTS,) + w.shape[1:], dtype=w.dtype, device=w.device)
        for half in range(2):
            o, n = half * SLOTS, half * LV_SLOTS
            new[n + LV_FATE_OFF : n + LV_FATE_OFF + 3] = w[o : o + 3]
            new[n + LV_MOVE_OFF : n + LV_MOVE_OFF + 37] = w[o : o + 37]
            new[n + LV_SPAWN_OFF : n + LV_SPAWN_OFF + 6] = w[o : o + 6]
        sd[k] = new
    w, bias = sd["vfc.2.weight"], sd["vfc.2.bias"]
    sd["vfc.2.weight"] = torch.cat([w, w, torch.zeros_like(w)], 0)
    sd["vfc.2.bias"] = torch.cat([bias, bias, torch.zeros_like(bias)], 0)
    w = sd["inp.0.weight"]
    c = w.shape[1] - SPARSE_PLANES - N_SCALARS
    old = lambda p: w[:, c + p]
    new = w.clone()
    # Old: 13 decided friend (at its destination), 14 decided enemy, 15
    # decided spawn, 16 next, 17 killed, 18 unsummoned, 19..24 gone units.
    # New: 13..16 fate stay/killed/unsummoned, 16 decided friend (origin),
    # 17..23 final friendly units, 23 the final unit is a spawn.
    new[:, c + 13] = old(14)
    new[:, c + 14] = old(14) + old(17)
    new[:, c + 15] = old(14) + old(18)
    new[:, c + 16] = 0
    for u in range(6):
        new[:, c + 17 + u] = old(13) + old(u)
    new[:, c + 23] = old(15)
    new[:, c + 24] = 0
    sd["inp.0.weight"] = new
    return sd


def dense_x(idx, scal):
    b = idx.shape[0]
    ones = torch.ones(b, MAX_ACTIVE, device=scal.device, dtype=scal.dtype)
    x = torch.zeros(b, SPARSE_DIM + 1, device=scal.device, dtype=scal.dtype)
    x = x.scatter(1, idx.long(), ones)[:, :SPARSE_DIM]
    return torch.cat([x, scal], 1)


class Tier1Export(nn.Module):
    def __init__(self, trunk, t2=None):
        super().__init__()
        self.trunk = trunk
        self.t2 = t2

    def forward(self, ridx, rscal):
        x = dense_x(ridx, rscal)
        h, v1 = self.trunk(x)
        if self.t2 is not None:
            # One value function: the tier-2 head's node value at the lattice
            # root (the turn start, nothing committed) -- the tier-1 call's
            # features are the root node's.
            out = self.t2(h, x)
            # --lview: its position value output instead.
            v1 = out[3][:, 0] if self.t2.lview else out[2]
        return torch.sigmoid(v1), h.half()


class Tier2Export(nn.Module):
    def __init__(self, t2):
        super().__init__()
        self.t2 = t2

    def forward(self, hs, hidx, nidx, nscal, kid):
        # hs: the call's distinct trunk images [U, C, 7, 7]; hidx [B] each
        # row's (one upload per turn start, not per row).
        # kid: flat [T] indices row * 49*37 + cell * 37 + slot (no padding).
        h = hs.index_select(0, hidx.long())
        logit, vlog, v2, extra = self.t2(h.float(), dense_x(nidx, nscal))
        k = kid.long()
        out = (logit.reshape(-1).gather(0, k), torch.sigmoid(vlog.reshape(-1).gather(0, k)), torch.sigmoid(v2))
        if self.t2.lview:
            out = out + (extra[:, 1],)
        return out


def export_onnx(trunk, t2, out, dev, value_prior_tau=0.0, t1_from_t2=False):
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
        Tier1Export(trunk, t2 if t1_from_t2 else None).to(dev).eval(),
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
        (
            torch.zeros(8, c, N, N, dtype=torch.float16, device=dev),
            torch.zeros(b, dtype=torch.int32, device=dev),
            pad.clone(),
            sc.clone(),
            torch.zeros(b * k, dtype=torch.int32, device=dev),
        ),
        os.path.join(out, "tier2.onnx"),
        input_names=["hs", "hidx", "nidx", "nscal", "kid"],
        output_names=["logit", "value", "v2"] + (["energy"] if t2.lview else []),
        dynamic_axes={
            "hs": {0: "u"},
            "hidx": {0: "b"},
            "nidx": {0: "b"},
            "nscal": {0: "b"},
            "kid": {0: "t"},
            "logit": {0: "t"},
            "value": {0: "t"},
            "v2": {0: "b"},
            "energy": {0: "b"},
        },
        opset_version=17,
        dynamo=False,
    )
    with open(os.path.join(out, "lattice.json"), "w") as f:
        meta = {"c_trunk": c, "trunk_gather": True}
        if t2.lview:
            meta["lview"] = True
        if value_prior_tau > 0:
            # The search's priors: children's values over this temperature.
            meta["value_prior_tau"] = value_prior_tau
        json.dump(meta, f)


def group_log_softmax(logit, group, n_groups):
    """log-softmax of `logit` within each group id."""
    m = torch.full((n_groups,), -1e30, device=logit.device).scatter_reduce(0, group, logit, "amax")
    e = torch.exp(logit - m[group])
    s = torch.zeros(n_groups, device=logit.device).index_add_(0, group, e)
    return logit - m[group] - torch.log(s[group])


class T2Set:
    """Records and kids on the device, with CSR offsets."""

    def __init__(self, recs, kids, dev, all_visit_targets=False, slots=SLOTS):
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
        self.kflat = t(kids["cell"], np.int64) * slots + t(kids["slot"], np.int64)
        self.kcell = t(kids["cell"], np.int64)
        self.kn = t(kids["n"], np.float32)
        self.kq = t(kids["q"], np.float32)
        # Value targets: sink-return means only (unless the older all-visit
        # values are asked for); records without one train the priors only.
        ok = (recs["sink"] == 1) | all_visit_targets
        self.qok = t(ok & (recs["q"] >= 0))
        self.kqok = t(np.repeat(ok, recs["kids"].astype(np.int64)) & (kids["q"] >= 0))
        self.stage = recs["stage"]
        self.n = len(recs)
        if slots == LV_SLOTS:
            # The old (37 shared slots) slot of each kid, for a teacher.
            off = np.array([LV_FATE_OFF, LV_MOVE_OFF, LV_SPAWN_OFF, LV_SPAWN_OFF])[np.repeat(recs["stage"], recs["kids"].astype(np.int64))]
            self.kslot_old = t(kids["slot"].astype(np.int64) - off)

    def set_old(self, old, dev):
        t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a if dt is None else a.astype(dt))).to(dev)
        self.oract = t(old["ract"].view(np.int16))
        self.orn = t(old["rn"])
        self.orscal = t(old["rscal"], np.float32)
        self.onact = t(old["nact"].view(np.int16))
        self.onn = t(old["nn"])
        self.onscal = t(old["nscal"], np.float32)

    def batch(self, idx):
        cnt = self.nkids[idx]
        rec_of = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), cnt)
        start = torch.repeat_interleave(self.off[idx], cnt)
        first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        kid = start + torch.arange(len(rec_of), device=idx.device) - first
        return rec_of, kid


def masked_bce(logit, y, m):
    if not bool(m.any()):
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], y[m])


def losses(trunk, t2, ds, idx, grad_trunk=False, counts=False):
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()):
        rx = dense(ds.ract[idx], ds.rn[idx], ds.rscal[idx])
        h, _ = trunk(rx)
    logit, vlog, v, _ = t2(h, dense(ds.nact[idx], ds.nn[idx], ds.nscal[idx]))
    rec_of, kid = ds.batch(idx)
    flat = ds.kflat[kid]
    lg = logit[rec_of, flat]
    vl = vlog[rec_of, flat]
    # Groups: (record in batch, cell).
    group = rec_of * CELLS + ds.kcell[kid]
    ng = len(idx) * CELLS
    lp = group_log_softmax(lg, group, ng)
    teach = getattr(ds, "teacher", None)
    if teach is not None:
        # An old-view teacher's outputs on the same nodes are the targets:
        # per-cell prior softmax over the offered kids, kid and node values,
        # and its tier-1 value for the position value at the root.
        tt, tt2 = teach
        with torch.no_grad():
            th, _ = tt(dense(ds.oract[idx], ds.orn[idx], ds.orscal[idx]))
            tlog, tvlog, tv, _ = tt2(th, dense(ds.onact[idx], ds.onn[idx], ds.onscal[idx]))
            oflat = ds.kcell[kid] * SLOTS + ds.kslot_old[kid]
            tl = tlog[rec_of, oflat]
            tp = torch.exp(group_log_softmax(tl, group, ng))
            tkv = torch.sigmoid(tvlog[rec_of, oflat])
            tnv = torch.sigmoid(tv)
            troot = torch.sigmoid(tt2(th, dense(ds.oract[idx], ds.orn[idx], ds.orscal[idx]))[2])
        n_cells = (torch.zeros(ng, device=lg.device).index_add_(0, group, torch.ones_like(tl)) > 0).sum().clamp(min=1)
        h_t = -(tp * torch.log(tp.clamp(min=1e-9))).sum() / n_cells
        l_prior = -(tp * lp).sum() / n_cells
        l_val = F.binary_cross_entropy_with_logits(vl, tkv)
        l_node = F.binary_cross_entropy_with_logits(v, tnv)
        pv = t2(h, rx)[3][:, 0]
        l_root = F.binary_cross_entropy_with_logits(pv, troot)
        ds.last_root = l_root
        if counts:
            return l_prior, l_val, l_node, h_t, len(kid), len(idx)
        return l_prior, l_val, l_node, h_t
    tot = torch.zeros(ng, device=lg.device).index_add_(0, group, ds.kn[kid])
    tgt = ds.kn[kid] / tot[group].clamp(min=1e-9)
    # Cells where fewer than min_support kids have visits carry no
    # comparison between choices (one-hot: the only one tried); skip them.
    supp = torch.zeros(ng, device=lg.device).index_add_(0, group, (ds.kn[kid] > 0).float())
    cell_ok = (tot > 0) & (supp >= getattr(ds, "min_support", 1))
    tgt = tgt * cell_ok[group]
    n_cells = cell_ok.sum().clamp(min=1)
    l_prior = -(tgt * lp).sum() / n_cells
    # Its floor: the targets' own entropy.
    h_tgt = -(tgt * torch.log(tgt.clamp(min=1e-9))).sum() / n_cells
    km, qm = ds.kqok[kid], ds.qok[idx]
    l_val = masked_bce(vl, ds.kq[kid], km)
    l_node = masked_bce(v, ds.q[idx], qm)
    if counts:
        return l_prior, l_val, l_node, h_tgt, int(km.sum()), int(qm.sum())
    return l_prior, l_val, l_node, h_tgt


def t1_loss(net, xs, ys, idx):
    return F.binary_cross_entropy_with_logits(net(xs.dense(idx)), ys[idx])


class RootValue(nn.Module):
    """Tier 1 as the tier-2 head's node value at the lattice root: trunk
    image, then the head on the same (turn-start) features."""

    def __init__(self, trunk, t2):
        super().__init__()
        self.trunk = trunk
        self.t2 = t2

    def forward(self, x):
        h, _ = self.trunk(x)
        out = self.t2(h, x)
        return out[3][:, 0] if self.t2.lview else out[2]


def load_energy(dirs):
    rs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "t1e", "chunk_*.bin"))):
            rs.append(read_chunk(p, b"MLTE", E_DTYPE))
    return np.concatenate(rs) if rs else np.zeros(0, E_DTYPE)


class ESet:
    """Energy groups: a turn start, then its tier-1 children (complete
    declarations) with their visits. Sparse rows on the device."""

    def __init__(self, recs, dev):
        kind = recs["kind"]
        start = np.flatnonzero(kind == 0)
        gid = np.cumsum(kind == 0) - 1
        child = np.flatnonzero((kind == 1) & (gid >= 0))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        self.act = t(recs["active"].view(np.int16))
        self.cnt = t(recs["n_active"])
        self.scal = t(recs["scalars"].astype(np.float32))
        self.visits = t(recs["visits"].astype(np.float32))
        self.root = t(start.astype(np.int64))
        # Children in CSR order per group.
        self.child = t(child.astype(np.int64))
        self.cgroup = t(gid[child].astype(np.int64))
        n = np.bincount(gid[child], minlength=len(start))
        self.n = len(start)
        self.nkids = t(n.astype(np.int64))
        self.off = torch.cumsum(self.nkids, 0) - self.nkids

    def x(self, rows):
        return dense(self.act[rows], self.cnt[rows], self.scal[rows])


def energy_loss(trunk, t2, es, g, grad_trunk=True):
    """Cross-entropy of the softmax over each group's children's energies
    against their visit shares; and its floor (the targets' entropy)."""
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()):
        h, _ = trunk(es.x(es.root[g]))
    cnt = es.nkids[g]
    grp = torch.repeat_interleave(torch.arange(len(g), device=g.device), cnt)
    first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    rows = es.child[torch.repeat_interleave(es.off[g], cnt) + torch.arange(len(grp), device=g.device) - first]
    e = t2(h[grp], es.x(rows))[3][:, 1]
    lp = group_log_softmax(e, grp, len(g))
    n = es.visits[rows]
    tot = torch.zeros(len(g), device=g.device).index_add_(0, grp, n)
    tgt = n / tot[grp].clamp(min=1e-9)
    ng = (tot > 0).sum().clamp(min=1)
    ce = -(tgt * lp).sum() / ng
    hf = -(tgt * torch.log(tgt.clamp(min=1e-9))).sum() / ng
    return ce, hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", help="lattice-gen output dirs (their t1/ and t2/)")
    ap.add_argument("--t1", help="tier-1 steps net.pt to start from (default: random init)")
    ap.add_argument("--t1-arch", default="res:32x4")
    ap.add_argument("--arch", default="32x2", help="tier-2 CHANNELSxBLOCKS")
    ap.add_argument("--init", help="tier-2 weights to start from (t2.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the tier-2 records")
    ap.add_argument("--batch", type=int, default=512, help="tier-2 records per step")
    ap.add_argument("--t1-batch", type=int, default=256, help="tier-1 records per step")
    ap.add_argument("--t1-weight", type=float, default=1.0)
    ap.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm statistics fixed (layers in eval mode) while training the weights")
    ap.add_argument("--bn-recal", type=int, default=200, help="re-estimate BatchNorm statistics exactly over this many batches after training (0 = keep the moving estimate)")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--t1-from-t2", action="store_true", help="no separate tier-1 value: tier 1 is the tier-2 head's node value at the lattice root (turn records train it there)")
    ap.add_argument("--value-prior-tau", type=float, default=0.0, help="export with value-based search priors at this temperature (lattice.json); the prior head is then unused")
    ap.add_argument("--lview", action="store_true", help="lattice-view records (lview.rs): per-kind output slots, a position-value output (tier 1 reads it) and an energy output trained on t1e/ groups")
    ap.add_argument("--teacher", help="an old-view lattice net dir (net.pt, t2.pt, meta.json): tier-2 targets are its outputs on the same nodes (records need --teacher-view .old files); the position value learns its tier-1 value")
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--energy-batch", type=int, default=64, help="energy groups per step")
    ap.add_argument("--min-cell-support", type=int, default=1, help="prior targets only for cells where at least this many kids have visits")
    ap.add_argument("--value-weight", type=float, default=1.0, help="tier-2 child and node values")
    ap.add_argument("--freeze-trunk", action="store_true", help="train tier 2 only, on a fixed tier 1")
    ap.add_argument("--t2-detach", action="store_true", help="tier-2 losses do not train the shared trunk (tier 1 learns from its own targets only)")
    ap.add_argument("--all-visit-targets", action="store_true", help="also train values on older records, whose values averaged every visit")
    ap.add_argument("--anchor", type=float, default=0.0, help="blend the trained weights toward the starting ones: w = a * w0 + (1 - a) * w")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--holdout-every", type=int, default=20)
    ap.add_argument("--export-only", action="store_true", help="export --t1/--init (or random weights) to --out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    net1 = build_t1(args.t1_arch).to(dev)
    if args.t1:
        sd = torch.load(args.t1, map_location=dev)
        if isinstance(net1, PosNet):
            sd = pos_from_res(sd)
        net1.load_state_dict(sd)
    trunk = Trunk(net1).to(dev)
    c_trunk = net1.stem[0].out_channels
    ch, blocks = (int(v) for v in args.arch.split("x"))
    t2 = T2Net(c_trunk, ch, blocks, args.lview).to(dev)
    if args.init:
        t2.load_state_dict(translate_t2(torch.load(args.init, map_location=dev), args.lview))
    v1net = RootValue(trunk, t2) if args.t1_from_t2 or args.lview else net1

    def save(meta):
        torch.save(net1.state_dict(), os.path.join(args.out, "net.pt"))
        torch.save(t2.state_dict(), os.path.join(args.out, "t2.pt"))
        export_onnx(trunk, t2, args.out, dev, args.value_prior_tau, args.t1_from_t2)
        # The tier-1 net alone, as a steps value net (steps-match etc.).
        steps_export_onnx(net1, args.out, dev)
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta | {"t1_arch": args.t1_arch, "arch": args.arch, "t1_from_t2": args.t1_from_t2, "lview": args.lview}, f, indent=1)

    if args.export_only:
        save({"t1": args.t1, "init": args.init, "export_only": True})
        print(f"exported to {args.out}")
        return
    t0 = time.time()
    if args.teacher:
        recs, kids, old = load_t2(args.data, old=True)
    else:
        recs, kids = load_t2(args.data)
    ds = T2Set(recs, kids, dev, args.all_visit_targets, t2.slots)
    if args.teacher:
        ds.set_old(old, dev)
        del old
        tm = json.load(open(os.path.join(args.teacher, "meta.json")))
        tn1 = build_t1(tm["t1_arch"]).to(dev)
        tn1.load_state_dict(torch.load(os.path.join(args.teacher, "net.pt"), map_location=dev))
        tch, tbl = (int(v) for v in tm["arch"].split("x"))
        tt2 = T2Net(tn1.stem[0].out_channels, tch, tbl).to(dev)
        tt2.load_state_dict(torch.load(os.path.join(args.teacher, "t2.pt"), map_location=dev))
        assert tm.get("t1_from_t2"), "teacher's tier-1 value must be its tier-2 root value"
        ds.teacher = (Trunk(tn1).eval(), tt2.eval())
    es = ESet(load_energy(args.data), dev) if args.lview else None
    if es is not None:
        eg = torch.arange(es.n, device=dev)
        hoe, tre = eg[eg % args.holdout_every == 0], eg[eg % args.holdout_every != 0]
        print(f"energy: {es.n} groups, {len(es.child)} children ({len(es.child) / max(es.n, 1):.1f}/group)")
    ds.min_support = args.min_cell_support
    all_idx = torch.arange(ds.n, device=dev)
    ho_mask = (all_idx % args.holdout_every) == 0
    ho, tr = all_idx[ho_mask], all_idx[~ho_mask]
    r1, _, nf1 = load_chunks([os.path.join(d, "t1") for d in args.data])
    keep = turn_records(r1)
    print(f"t1: {len(r1)} records, {keep.sum()} turn starts kept")
    r1 = r1[keep]
    xs1 = SparseSet(r1, dev)
    ys1 = torch.from_numpy(r1["target"].astype(np.float32)).to(dev)
    i1 = torch.arange(len(r1), device=dev)
    ho1, tr1 = i1[i1 % args.holdout_every == 0], i1[i1 % args.holdout_every != 0]
    print(
        f"t2: {ds.n} records ({len(kids)} kids, {len(kids) / max(ds.n, 1):.1f}/record), {len(tr)} train; "
        f"t1: {len(r1)} records ({nf1} chunks, visits median {int(np.median(r1['visits'])) if len(r1) else 0}), "
        f"target mean {r1['target'].mean():.3f} std {r1['target'].std():.3f}; load {time.time() - t0:.0f}s"
    )
    print("t2 stages:", np.bincount(ds.stage, minlength=4).tolist())
    print(
        f"value targets: {ds.qok.float().mean().item():.3f} of records, "
        f"{ds.kqok.float().mean().item():.3f} of kids (sink-return records {(recs['sink'] == 1).mean():.3f})"
    )

    def holdout():
        trunk.eval()
        t2.eval()
        out = np.zeros(5)
        with torch.no_grad():
            # Value losses weighted by their (masked) target counts.
            wsum = np.zeros(4)
            for i in range(0, len(ho), 4096):
                lp, lv, ln, h, nk, nq = losses(trunk, t2, ds, ho[i : i + 4096], counts=True)
                w = np.array([len(ho[i : i + 4096]), nk, nq, len(ho[i : i + 4096])], dtype=float)
                out[:4] += np.array([lp.item(), lv.item(), ln.item(), h.item()]) * w
                wsum += w
            out[:4] /= np.maximum(wsum, 1)
            for i in range(0, len(ho1), 8192):
                out[4] += t1_loss(v1net, xs1, ys1, ho1[i : i + 8192]).item() * len(ho1[i : i + 8192])
            out[4] /= max(len(ho1), 1)
            # The t1 targets' own entropy: the floor of their BCE.
            y = ys1[ho1].clamp(1e-6, 1 - 1e-6)
            h1 = -(y * y.log() + (1 - y) * (1 - y).log()).mean().item() if len(ho1) else 0.0
            ex = {}
            if es is not None and len(hoe):
                ce = hf = 0.0
                for i in range(0, len(hoe), 512):
                    c_, h_ = energy_loss(trunk, t2, es, hoe[i : i + 512])
                    ce += c_.item() * len(hoe[i : i + 512])
                    hf += h_.item() * len(hoe[i : i + 512])
                ex = {"energy": ce / len(hoe), "energy_excess": (ce - hf) / len(hoe)}
        trunk.train()
        t2.train()
        o = out
        return {"prior": o[0], "prior_excess": o[0] - o[3], "value": o[1], "node": o[2], "t1_bce_excess": o[4] - h1} | ex

    base = holdout()
    print("holdout before:", json.dumps(base))
    w0 = {k: v.detach().clone() for k, v in list(net1.state_dict().items()) + [("t2." + k, v) for k, v in t2.state_dict().items()]}
    if args.freeze_trunk:
        for p in trunk.parameters():
            p.requires_grad_(False)
        params = list(t2.parameters())
    else:
        params = list(t2.parameters()) + list(net1.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = max(int(args.epochs * len(tr) / args.batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    trunk.train(not args.freeze_trunk)
    t2.train()
    bn_all = [m for m in list(net1.modules()) + list(t2.modules()) if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if args.freeze_bn:
        for m in bn_all:
            m.eval()
    t0 = time.time()
    for step in range(steps):
        idx = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        lp, lv, ln, _ = losses(trunk, t2, ds, idx, grad_trunk=not (args.freeze_trunk or args.t2_detach))
        if args.teacher:
            l1 = ds.last_root
        elif args.freeze_trunk:
            l1 = torch.zeros((), device=dev)
        else:
            l1 = t1_loss(v1net, xs1, ys1, tr1[torch.randint(len(tr1), (args.t1_batch,), device=dev)]) if len(tr1) else torch.zeros((), device=dev)
        le = torch.zeros((), device=dev)
        if es is not None and len(tre):
            le, _ = energy_loss(trunk, t2, es, tre[torch.randint(len(tre), (args.energy_batch,), device=dev)], not (args.freeze_trunk or args.t2_detach))
        loss = args.prior_weight * lp + args.value_weight * (lv + ln) + args.t1_weight * l1 + args.energy_weight * le
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"step {step}/{steps} prior {lp.item():.4f} value {lv.item():.4f} node {ln.item():.4f} t1 {l1.item():.4f} energy {le.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    if args.bn_recal > 0 and not args.freeze_bn:
        # BatchNorm's moving estimate (momentum 0.1) follows the last few
        # dozen batches; a rare feature's channel then gets a near-zero
        # variance and the net breaks at inference (2026-10-05: training at
        # lr 1e-8 alone cost gen 45 0.30 vs itself). Re-estimate exactly:
        # a cumulative average over many batches of both streams.
        bns = [m for m in list(net1.modules()) + list(t2.modules()) if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        trunk.train()
        t2.train()
        with torch.no_grad():
            for _ in range(args.bn_recal):
                idx = tr[torch.randint(len(tr), (args.batch,), device=dev)]
                losses(trunk, t2, ds, idx)
                if len(tr1):
                    v1net(xs1.dense(tr1[torch.randint(len(tr1), (args.t1_batch,), device=dev)]))
                if es is not None and len(tre):
                    energy_loss(trunk, t2, es, tre[torch.randint(len(tre), (args.energy_batch,), device=dev)])
        for m in bns:
            m.momentum = 0.1
        print(f"BatchNorm statistics re-estimated over {args.bn_recal} batches")
    if args.anchor > 0:
        a = args.anchor
        with torch.no_grad():
            for m, pre in ((net1, ""), (t2, "t2.")):
                sd = m.state_dict()
                for k, v in sd.items():
                    if v.dtype.is_floating_point:
                        v.mul_(1 - a).add_(w0[pre + k], alpha=a)
    final = holdout()
    print("holdout after:", json.dumps(final))
    save(
        {
            "t1": args.t1,
            "init": args.init,
            "data": args.data,
            "records": ds.n,
            "t1_records": len(r1),
            "holdout_before": base,
            "holdout": final,
            "steps": steps,
            "train_seconds": time.time() - t0,
        }
    )
    print("final", json.dumps({"holdout": final, "records": ds.n, "t1_records": len(r1), "steps": steps}))


if __name__ == "__main__":
    main()
