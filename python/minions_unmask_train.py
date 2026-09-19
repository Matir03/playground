"""Train the unmasking-consistency objective on its own, from the replay buffer.

This is the standalone counterpart to `--w-unmask` in the learner: the same loss,
with nothing else in the objective. It exists to answer one question before the
signal is built on — **does a committed field's output actually stop echoing the
committed value?** A loss that falls proves the term computes; it does not prove
the model learned anything usable, because "predict the value that is sitting in
your input" is a zero-loss solution for a differently-shaped objective and a
near-zero-loss attractor for this one.

So the metric that matters here is not the loss. It is `p_committed`: the
probability pass B puts on the value that is committed at the supervised field.
Near 1.0 means the head is echoing its input and the reroll signal would be
vacuous. It should fall toward `p_committed` under pass A — the probability the
model assigns that same value when it *cannot* see it.

Defaults, and why:

* **The trunk is frozen.** Only the diffusion head trains. The trunk is shared
  with value, energy, score and the auxiliaries; fitting it to this one
  objective in isolation would damage every other head, and those are what the
  live run depends on. `--unfreeze-trunk` lifts this.
* **Nothing is written to the source directory.** Output goes to `--out`, which
  gets its own checkpoint, config and ONNX export, so the live run's weights are
  untouched and the result can be served or discarded independently.
* **Totally-masked inputs are sampled explicitly.** `--total-frac` of rows use
  ratio 1.0, where pass A has *nothing* committed and pass B exactly one field.
  That is the purest form of the objective and it has measure zero under a plain
  uniform ratio, so it has to be drawn deliberately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from playground.model_server.minions_net import (  # noqa: E402
    CELLS, IN_PLANES, N, arch_of, build_model, export_onnx_atomic,
)
from minions_az_learner import (  # noqa: E402
    Buffer, C_MOVE, C_SPAWN, C_TARGET, decl_planes, sample_target_spec, save_atomic,
)


def build_masks(live_f, full_rows, rng, total_frac):
    """-> (hidden, committed). `total_frac` of rows are fully masked."""
    bs = live_f.shape[0]
    ratio = rng.random((bs, 1, 1))
    # Totally masked: every live field hidden, nothing committed in pass A.
    ratio[rng.random((bs, 1, 1)) < total_frac] = 1.0
    hidden = live_f & (rng.random(live_f.shape) < ratio) & full_rows[:, None, None]
    return hidden, live_f & ~hidden


def step(net, hh, spec, live_f, hidden, committed, rng, device):
    """One unmasking-consistency step -> (loss, diagnostics dict)."""
    bs = live_f.shape[0]
    decl_a = torch.from_numpy(decl_planes(spec, committed)).to(device)
    mv_a, tg_a, sp_a = net.diffusion_head(hh, decl_a)

    flat_hidden = hidden.reshape(bs, -1)
    has = flat_hidden.any(axis=1)
    if not has.any():
        return None, {}
    # Uniform among each row's hidden slots: hidden land in [-1, 0), rest [0, 1).
    jsel = np.argmin(rng.random(flat_hidden.shape) - flat_hidden.astype(np.float64), axis=1)
    rows = np.flatnonzero(has)
    committed_b = committed.reshape(bs, -1).copy()
    # Only rows that have a hidden slot; elsewhere jsel is meaningless and
    # writing it would un-commit a real field.
    committed_b[rows, jsel[rows]] = True
    decl_b = torch.from_numpy(
        decl_planes(spec, committed_b.reshape(live_f.shape))
    ).to(device)
    mv_b, tg_b, sp_b = net.diffusion_head(hh, decl_b)

    cell, field = jsel // 3, jsel % 3
    # The value committed at the supervised field, per field kind.
    vals = np.stack(
        [spec[:, :CELLS], spec[:, CELLS:2 * CELLS], spec[:, 2 * CELLS:]], axis=2
    )
    ces, p_com_b, p_com_a, ent_a = [], [], [], []
    for fi, (la, lb) in enumerate(((mv_a, mv_b), (tg_a, tg_b), (sp_a, sp_b))):
        sel = np.flatnonzero(has & (field == fi))
        if not len(sel):
            continue
        st = torch.from_numpy(sel).to(device)
        ct = torch.from_numpy(cell[sel]).to(device)
        k = la.shape[-1]
        # detach: pass A's belief is the target, held fixed. Without it both
        # sides drift together and the pair can agree on anything at all.
        pa = F.softmax(la[st, ct].detach(), dim=-1)
        logpb = F.log_softmax(lb[st, ct], dim=-1)
        ces.append(-(pa * logpb).sum(dim=-1))
        v = torch.from_numpy(
            np.clip(vals[sel, cell[sel], fi], 0, k - 1).astype(np.int64)
        ).to(device)
        # The echo diagnostic: how much mass B puts on the value it can see,
        # against how much A puts there without seeing it.
        p_com_b.append(logpb.exp().gather(1, v[:, None]).squeeze(1))
        p_com_a.append(pa.gather(1, v[:, None]).squeeze(1))
        ent_a.append(-(pa * torch.log(pa.clamp_min(1e-12))).sum(dim=-1))
    if not ces:
        return None, {}
    loss = torch.cat(ces).mean()
    with torch.no_grad():
        diag = {
            "p_committed_b": float(torch.cat(p_com_b).mean()),
            "p_committed_a": float(torch.cat(p_com_a).mean()),
            "entropy_a": float(torch.cat(ent_a).mean()),
            "n": int(has.sum()),
        }
    return loss, diag


def main() -> None:
    ap = argparse.ArgumentParser(description="train the unmasker alone")
    ap.add_argument("--live", default="data/models/minions/live",
                    help="source of weights, config and the replay buffer; never written to")
    ap.add_argument("--out", required=True, help="where the trained net is written")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--total-frac", type=float, default=0.25,
                    help="fraction of rows drawn fully masked (pass A sees nothing)")
    ap.add_argument("--unfreeze-trunk", action="store_true",
                    help="also train the trunk. Off by default: the trunk is shared "
                         "with value/energy/score/aux and fitting it to this objective "
                         "alone would damage them.")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    # The buffer has been destroyed twice by a trainer pointed at the live dir.
    if os.path.abspath(args.out) == os.path.abspath(args.live):
        raise SystemExit("--out must not be --live; this run never writes into the live run")

    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cfg = json.load(open(os.path.join(args.live, "config.json")))
    arch = cfg["arch"]
    # build_model, not MinionsNet(...): the arch dict carries `proposer`, and a
    # net built without it has no diffusion head to train.
    net = build_model(arch).to(device)
    missing, unexpected = net.load_state_dict(
        load_file(os.path.join(args.live, "latest.safetensors")), strict=False
    )
    if missing or unexpected:
        raise SystemExit(f"weight mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"loaded generation {cfg.get('generation')} from {args.live}")

    names = {n_ for n_, _ in net.named_parameters()
             if args.unfreeze_trunk or n_.startswith("dif_")}
    for n_, p in net.named_parameters():
        p.requires_grad_(n_ in names)
    params = [p for n_, p in net.named_parameters() if n_ in names]
    print(f"training {len(params)} tensors ({sum(p.numel() for p in params)} params)"
          f"{'' if args.unfreeze_trunk else ', trunk frozen'}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    buf = Buffer(os.path.join(args.live, "buffer"))
    size = buf.refresh()
    batch = buf.merged()
    n = len(batch["value"])
    print(f"buffer: {size} examples, {n} usable rows")

    # Mode matters more than requires_grad here. `no_grad` blocks gradients but
    # NOT BatchNorm running-stat updates, so a trunk left in train() would drift
    # its running_mean/var on every step and silently move the value, energy and
    # score heads this run is supposed to leave untouched.
    if args.unfreeze_trunk:
        net.train()
    else:
        net.eval()
        for m in (net.dif_in, net.dif_bn, net.dif_block, net.dif_gpool, net.dif_out):
            m.train()

    t0 = time.time()
    run = []
    for it in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.batch)
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).to(device)
        planes = planes.view(args.batch, IN_PLANES, N, N)
        if args.unfreeze_trunk:
            hh = net.head_features(net.trunk(planes))
        else:
            with torch.no_grad():
                hh = net.head_features(net.trunk(planes))
        spec = sample_target_spec(batch, idx, rng)
        live = batch["live"][idx]
        live_f = np.stack(
            [(live & C_MOVE) > 0, (live & C_TARGET) > 0, (live & C_SPAWN) > 0], axis=2
        )
        full_rows = batch["is_full"][idx].astype(bool)
        hidden, committed = build_masks(live_f, full_rows, rng, args.total_frac)
        loss, diag = step(net, hh, spec, live_f, hidden, committed, rng, device)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append((float(loss.detach()), diag["p_committed_b"], diag["p_committed_a"],
                    diag["entropy_a"]))
        if it % args.log_every == 0:
            a = np.array(run[-args.log_every:])
            print(
                f"step {it:5}/{args.steps}  loss {a[:, 0].mean():.4f}  "
                f"p_committed B {a[:, 1].mean():.3f} (A {a[:, 2].mean():.3f})  "
                f"H(A) {a[:, 3].mean():.3f}  {time.time() - t0:.0f}s",
                flush=True,
            )

    os.makedirs(args.out, exist_ok=True)
    save_atomic(net, os.path.join(args.out, "latest.safetensors"))
    export_onnx_atomic(net, args.out)
    json.dump({"arch": arch_of(net), "generation": cfg.get("generation", 0),
               "unmask_trained_steps": args.steps},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    a = np.array(run[-args.log_every:])
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    print(f"final: loss {a[:, 0].mean():.4f}  p_committed B {a[:, 1].mean():.3f} "
          f"vs A {a[:, 2].mean():.3f}")
    print("B near 1.0 means the head is still echoing its input; it should sit near A.")


if __name__ == "__main__":
    main()
