"""Distil a Minions net into a smaller student, and measure how well it fits.

The teacher is a trained checkpoint (a live snapshot dir: `checkpoint.safetensors`
+ `config.json`). The student is any `MinionsNet` architecture -- trunk width and
depth, and the diffusion / energy heads' width and depth independently. The
student is trained to reproduce the teacher's *outputs* on replay-buffer
positions, never the buffer's own targets:

- value: MSE to the teacher's value
- diffusion: KL(teacher || student) over every live field, masked *and*
  committed -- the committed-field (unmasking) behaviour is part of the net
- energy: KL between the two softmaxes over each position's recorded children,
  which is what the search's priors are made of
- aux / score: light MSE to the teacher's logits, so the heads the ladder keeps
  training start somewhere sensible

Fit is reported on a held-out slice with fixed masks, per head, including the
agreement numbers that decide play: top-1 move agreement at masked fields and
top-1 energy agreement over children.

Writes the student as a snapshot dir (`checkpoint.safetensors`, `config.json`,
the ONNX graphs, `distill.json`), so it can be benchmarked and matched directly.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
from minions_az_learner import (  # noqa: E402
    C_MOVE, C_SPAWN, C_TARGET, CELLS, IN_PLANES, N, Buffer, candidate_rows,
    decl_planes, energy_planes, sample_target_spec,
)
from playground.model_server.minions_net import (  # noqa: E402
    MinionsNet, arch_of, build_model, export_onnx_atomic,
)


def live_fields(batch, idx):
    live = batch["live"][idx]
    return np.stack(
        [(live & C_MOVE) > 0, (live & C_TARGET) > 0, (live & C_SPAWN) > 0], axis=2
    )


def field_kl(t_logits, s_logits, mask):
    """Mean KL(teacher || student) over the positions in `mask` (B, CELLS)."""
    if not bool(mask.any()):
        return t_logits.sum() * 0, 0
    tl = t_logits[mask].float()
    sl = s_logits[mask].float()
    tp = F.log_softmax(tl, dim=-1)
    sp = F.log_softmax(sl, dim=-1)
    return (tp.exp() * (tp - sp)).sum(-1).mean(), int(mask.sum())


def run_heads(net, planes, spec, committed, batch, idx, live_f, device, max_children):
    """Every output the student is matched on, for one batch."""
    h = net.trunk(planes)
    out = {"v": net.value_head(h)}
    hh = net.head_features(h)
    out["hh"] = hh
    decl = torch.from_numpy(decl_planes(spec, committed)).to(device)
    out["mv"], out["tg"], out["sp"] = net.diffusion_head(hh, decl)
    out["hex"], out["necro"] = net.aux_head(h)
    out["score"] = net.score_head(h)
    n_c = batch["n_cands"][idx].astype(np.int64)
    use = np.flatnonzero(n_c > 0)
    # Whole rows up to the cap, so every softmax is over a complete child set.
    if len(use) and max_children:
        keep = np.cumsum(n_c[use]) <= max_children
        keep[0] = True
        use = use[keep]
    out["use"] = use
    if len(use):
        row, flat, slot = candidate_rows(batch, idx, use)
        edecl = torch.from_numpy(energy_planes(batch["cand_spec"][flat], live_f[row])).to(device)
        e = net.energy_head(hh[torch.from_numpy(row).to(device)], edecl).view(-1)
        dense = torch.from_numpy(np.searchsorted(use, row)).to(device)
        slot_t = torch.from_numpy(slot).to(device)
        scores = e.new_full((len(use), int(n_c[use].max())), -1e4)
        out["energy"] = scores.index_put((dense, slot_t), e.float())
    else:
        out["energy"] = None
    return out


def losses(t, s, live_f, hidden, device):
    lf = torch.from_numpy(live_f).to(device)
    hid = torch.from_numpy(hidden).to(device)
    lv = F.mse_loss(s["v"].float(), t["v"].float())
    kl_d = 0.0
    parts = {}
    for fi, key in enumerate(("mv", "tg", "sp")):
        k, _ = field_kl(t[key], s[key], lf[:, :, fi])
        kl_d = kl_d + k
        parts[key] = k
    if t["energy"] is not None:
        tp = F.log_softmax(t["energy"], dim=1)
        sp = F.log_softmax(s["energy"], dim=1)
        kl_e = (tp.exp() * (tp - sp)).sum(1).mean()
    else:
        kl_e = s["v"].sum() * 0
    aux = (
        F.mse_loss(s["hex"].float(), t["hex"].float())
        + F.mse_loss(s["necro"].float(), t["necro"].float())
        + F.mse_loss(s["score"].float(), t["score"].float())
    )
    return lv, kl_d, kl_e, aux, parts


@torch.no_grad()
def evaluate(teacher, student, batch, hold, device, bs, max_children):
    """Held-out fit with fixed masks (seeded per call), per head."""
    student.eval()
    rng = np.random.default_rng(12345)
    acc = {k: 0.0 for k in ("v_mse", "kl_mv_hid", "kl_mv_com", "kl_tg", "kl_sp",
                            "kl_e", "top1_mv", "top1_e")}
    n = {k: 0 for k in acc}
    vt_all, vs_all = [], []
    for i in range(0, len(hold), bs):
        idx = hold[i:i + bs]
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).to(device)
        planes = planes.view(len(idx), IN_PLANES, N, N)
        live_f = live_fields(batch, idx)
        spec = sample_target_spec(batch, idx, rng)
        full = batch["n_cands"][idx] > 0
        ratio = rng.random((len(idx), 1, 1))
        hidden = live_f & (rng.random(live_f.shape) < ratio) & full[:, None, None]
        committed = live_f & ~hidden
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = run_heads(teacher, planes, spec, committed, batch, idx, live_f, device, max_children)
            s = run_heads(student, planes, spec, committed, batch, idx, live_f, device, max_children)
        vt_all.append(t["v"].float().cpu())
        vs_all.append(s["v"].float().cpu())
        acc["v_mse"] += F.mse_loss(s["v"].float(), t["v"].float(), reduction="sum").item()
        n["v_mse"] += len(idx)
        hid = torch.from_numpy(hidden).to(device)
        com = torch.from_numpy(committed).to(device)
        for key, mask, name in (("mv", hid[:, :, 0], "kl_mv_hid"), ("mv", com[:, :, 0], "kl_mv_com"),
                                ("tg", hid[:, :, 1], "kl_tg"), ("sp", hid[:, :, 2], "kl_sp")):
            k, cnt = field_kl(t[key], s[key], mask)
            acc[name] += float(k) * cnt
            n[name] += cnt
        m = hid[:, :, 0]
        if bool(m.any()):
            agree = (t["mv"][m].argmax(-1) == s["mv"][m].argmax(-1)).float()
            acc["top1_mv"] += agree.sum().item()
            n["top1_mv"] += int(m.sum())
        if t["energy"] is not None:
            tp = F.log_softmax(t["energy"], dim=1)
            sp = F.log_softmax(s["energy"], dim=1)
            acc["kl_e"] += (tp.exp() * (tp - sp)).sum(1).sum().item()
            n["kl_e"] += t["energy"].shape[0]
            acc["top1_e"] += (t["energy"].argmax(1) == s["energy"].argmax(1)).float().sum().item()
            n["top1_e"] += t["energy"].shape[0]
    res = {k: acc[k] / max(n[k], 1) for k in acc}
    vt = torch.cat(vt_all)
    vs = torch.cat(vs_all)
    res["v_corr"] = float(torch.corrcoef(torch.stack([vt, vs]))[0, 1])
    student.train()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="snapshot dir of the teacher")
    ap.add_argument("--buffer", required=True, help="replay buffer dir (read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dif-ch", type=int, default=None)
    ap.add_argument("--dif-blocks", type=int, default=1)
    ap.add_argument("--eng-ch", type=int, default=None)
    ap.add_argument("--energy-blocks", type=int, default=2)
    ap.add_argument("--init", default=None,
                    help="warm-start from this student dir's checkpoint (same arch)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-children", type=int, default=4096)
    ap.add_argument("--holdout", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--w-aux", type=float, default=0.1)
    ap.add_argument("--w-feat", type=float, default=0.0,
                    help="hint loss: a 1x1 projection of the student's head features "
                         "regressed onto the teacher's (FitNets)")
    ap.add_argument("--copy-teacher", action="store_true",
                    help="start every student tensor whose name and shape match the "
                         "teacher's from the teacher's value")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tcfg = json.load(open(os.path.join(args.teacher, "config.json")))
    teacher = build_model(tcfg["arch"]).to(device)
    teacher.load_state_dict(load_file(os.path.join(args.teacher, "checkpoint.safetensors")))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = MinionsNet(args.channels, args.blocks, args.energy_blocks,
                         dif_ch=args.dif_ch, dif_blocks=args.dif_blocks,
                         eng_ch=args.eng_ch).to(device)
    if args.init:
        student.load_state_dict(load_file(os.path.join(args.init, "checkpoint.safetensors")))
    elif args.copy_teacher:
        tsd, ssd = teacher.state_dict(), student.state_dict()
        same = [k for k in ssd if k in tsd and tsd[k].shape == ssd[k].shape]
        student.load_state_dict({**ssd, **{k: tsd[k].clone() for k in same}})
        print(f"copied {len(same)} of {len(ssd)} tensors from the teacher", flush=True)
    proj = None
    if args.w_feat > 0:
        proj = torch.nn.Conv2d(student.head_ch, teacher.head_ch, 1).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"student {arch_of(student)}  {n_params:,} params "
          f"(teacher {sum(p.numel() for p in teacher.parameters()):,})", flush=True)

    buf = Buffer(args.buffer)
    buf.refresh()
    batch = buf.merged()
    total = len(batch["value"])
    perm = np.random.default_rng(999).permutation(total)
    hold = np.sort(perm[:args.holdout])
    train = perm[args.holdout:]
    print(f"{total:,} rows: {len(train):,} train, {len(hold):,} held out", flush=True)

    params = list(student.parameters()) + (list(proj.parameters()) if proj else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = train[rng.integers(0, len(train), size=args.batch)]
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).to(device)
        planes = planes.view(len(idx), IN_PLANES, N, N)
        live_f = live_fields(batch, idx)
        spec = sample_target_spec(batch, idx, rng)
        full = batch["n_cands"][idx] > 0
        ratio = rng.random((len(idx), 1, 1))
        hidden = live_f & (rng.random(live_f.shape) < ratio) & full[:, None, None]
        committed = live_f & ~hidden
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t = run_heads(teacher, planes, spec, committed, batch, idx, live_f, device,
                              args.max_children)
            s = run_heads(student, planes, spec, committed, batch, idx, live_f, device,
                          args.max_children)
        lv, kl_d, kl_e, aux, _ = losses(t, s, live_f, hidden, device)
        # Value is the only head the search backs up, so it is weighted up to
        # sit on the same scale as the KL terms (value MSE is ~1e-3 once fit).
        loss = 10.0 * lv + kl_d + kl_e + args.w_aux * aux
        if proj is not None:
            loss = loss + args.w_feat * F.mse_loss(proj(s["hh"].float()), t["hh"].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step}  loss {loss.item():.4f}  v {lv.item():.5f}  "
                  f"kl_dif {float(kl_d):.4f}  kl_e {float(kl_e):.4f}  "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(teacher, student, batch, hold, device, 512, args.max_children)
            res["step"] = step
            history.append(res)
            print("  eval " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "checkpoint.safetensors.tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}, tmp)
    os.replace(tmp, os.path.join(args.out, "checkpoint.safetensors"))
    json.dump({"arch": arch_of(student), "generation": int(tcfg.get("generation", 0))},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    export_onnx_atomic(student, args.out)
    json.dump({"arch": arch_of(student), "params": n_params, "steps": args.steps,
               "seconds": round(time.time() - t0, 1), "history": history},
              open(os.path.join(args.out, "distill.json"), "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
