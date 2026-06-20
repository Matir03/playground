"""Continuous Amazons AlphaZero learner (AmazonsAZNet).

Async actor/learner like go9_az_learner.py, but the policy is the factored two-head
target (queen + arrow), stored SPARSE in the actor's dump (dense 2×7200 would be
~57 KB/example). Loss = value MSE + queen CE + arrow CE. From scratch: writes a
random gen-0 net (+config.json + model.onnx) so the actor can start.

Pair with:
  actor: amazons-engine selfplay-loop --live data/models/amazons_az/live ...
  eval : amazons_eval_loop.py --live data/models/amazons_az/live
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.amazons_net import (
    AmazonsAZNet,
    PER_HEAD,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)

PLANES = 4 * 100  # floats per encoded position
_PAIR = np.dtype([("idx", "<u4"), ("prob", "<f4")])


def read_examples(path: str) -> dict:
    """Parse the actor's sparse dump into arrays + CSR-style sparse head targets."""
    with open(path, "rb") as f:
        buf = f.read()
    (count,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    planes = np.empty((count, 4, 10, 10), dtype=np.float32)
    value = np.empty(count, dtype=np.float32)
    q_idx, q_prob, q_cnt = [], [], np.empty(count, np.int64)
    a_idx, a_prob, a_cnt = [], [], np.empty(count, np.int64)
    for i in range(count):
        planes[i] = np.frombuffer(buf, "<f4", PLANES, off).reshape(4, 10, 10)
        off += PLANES * 4
        (value[i],) = struct.unpack_from("<f", buf, off)
        off += 4
        for cnt_arr, idx_l, prob_l in ((q_cnt, q_idx, q_prob), (a_cnt, a_idx, a_prob)):
            (n,) = struct.unpack_from("<I", buf, off)
            off += 4
            cnt_arr[i] = n
            if n:
                arr = np.frombuffer(buf, _PAIR, n, off)
                idx_l.append(arr["idx"])
                prob_l.append(arr["prob"])
                off += n * 8
    cat_u = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.uint32)
    cat_f = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.float32)
    return {
        "planes": planes,
        "value": value,
        "q_cnt": q_cnt, "q_idx": cat_u(q_idx), "q_prob": cat_f(q_prob),
        "a_cnt": a_cnt, "a_idx": cat_u(a_idx), "a_prob": cat_f(a_prob),
    }


def merge(chunks: list[dict]) -> dict:
    """Concatenate chunk dicts; rebuild CSR offsets from per-example counts."""
    out = {}
    out["planes"] = np.concatenate([c["planes"] for c in chunks])
    out["value"] = np.concatenate([c["value"] for c in chunks])
    for h in ("q", "a"):
        cnt = np.concatenate([c[f"{h}_cnt"] for c in chunks])
        out[f"{h}_off"] = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        out[f"{h}_idx"] = np.concatenate([c[f"{h}_idx"] for c in chunks])
        out[f"{h}_prob"] = np.concatenate([c[f"{h}_prob"] for c in chunks])
    return out


def scatter(b, off, idx, prob, dim, device):
    """Dense (B,dim) target from CSR slices for the sampled examples `b`."""
    starts, ends = off[b], off[b + 1]
    lens = ends - starts
    B = len(b)
    t = torch.zeros(B, dim, device=device)
    total = int(lens.sum())
    if total == 0:
        return t
    rows = np.repeat(np.arange(B), lens)
    pos = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends) if e > s])
    t[torch.from_numpy(rows).to(device),
      torch.from_numpy(idx[pos].astype(np.int64)).to(device)] = \
        torch.from_numpy(prob[pos]).to(device)
    return t


def save_atomic(net: AmazonsAZNet, path: str) -> None:
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu") for k, v in net.state_dict().items()}
    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)


def export_onnx_atomic(net: AmazonsAZNet, path: str) -> None:
    was_training = net.training
    net.eval()
    dev = next(net.parameters()).device
    tmp = path + ".tmp"
    torch.onnx.export(
        net, torch.zeros(1, 4, 10, 10, device=dev), tmp,
        input_names=["planes"], output_names=["value", "queen", "arrow"],
        dynamic_axes={k: {0: "b"} for k in ("planes", "value", "queen", "arrow")},
        opset_version=17, dynamo=False,
    )
    os.replace(tmp, path)
    if was_training:
        net.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/amazons_az/live")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-snapshot", type=int, default=300)
    ap.add_argument("--max-reuse", type=float, default=4.0)
    # Bursty training: stay idle until this many FRESH self-play examples accumulate,
    # then train the whole generation in one burst so the actor runs uncontended in
    # between. ~steps_per_snapshot*batch/max_reuse keeps the effective reuse at max.
    # 0 disables the gate (cadence is then driven externally, e.g. the orchestrator).
    ap.add_argument("--gen-examples", type=int, default=20000)
    # Exit after producing this many snapshots (0 = run forever). The orchestrator
    # uses 1 so the learner is a fresh, short-lived process per generation.
    ap.add_argument("--exit-after-gens", type=int, default=0)
    ap.add_argument("--eval-lead", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    live = args.live
    buffer_dir = os.path.join(live, "buffer")
    latest = os.path.join(live, "latest.safetensors")
    snaps = os.path.join(live, "snapshots")
    cfg_path = os.path.join(live, "config.json")
    os.makedirs(snaps, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    net = AmazonsAZNet(channels=args.channels, blocks=args.blocks)
    if os.path.exists(latest):
        load_checkpoint(net, latest)
        print(f"seeded from {latest}", flush=True)
    else:
        save_atomic(net, latest)
        with open(cfg_path, "w") as f:
            json.dump({"arch": arch_of(net), "generation": 0}, f, indent=2)
        print(f"from scratch: wrote random gen-0 {latest}", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # Persist optimizer (Adam moment) state across runs so killing/relaunching the
    # learner each generation doesn't reset the adaptive LR every time.
    opt_path = os.path.join(live, "opt_state.pt")
    if os.path.exists(opt_path):
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"loaded optimizer state from {opt_path}", flush=True)
        except Exception as e:
            print(f"optimizer state load failed (fresh Adam): {e}", flush=True)

    def save_opt():
        tmp = opt_path + ".tmp"
        torch.save(opt.state_dict(), tmp)
        os.replace(tmp, opt_path)

    onnx_path = os.path.join(live, "model.onnx")
    export_onnx_atomic(net, onnx_path)

    anchor_gen = int(json.load(open(cfg_path)).get("generation", 0))
    elo_path = os.path.join(live, "elo.json")

    cache: dict[str, dict] = {}
    big: dict | None = None
    produced = 0
    consumed = 0

    def refresh() -> bool:
        nonlocal produced
        files = set(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))
        changed = False
        for f in files - cache.keys():
            try:
                ex = read_examples(f)
            except Exception:
                continue
            cache[f] = ex
            produced += ex["planes"].shape[0]
            changed = True
        for f in list(cache.keys() - files):
            del cache[f]
            changed = True
        return changed

    def processed_through() -> int:
        try:
            return int(json.load(open(elo_path)).get("processed_through", anchor_gen))
        except Exception:
            return anchor_gen

    def train_block(big, k: int):
        nonlocal consumed
        n = big["planes"].shape[0]
        net.train()
        acc = [0.0, 0.0, 0.0]
        for _ in range(k):
            b = rng.integers(0, n, size=args.batch_size)
            pl = torch.from_numpy(big["planes"][b]).to(device)
            tgt_z = torch.from_numpy(big["value"][b]).to(device)
            tgt_q = scatter(b, big["q_off"], big["q_idx"], big["q_prob"], PER_HEAD, device)
            tgt_a = scatter(b, big["a_off"], big["a_idx"], big["a_prob"], PER_HEAD, device)
            v_pred, q_logits, a_logits = net(pl)
            loss_v = F.mse_loss(v_pred, tgt_z)
            loss_q = -(tgt_q * F.log_softmax(q_logits, dim=1)).sum(1).mean()
            loss_a = -(tgt_a * F.log_softmax(a_logits, dim=1)).sum(1).mean()
            loss = loss_v + loss_q + loss_a
            opt.zero_grad()
            loss.backward()
            opt.step()
            consumed += args.batch_size
            acc[0] += float(loss_v)
            acc[1] += float(loss_q)
            acc[2] += float(loss_a)
        return acc

    existing = [int(os.path.basename(d)[3:]) for d in glob.glob(os.path.join(snaps, "gen*"))
                if os.path.basename(d)[3:].isdigit()]
    snap_gen = max(existing + [anchor_gen])
    refresh()
    produced_at_gen_start = produced  # bursty pacing: fresh examples since this gen began
    print(f"amazons-az learner: device {device}  batch {args.batch_size}  "
          f"throttle {args.max_reuse}x  gen-examples {args.gen_examples} (bursty)  "
          f"eval-lead {args.eval_lead}  snapshot/{args.steps_per_snapshot}  "
          f"after gen {snap_gen}", flush=True)

    BLOCK = 50
    steps_into_snap = 0
    losses = [0.0, 0.0, 0.0]
    nblk = 0
    gens_this_run = 0
    while True:
        if refresh() or big is None:
            if cache:
                big = merge(list(cache.values()))
        n = big["planes"].shape[0] if big else 0
        if n < args.min_samples:
            time.sleep(2.0)
            continue
        if steps_into_snap == 0:
            need = (snap_gen + 1) - args.eval_lead
            if need > anchor_gen and processed_through() < need:
                time.sleep(2.0)
                continue
            # Stay fully idle (actor uncontended) until a whole generation of FRESH
            # self-play has accumulated, then train the gen in one burst below.
            if args.gen_examples > 0 and produced - produced_at_gen_start < args.gen_examples:
                time.sleep(5.0)
                continue
            # Safety throttle, checked only at gen start; once the burst is underway
            # we run it to completion without interleaving (no actor-contention churn).
            if int(args.max_reuse * produced) - consumed < args.batch_size:
                time.sleep(2.0)
                continue

        block = min(BLOCK, args.steps_per_snapshot - steps_into_snap)
        acc = train_block(big, block)
        for i in range(3):
            losses[i] += acc[i]
        nblk += block
        steps_into_snap += block
        net.eval()
        save_atomic(net, latest)

        if steps_into_snap >= args.steps_per_snapshot:
            snap_gen += 1
            d = os.path.join(snaps, f"gen{snap_gen:03d}")
            os.makedirs(d, exist_ok=True)
            save_checkpoint(net.to("cpu"), os.path.join(d, "checkpoint.safetensors"))
            net.to(device)
            # Refresh the ORT actor's weights once per snapshot (not per block).
            export_onnx_atomic(net, onnx_path)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"arch": arch_of(net), "generation": snap_gen}, f)
            fresh = produced - produced_at_gen_start
            print(f"snapshot gen{snap_gen}  samples {n}  chunks {len(cache)}  "
                  f"fresh {fresh} ({args.steps_per_snapshot * args.batch_size / max(1, fresh):.1f}x gen)  "
                  f"produced {produced}  consumed {consumed} ({consumed/max(1,produced):.1f}x)  "
                  f"loss v {losses[0]/max(1,nblk):.4f} q {losses[1]/max(1,nblk):.4f} "
                  f"a {losses[2]/max(1,nblk):.4f}", flush=True)
            steps_into_snap = 0
            losses = [0.0, 0.0, 0.0]
            nblk = 0
            produced_at_gen_start = produced  # begin accumulating the next gen's data
            save_opt()
            gens_this_run += 1
            if args.exit_after_gens and gens_this_run >= args.exit_after_gens:
                print(f"learner: trained {gens_this_run} gen(s), exiting", flush=True)
                return


if __name__ == "__main__":
    main()
