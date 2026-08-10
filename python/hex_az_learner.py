"""Continuous Hex AlphaZero learner (HexAZNet).

Plain AZ: loss = value MSE + policy cross-entropy against the MCTS visit
distribution. The actor's dump is dense (169 floats/example — small enough that
the sparse encoding amazons needs would only add complexity), and everything is
already in the canonical frame, so there is no colour handling here at all.

From scratch it writes a random gen-0 net (+ config.json) so the actor can start.

Pair with:
  actor: hex-engine selfplay-loop --live data/models/hex_az/live
  eval : hex-engine eval-match --a <new> --b <old>      (relative)
         hex-engine vs-mohex --model <ckpt>             (absolute)
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

from playground.model_server.hex_net import (
    CELLS,
    IN_PLANES,
    N,
    POLICY_DIM,
    HexAZNet,
    arch_of,
    load_checkpoint,
    save_checkpoint,
)

PLANES = IN_PLANES * CELLS  # floats per encoded position
_FLOATS = PLANES + 1 + POLICY_DIM + POLICY_DIM  # planes, value, policy, ownership
_STRIDE = _FLOATS * 4
_MAGIC = 0x58454831  # "HEX1"
_VERSION = 2
_FIELDS = ("planes", "value", "policy", "ownership")


def read_examples(path: str) -> dict:
    """Parse the actor's dump: magic u32, version u32, count u64, then per example
    planes, value, policy, ownership — all f32 little-endian.

    The header is checked rather than assumed: a chunk written by a different
    build would otherwise be read as garbage with the right shape."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, version = struct.unpack_from("<II", buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"{path}: not a hex self-play chunk (magic {magic:#x})")
    if version != _VERSION:
        raise ValueError(f"{path}: chunk version {version}, this learner reads {_VERSION}")
    (count,) = struct.unpack_from("<Q", buf, 8)
    expect = 16 + count * _STRIDE
    if len(buf) != expect:
        raise ValueError(f"{path}: {len(buf)} bytes, expected {expect} for {count} examples")
    raw = np.frombuffer(buf, dtype="<f4", count=count * _FLOATS, offset=16).reshape(count, _FLOATS)
    return {
        "planes": np.ascontiguousarray(raw[:, :PLANES]).reshape(count, IN_PLANES, N, N),
        "value": np.ascontiguousarray(raw[:, PLANES]),
        "policy": np.ascontiguousarray(raw[:, PLANES + 1 : PLANES + 1 + POLICY_DIM]),
        "ownership": np.ascontiguousarray(raw[:, PLANES + 1 + POLICY_DIM :]),
    }


def merge(chunks: list[dict]) -> dict:
    return {k: np.concatenate([c[k] for c in chunks]) for k in _FIELDS}


def save_atomic(net: HexAZNet, path: str) -> None:
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu") for k, v in net.state_dict().items()}
    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)


def export_onnx_atomic(net: HexAZNet, path: str) -> None:
    """Export for the actor's ORT/TensorRT backend (`--backend ort`).

    Dynamic batch axis: the actor's batch varies with how many games have a leaf
    pending, and TensorRT builds (and caches) an engine per shape it sees.
    """
    was_training = net.training
    net.eval()
    dev = next(net.parameters()).device
    tmp = path + ".tmp"
    torch.onnx.export(
        net, torch.zeros(1, IN_PLANES, N, N, device=dev), tmp,
        input_names=["planes"], output_names=["value", "policy", "ownership"],
        dynamic_axes={k: {0: "b"} for k in ("planes", "value", "policy", "ownership")},
        opset_version=17, dynamo=False,
    )
    os.replace(tmp, path)
    if was_training:
        net.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/hex_az/live")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-snapshot", type=int, default=300)
    # Cap how many times an example may be replayed on average.
    ap.add_argument("--max-reuse", type=float, default=4.0)
    # Bursty training: stay idle until this many FRESH self-play examples exist,
    # then train the generation in one burst so the actor runs uncontended in
    # between. 0 disables the gate (cadence then comes from the orchestrator).
    ap.add_argument("--gen-examples", type=int, default=20000)
    # Exit after this many snapshots (0 = forever). The orchestrator passes 1 so
    # the learner is a fresh, short-lived process per generation.
    ap.add_argument("--exit-after-gens", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=4000)
    ap.add_argument("--ownership-weight", type=float, default=0.15,
                    help="weight on the auxiliary per-cell ownership loss")
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

    net = HexAZNet(channels=args.channels, blocks=args.blocks)
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
    # Persist Adam's moments so relaunching the learner each generation doesn't
    # reset the adaptive LR every time.
    opt_path = os.path.join(live, "opt_state.pt")
    if os.path.exists(opt_path):
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"loaded optimizer state from {opt_path}", flush=True)
        except Exception as e:
            print(f"optimizer state load failed (fresh Adam): {e}", flush=True)

    def save_opt() -> None:
        tmp = opt_path + ".tmp"
        torch.save(opt.state_dict(), tmp)
        os.replace(tmp, opt_path)

    onnx_path = os.path.join(live, "model.onnx")
    export_onnx_atomic(net, onnx_path)

    anchor_gen = int(json.load(open(cfg_path)).get("generation", 0)) if os.path.exists(cfg_path) else 0

    cache: dict[str, dict] = {}
    big: dict | None = None
    produced = 0
    consumed = 0

    def refresh() -> bool:
        """Sync the in-memory replay buffer with the actor's chunk files."""
        nonlocal produced
        files = set(glob.glob(os.path.join(buffer_dir, "chunk_*.bin")))
        changed = False
        for f in files - cache.keys():
            try:
                ex = read_examples(f)
            except Exception:
                continue  # still being written; pick it up next pass
            cache[f] = ex
            produced += ex["planes"].shape[0]
            changed = True
        for f in list(cache.keys() - files):
            del cache[f]  # pruned by the actor
            changed = True
        return changed

    def train_block(big: dict, k: int) -> list[float]:
        nonlocal consumed
        n = big["planes"].shape[0]
        net.train()
        acc = [0.0, 0.0, 0.0]
        for _ in range(k):
            b = rng.integers(0, n, size=args.batch_size)
            pl = torch.from_numpy(big["planes"][b]).to(device)
            tgt_z = torch.from_numpy(big["value"][b]).to(device)
            tgt_p = torch.from_numpy(big["policy"][b]).to(device)
            tgt_o = torch.from_numpy(big["ownership"][b]).to(device)
            v_pred, p_logits, o_pred = net(pl)
            loss_v = F.mse_loss(v_pred, tgt_z)
            loss_p = -(tgt_p * F.log_softmax(p_logits, dim=1)).sum(1).mean()
            # Ownership is 169 numbers per position against the value head's one,
            # so it is weighted down to keep it an auxiliary signal rather than
            # the objective. (KataGo scales its ownership loss similarly.)
            loss_o = F.mse_loss(o_pred, tgt_o)
            loss = loss_v + loss_p + args.ownership_weight * loss_o
            opt.zero_grad()
            loss.backward()
            opt.step()
            consumed += args.batch_size
            acc[0] += loss_v.item()
            acc[1] += loss_p.item()
            acc[2] += loss_o.item()
        return acc

    existing = [
        int(os.path.basename(d)[3:])
        for d in glob.glob(os.path.join(snaps, "gen*"))
        if os.path.basename(d)[3:].isdigit()
    ]
    snap_gen = max(existing + [anchor_gen])
    refresh()
    produced_at_gen_start = produced
    print(
        f"hex-az learner: device {device}  batch {args.batch_size}  throttle {args.max_reuse}x  "
        f"gen-examples {args.gen_examples} (bursty)  snapshot/{args.steps_per_snapshot}  "
        f"after gen {snap_gen}",
        flush=True,
    )

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
            # Stay fully idle (actor uncontended) until a generation of FRESH
            # self-play has accumulated, then train it in one burst.
            if args.gen_examples > 0 and produced - produced_at_gen_start < args.gen_examples:
                time.sleep(5.0)
                continue
            # Safety throttle, checked only at generation start.
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
        save_atomic(net, latest)  # the actor hot-reloads this

        if steps_into_snap >= args.steps_per_snapshot:
            snap_gen += 1
            d = os.path.join(snaps, f"gen{snap_gen:03d}")
            os.makedirs(d, exist_ok=True)
            save_checkpoint(net.to("cpu"), os.path.join(d, "checkpoint.safetensors"))
            net.to(device)
            # Refresh the ORT actor's weights once per snapshot, not per block.
            export_onnx_atomic(net, onnx_path)
            for path in (os.path.join(d, "config.json"), cfg_path):
                with open(path, "w") as f:
                    json.dump({"arch": arch_of(net), "generation": snap_gen}, f)
            fresh = produced - produced_at_gen_start
            print(
                f"snapshot gen{snap_gen}  samples {n}  chunks {len(cache)}  "
                f"fresh {fresh} ({args.steps_per_snapshot * args.batch_size / max(1, fresh):.1f}x gen)  "
                f"produced {produced}  consumed {consumed} ({consumed / max(1, produced):.1f}x)  "
                f"loss v {losses[0] / max(1, nblk):.4f} p {losses[1] / max(1, nblk):.4f} "
                f"o {losses[2] / max(1, nblk):.4f}",
                flush=True,
            )
            steps_into_snap = 0
            losses = [0.0, 0.0, 0.0]
            nblk = 0
            produced_at_gen_start = produced
            save_opt()
            gens_this_run += 1
            if args.exit_after_gens and gens_this_run >= args.exit_after_gens:
                print(f"learner: trained {gens_this_run} gen(s), exiting", flush=True)
                return


if __name__ == "__main__":
    main()
