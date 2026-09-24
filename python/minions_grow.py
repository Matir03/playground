"""Deepen a Minions net without changing what it computes.

Adds residual blocks to the trunk (and optionally the diffusion / energy heads)
of a trained checkpoint. Each new block's second BatchNorm starts at gamma =
beta = 0, so its residual branch outputs exactly zero and the block is
`relu(x + 0) = x` -- the identity, since every block's input is already
post-ReLU. The grown net therefore starts as the source net, output for output,
and training switches the new blocks on through that gamma (the zero-init
residual of Goyal et al.).

New blocks go after the existing ones, so every source tensor keeps its name.
The script checks the equivalence on buffer positions before writing anything.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
from minions_az_learner import IN_PLANES, N, Buffer, decl_planes, energy_planes  # noqa: E402
from minions_distill import live_fields  # noqa: E402
from playground.model_server.minions_net import (  # noqa: E402
    MinionsNet, arch_of, build_model, export_onnx_atomic,
)


def zero_residual(block):
    torch.nn.init.zeros_(block.b2.weight)
    torch.nn.init.zeros_(block.b2.bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="snapshot dir to grow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--dif-blocks", type=int, default=None)
    ap.add_argument("--energy-blocks", type=int, default=None)
    ap.add_argument("--buffer", default=None, help="check the equivalence on these rows")
    ap.add_argument("--opt-state", default=None,
                    help="the source run's opt_state.pt: Adam's moments are carried over "
                         "by parameter name (they are stored by index, which growing "
                         "shifts); new blocks start with zero moments")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    src = build_model(cfg["arch"])
    src.load_state_dict(load_file(os.path.join(args.src, "checkpoint.safetensors")))
    a = arch_of(src)
    dif_blocks = args.dif_blocks or a["dif_blocks"]
    eng_blocks = args.energy_blocks or a["energy_blocks"]
    assert args.blocks >= a["blocks"] and dif_blocks >= a["dif_blocks"] \
        and eng_blocks >= a["energy_blocks"], "only grows"

    torch.manual_seed(0)
    net = MinionsNet(a["channels"], args.blocks, eng_blocks, dif_ch=a["dif_ch"],
                     dif_blocks=dif_blocks, eng_ch=a["eng_ch"])
    for blk in list(net.blocks[a["blocks"]:]) + list(net.dif_extra[a["dif_blocks"] - 1:]) \
            + list(net.eng_blocks[a["energy_blocks"]:]):
        zero_residual(blk)
    missing, unexpected = net.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected, unexpected
    print(f"grew {a['blocks']}->{args.blocks} trunk, {a['dif_blocks']}->{dif_blocks} diffusion, "
          f"{a['energy_blocks']}->{eng_blocks} energy blocks; {len(missing)} new tensors; "
          f"{sum(p.numel() for p in src.parameters()):,} -> "
          f"{sum(p.numel() for p in net.parameters()):,} params", flush=True)

    src.eval()
    net.eval()
    if args.buffer:
        buf = Buffer(args.buffer)
        buf.refresh()
        batch = buf.merged()
        idx = np.random.default_rng(0).choice(len(batch["value"]), 256, replace=False)
        planes = torch.from_numpy(batch["planes"][idx].astype(np.float32)).view(-1, IN_PLANES, N, N)
        live_f = live_fields(batch, idx)
        spec = np.zeros((len(idx), 3 * N * N), dtype=np.uint8)
        committed = live_f & (np.random.default_rng(1).random(live_f.shape) < 0.5)
        decl = torch.from_numpy(decl_planes(spec, committed))
        edecl = torch.from_numpy(energy_planes(spec, live_f))
        with torch.no_grad():
            outs_a = src(planes, decl, edecl)
            outs_b = net(planes, decl, edecl)
        worst = max(float((x - y).abs().max()) for x, y in zip(outs_a, outs_b))
        print(f"max |difference| over value, diffusion and energy outputs: {worst:.2e}")
        assert worst < 1e-4, "grown net is not equivalent"

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()},
              os.path.join(args.out, "checkpoint.safetensors"))
    json.dump({"arch": arch_of(net), "generation": int(cfg.get("generation", 0))},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    export_onnx_atomic(net, args.out)
    if args.opt_state:
        old = torch.load(args.opt_state, map_location="cpu")
        src_names = [n for n, _ in src.named_parameters()]
        assert len(old["state"]) <= len(src_names)
        by_name = {src_names[i]: st for i, st in old["state"].items()}
        state = {}
        for i, (n, p) in enumerate(net.named_parameters()):
            if n in by_name:
                state[i] = by_name[n]
            else:
                state[i] = {"step": torch.tensor(0.0), "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p)}
        group = dict(old["param_groups"][0])
        group["params"] = list(range(len(state)))
        torch.save({"state": state, "param_groups": [group]},
                   os.path.join(args.out, "opt_state.pt"))
        print(f"carried Adam moments for {len(by_name)} tensors, "
              f"{len(state) - len(by_name)} new")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
