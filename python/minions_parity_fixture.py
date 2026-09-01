"""Write a fixture that pins the candle net against PyTorch.

The two implementations load the *same* `.safetensors`, so a renamed parameter or
a transposed reshape breaks one silently. This writes a checkpoint, a fixed
input, and PyTorch's outputs; `net.rs::tests::parity_with_python_net` replays it
through candle and requires agreement to 1e-4.

    python python/minions_parity_fixture.py --out data/minions_parity
    MINIONS_PARITY_DIR=data/minions_parity cargo test -p minions-engine parity -- --nocapture
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

from playground.model_server.minions_net import (
    CELLS, DECL_PLANES, IN_PLANES, MinionsNet, arch_of, N,
)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/minions_parity")
ap.add_argument("--channels", type=int, default=24)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--energy-blocks", type=int, default=2)
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

torch.manual_seed(args.seed)
net = MinionsNet(args.channels, args.blocks, args.energy_blocks).eval()

# Give BatchNorm non-trivial running stats: freshly initialised BN is nearly the
# identity, which would let a genuine normalisation bug slip through parity.
with torch.no_grad():
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0.0, 0.5)
            m.running_var.uniform_(0.5, 1.5)
            m.weight.normal_(1.0, 0.2)
            m.bias.normal_(0.0, 0.2)

os.makedirs(args.out, exist_ok=True)
save_file({k: v.contiguous() for k, v in net.state_dict().items()},
          os.path.join(args.out, "checkpoint.safetensors"))
with open(os.path.join(args.out, "config.json"), "w") as fh:
    json.dump({"arch": arch_of(net), "generation": 0}, fh, indent=2)

B = 2
g = torch.Generator().manual_seed(args.seed + 1)
planes = torch.randn(B, IN_PLANES, N, N, generator=g)
decl = torch.randn(B, DECL_PLANES, N, N, generator=g)

# The energy head takes the narrower encoding: `decl` without the move-commit
# plane. Slicing it here keeps both heads scoring the *same* declaration.
edecl = torch.cat([decl[:, :2], decl[:, 3:]], dim=1)

with torch.no_grad():
    h = net.trunk(planes)
    v = net.value_head(h)
    hh = net.head_features(h)
    mv, tg, sp = net.diffusion_head(hh, decl)
    logits = torch.cat([mv, tg, sp], dim=2)  # (B, CELLS, 46) — candle's layout
    e = net.energy_head(hh, edecl)

with open(os.path.join(args.out, "input.bin"), "wb") as fh:
    fh.write(planes.flatten().numpy().astype("<f4").tobytes())
    fh.write(decl.flatten().numpy().astype("<f4").tobytes())
with open(os.path.join(args.out, "expected.bin"), "wb") as fh:
    for t in (v, logits.flatten(), e):
        fh.write(t.numpy().astype("<f4").tobytes())

print(f"wrote fixture to {args.out}: {args.channels}ch/{args.blocks}b, "
      f"expected width {v.numel() + logits.numel() + e.numel()}")
