"""Export a Go9 net checkpoint to ONNX for the ORT/TensorRT inference backend.

Outputs an ONNX graph with a dynamic batch axis: input planes (B,3,9,9) ->
(value (B,), policy_logits (B,82)). The Rust OrtModel softmaxes the logits to
match candle's value_policy_batch. Supports both arch kinds (go9-az, go9-resnet);
for the reification net it exports the value+policy path only (no reifier/latent).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from playground.model_server.go9_net import Go9AZNet, Go9ResNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a .safetensors checkpoint")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    arch = json.load(open(cfg_path)).get("arch", {}) if os.path.exists(cfg_path) else {}
    kind = arch.get("kind", "go9-az")
    channels = int(arch.get("channels", 96))
    blocks = int(arch.get("blocks", 10))

    net = Go9AZNet(channels=channels, blocks=blocks) if kind == "go9-az" \
        else Go9ResNet(channels=channels, blocks=blocks)
    from safetensors.torch import load_file
    net.load_state_dict(load_file(args.ckpt))
    net.eval()

    # Both nets' forward already returns (value, policy_logits); wrap only to pin
    # the output order/names for the exported graph.
    class VP(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, planes):
            v, logits = self.m(planes)
            return v, logits

    torch.onnx.export(
        VP(net).eval(), torch.zeros(1, 3, 9, 9), args.out,
        input_names=["planes"], output_names=["value", "policy"],
        dynamic_axes={"planes": {0: "b"}, "value": {0: "b"}, "policy": {0: "b"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"exported {kind} ({channels}ch/{blocks}b) -> {args.out}")


if __name__ == "__main__":
    main()
