"""fp16 edges for the Minions graphs.

Everything the trunk hands to the heads crosses PCIe as ``h``: 48x7x7 per row,
downloaded from the trunk, then uploaded to a head on every call (``T`` times
per expansion for diffusion). A diffusion step also downloads 49x46 logits
per row. TensorRT runs the graphs in fp16 internally anyway (``--fp16``), so
declaring these edges fp16 halves the bytes moved without changing what the
kernels compute. The transform only wraps the exported fp32 graph in Cast
nodes; the body is untouched. ``value`` stays fp32.

The engine reads each graph's edge dtypes at load, so older snapshots
(fp32 edges, still played by the ladder) keep working with no flag.

    python -m playground.model_server.onnx_f16io <graph dir>...   # in place
"""
from __future__ import annotations

import os
import shutil
import sys

import onnx
from onnx import TensorProto, helper

# Heads first and the trunk last: the actor reloads when `trunk.onnx` changes.
NATIVE = {
    "diffusion.onnx": (("h", "decl"), ("logits",)),
    "energy.onnx": (("h", "decl"), ("energy",)),
    "trunk.onnx": (("planes",), ("h",)),
}


def f16io(src: str, dst: str, inputs, outputs) -> None:
    """Rewrite ``src`` with fp16 ``inputs``/``outputs`` into ``dst`` (may equal
    ``src``). Idempotent: a graph already fp16 at those edges is left as is."""
    m = onnx.load(src)
    g = m.graph
    edge = {vi.name: vi for vi in (*g.input, *g.output)}
    missing = [n for n in (*inputs, *outputs) if n not in edge]
    if missing:
        raise ValueError(f"{src}: no graph input/output named {missing}")
    if all(edge[n].type.tensor_type.elem_type == TensorProto.FLOAT16 for n in (*inputs, *outputs)):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        return
    head, tail = [], []
    for name in inputs:
        inner = name + "__f32"
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = inner
        head.append(helper.make_node("Cast", [name], [inner], to=TensorProto.FLOAT, name=f"{name}_to_f32"))
        edge[name].type.tensor_type.elem_type = TensorProto.FLOAT16
    for name in outputs:
        inner = name + "__f32"
        for n in g.node:
            for i, x in enumerate(n.output):
                if x == name:
                    n.output[i] = inner
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = inner
        tail.append(helper.make_node("Cast", [inner], [name], to=TensorProto.FLOAT16, name=f"{name}_to_f16"))
        edge[name].type.tensor_type.elem_type = TensorProto.FLOAT16
    nodes = head + list(g.node) + tail
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    tmp = dst + ".f16tmp"
    onnx.save(m, tmp)
    os.replace(tmp, dst)


def convert_dir(out_dir: str) -> None:
    """Give the graphs in ``out_dir`` fp16 edges, in place, trunk last."""
    for name, (ins, outs) in NATIVE.items():
        path = os.path.join(out_dir, name)
        f16io(path, path, ins, outs)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        convert_dir(d)
        print(f"fp16 edges in {d}")
