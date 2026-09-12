"""fp16-I/O variants of the Minions head graphs.

A head's tensors cross PCIe on every call. A diffusion step uploads ``h``
(48x7x7) and ``decl`` and downloads 49x46 logits per row, ``T`` times per
expansion. TensorRT runs the graphs in fp16 internally anyway (``--fp16``), so
declaring the I/O fp16 halves the bytes moved without changing what the kernels
compute. The transform only wraps the exported fp32 graph in Cast nodes; the
body is untouched.

Each variant records the sha256 of the graph it was made from
(``metadata_props["f16io_source_sha256"]``). The engine refuses a variant whose
source no longer matches: a stale variant beside a new trunk would produce
plausible garbage rather than an error.

    python -m playground.model_server.onnx_f16io <graph dir>...
"""
from __future__ import annotations

import hashlib
import os
import sys

import onnx
from onnx import TensorProto, helper

SUFFIX = ".f16io.onnx"
HEADS = {
    "diffusion.onnx": (("h", "decl"), ("logits",)),
    "energy.onnx": (("h", "decl"), ("energy",)),
}
META_KEY = "f16io_source_sha256"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def f16io(src: str, dst: str, inputs, outputs) -> None:
    m = onnx.load(src)
    g = m.graph
    head = []
    for vi in g.input:
        if vi.name not in inputs:
            continue
        inner = vi.name + "__f32"
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == vi.name:
                    n.input[i] = inner
        head.append(helper.make_node("Cast", [vi.name], [inner], to=TensorProto.FLOAT,
                                     name=f"{vi.name}_to_f32"))
        vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    tail = []
    for vo in g.output:
        if vo.name not in outputs:
            continue
        inner = vo.name + "__f32"
        for n in g.node:
            for i, x in enumerate(n.output):
                if x == vo.name:
                    n.output[i] = inner
            for i, x in enumerate(n.input):
                if x == vo.name:
                    n.input[i] = inner
        tail.append(helper.make_node("Cast", [inner], [vo.name], to=TensorProto.FLOAT16,
                                     name=f"{vo.name}_to_f16"))
        vo.type.tensor_type.elem_type = TensorProto.FLOAT16
    missing = (set(inputs) - {n.output[0].removesuffix("__f32") for n in head}) | \
              (set(outputs) - {n.output[0] for n in tail})
    if missing:
        raise ValueError(f"{src}: no graph input/output named {sorted(missing)}")
    nodes = head + list(g.node) + tail
    del g.node[:]
    g.node.extend(nodes)
    meta = m.metadata_props.add()
    meta.key, meta.value = META_KEY, sha256_file(src)
    onnx.checker.check_model(m)
    tmp = dst + ".tmp"
    onnx.save(m, tmp)
    os.replace(tmp, dst)


def write_variants(out_dir: str) -> None:
    """Write ``<head>.f16io.onnx`` beside each head graph in ``out_dir``."""
    for name, (ins, outs) in HEADS.items():
        f16io(os.path.join(out_dir, name), os.path.join(out_dir, name[: -len(".onnx")] + SUFFIX),
              ins, outs)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        write_variants(d)
        print(f"wrote fp16-I/O head variants in {d}")
