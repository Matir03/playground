"""Store-indexed variants of the Minions head graphs (``*_g.onnx``).

Self-play's head calls were bound by bytes, not arithmetic. At ~160 rows a
diffusion call moved ~10.4 KB per row: the position's features ``h`` (4.7 KB),
identical on every one of a chain's steps; the declaration planes (1.2 KB); and
the full ``(49, 46)`` logits (4.5 KB), of which the sampler reads only the live
cells, ~8 of 49. The variants written here take ``h`` from a device-resident
store by index, so it crosses PCIe once per wave rather than once per step, and
the diffusion variant returns only the logit rows the caller names:

``diffusion_g.onnx``: ``h_store (M, C, N, N)``, ``h_idx (B,)`` int64,
``decl (B, 12, N, N)``, ``cell_flat (P,)`` int64 -> ``logits_g (P, 46)``, where
``cell_flat[j] = row * CELLS + cell`` picks row ``row``'s cell ``cell``.

``energy_g.onnx``: ``h_store``, ``h_idx (B,)``, ``decl (B, 11, N, N)`` ->
``energy (B,)``.

Both are graph surgery on the exported graphs -- a ``Gather`` in front of the
original ``h`` input, and for the diffusion head a reshape and ``Gather`` after
its output -- so they compute exactly what the originals do on the rows asked
for, with the originals' edge dtypes.
"""

from __future__ import annotations

import os
import sys

import onnx
from onnx import TensorProto, helper


def _store_input(g: onnx.GraphProto) -> None:
    """Replace graph input ``h`` with ``Gather(h_store, h_idx)``."""
    (h_in,) = [i for i in g.input if i.name == "h"]
    tt = h_in.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in tt.shape.dim]
    g.input.remove(h_in)
    g.input.insert(0, helper.make_tensor_value_info("h_store", tt.elem_type, ["m", *dims[1:]]))
    g.input.insert(1, helper.make_tensor_value_info("h_idx", TensorProto.INT64, [dims[0]]))
    g.node.insert(0, helper.make_node("Gather", ["h_store", "h_idx"], ["h"], axis=0, name="store_gather"))


def with_store(src: str, dst: str, gather_logits: bool) -> None:
    m = onnx.load(src)
    g = m.graph
    _store_input(g)
    if gather_logits:
        (out,) = [o for o in g.output if o.name == "logits"]
        tt = out.type.tensor_type
        # The export leaves the concatenated width symbolic; it is fixed by the
        # declaration encoding.
        from playground.model_server.minions_net import DECL_LOGITS as width

        g.output.remove(out)
        g.initializer.append(helper.make_tensor("logits_flat_shape", TensorProto.INT64, [2], [-1, width]))
        g.node.append(helper.make_node("Reshape", ["logits", "logits_flat_shape"], ["logits_flat"], name="flat"))
        g.input.append(helper.make_tensor_value_info("cell_flat", TensorProto.INT64, ["p"]))
        g.node.append(helper.make_node("Gather", ["logits_flat", "cell_flat"], ["logits_g"], axis=0, name="cell_gather"))
        g.output.append(helper.make_tensor_value_info("logits_g", tt.elem_type, ["p", width]))
    onnx.checker.check_model(m)
    tmp = dst + ".tmp"
    onnx.save(m, tmp)
    os.replace(tmp, dst)


def write_store_graphs(out_dir: str) -> None:
    """``diffusion_g.onnx`` and ``energy_g.onnx`` beside their originals, when
    those exist. Called by the export before ``trunk.onnx`` is written, so the
    actor's reload (keyed on the trunk) sees a matching set."""
    for src, dst, gather in (
        ("diffusion.onnx", "diffusion_g.onnx", True),
        ("energy.onnx", "energy_g.onnx", False),
    ):
        path = os.path.join(out_dir, src)
        if os.path.exists(path):
            with_store(path, os.path.join(out_dir, dst), gather)
        elif os.path.exists(os.path.join(out_dir, dst)):
            os.remove(os.path.join(out_dir, dst))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        write_store_graphs(d)
        print(f"store graphs in {d}")
