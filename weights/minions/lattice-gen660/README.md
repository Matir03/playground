# Minions turn-lattice player: gen 660 (lattice537 lineage)

Champion of the `lattice537` training cycle since 2026-10-09; it is the
self-play net of the current run (champion self-play from gen 759). The
design is in [docs/minions-lattice.md](../../../docs/minions-lattice.md) and
the writeup in [docs/minions-lattice-writeup.pdf](../../../docs/minions-lattice-writeup.pdf).

| file | what |
|---|---|
| `net.pt` | tier-1 trunk + value head (PyTorch, `PosNet`, arch `pos:96x8`, 1.40M params) |
| `t2.pt` | tier-2 head (lattice view, arch `96x6`, 1.03M params) |
| `tier1.onnx` | start-of-turn position -> value + fp16 trunk image |
| `tier2.onnx` | cached trunk + partial-commitment planes -> per-cell prior logits, per-choice values, node value |
| `value.onnx`, `value_sparse.onnx` | tier 1 exported as a steps-format value net (bids, steps tools) |
| `lattice.json` | export options read by the engine (`c_trunk`, `trunk_gather`, `lview`) |
| `meta.json` | training data sizes and held-out losses |

The ONNX graphs need TensorRT (`source scripts/ort-env.sh`); the CUDA EP
returns wrong values for some shapes. Play it in the UI:

    source scripts/ort-env.sh
    ./target/release/minions-engine serve --addr 127.0.0.1:7105 \
        --lattice-net vprior=100:weights/minions/lattice-gen660

`vprior=100:` selects the near-uniform tier-1 prior the net was trained and
gated with. The same `vprior=100:<dir>` spec works as a `steps-match` player.
To continue training: `python/minions_lattice_learner.py --t1 net.pt
--t1-arch pos:96x8 --init t2.pt --arch 96x6 --lview --t1-node`.
