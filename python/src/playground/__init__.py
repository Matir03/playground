"""playground — MuZero-variant research playground.

Sub-packages:
    model_server  — hosts encoder/eval/policy/reifier NNs over gRPC
    orchestrator  — self-play loop, match scheduler, replay writer
    training      — training loops for encoder/eval/policy/reifier
    eval          — tournament runner + ELO fitting
    gen           — generated protobuf types (from schemas/proto)
"""

__version__ = "0.1.0"
