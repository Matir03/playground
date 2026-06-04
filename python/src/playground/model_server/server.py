"""ModelService gRPC server.

Serves the reification-based MuZero variant's four heads (encoder / eval /
policy / reifier) for a single game. Today only Connect 4 is wired, with
*random* initial weights — no training yet (see docs/algorithm.md and
``net.py``). This is enough to play against the NN end-to-end from the web UI.

Run with ``playground-model-server --port 7200 --game connect4``.
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent import futures

import grpc
import torch

from playground.gen.playground.game import game_pb2
from playground.gen.playground.model import model_pb2, model_pb2_grpc

import json
import os

from .net import (
    GS_DIM,
    ConnectFourModel,
    build_model,
    connect4_legal_columns,
    connect4_planes,
    gs_from_bytes,
    gs_to_bytes,
    load_checkpoint,
)

log = logging.getLogger(__name__)

# Default policy width (Connect 4 has at most 7 legal moves).
C4_DEFAULT_B = 7


class ModelServicer(model_pb2_grpc.ModelServiceServicer):
    def __init__(self, game: str, seed: int = 0, checkpoint: str | None = None) -> None:
        if game != "connect4":
            raise SystemExit(f"model server: only 'connect4' is wired today (got {game!r})")
        self.game = game
        self.game_id = game_pb2.GAME_ID_CONNECT4
        self._lock = threading.Lock()
        self._reloads = 0
        self._load(seed, checkpoint)

    def _load(self, seed: int, checkpoint: str | None = None) -> None:
        torch.manual_seed(seed)
        if checkpoint:
            # Build the architecture recorded next to the checkpoint (a sibling
            # config.json with {"arch": ...}); default to the small net.
            arch = None
            cfg = os.path.join(os.path.dirname(checkpoint), "config.json")
            if os.path.exists(cfg):
                with open(cfg) as f:
                    arch = json.load(f).get("arch")
            net = build_model(arch)
            load_checkpoint(net, checkpoint)
        else:
            net = ConnectFourModel()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        if checkpoint:
            self.model_version = f"connect4-{os.path.splitext(os.path.basename(checkpoint))[0]}"
            log.info("loaded model %s from %s", self.model_version, checkpoint)
        else:
            self.model_version = f"connect4-random-seed{seed}"
            log.info("loaded model %s (random weights)", self.model_version)

    def _gs_msg(self, gs_tensor) -> game_pb2.LatentGameState:
        return game_pb2.LatentGameState(
            payload=gs_to_bytes(gs_tensor),
            shape=[GS_DIM],
            model_version=self.model_version,
        )

    # — unary —
    def Encode(self, request, context):
        planes = connect4_planes(request.state.payload)
        with self._lock:
            gs = self.net.encode(planes)
        return model_pb2.EncodeResponse(gs=self._gs_msg(gs))

    def Eval(self, request, context):
        gs = gs_from_bytes(request.gs.payload)
        with self._lock:
            value = self.net.value(gs)
        return model_pb2.EvalResponse(value=value)

    def Policy(self, request, context):
        gs = gs_from_bytes(request.gs.payload)
        with self._lock:
            cands = self.net.policy(gs, request.top_b or C4_DEFAULT_B)
        resp = model_pb2.PolicyResponse()
        for prior, action, next_gs in cands:
            resp.candidates.append(
                model_pb2.PolicyCandidate(
                    prior=prior,
                    action_vector=gs_to_bytes(action),
                    next_gs=self._gs_msg(next_gs),
                )
            )
        return resp

    def Reify(self, request, context):
        # Discrete action space: decode the column the policy edge selected.
        legal = connect4_legal_columns(request.from_state.payload)
        col = self.net.reify_column(request.action_vector, legal)
        move = game_pb2.Move(game_id=self.game_id, payload=bytes([col]))
        return model_pb2.ReifyResponse(move=move, score=0.0)

    # — batched (simple fan-out over the unary heads) —
    def EncodeBatch(self, request, context):
        resp = model_pb2.EncodeBatchResponse()
        for state in request.states:
            resp.gs.append(self.Encode(model_pb2.EncodeRequest(state=state), context).gs)
        return resp

    def EvalBatch(self, request, context):
        resp = model_pb2.EvalBatchResponse()
        for gs in request.gs:
            resp.values.append(self.Eval(model_pb2.EvalRequest(gs=gs), context).value)
        return resp

    def PolicyBatch(self, request, context):
        resp = model_pb2.PolicyBatchResponse()
        for req in request.requests:
            resp.responses.append(self.Policy(req, context))
        return resp

    def ReifyBatch(self, request, context):
        resp = model_pb2.ReifyBatchResponse()
        for req in request.requests:
            resp.responses.append(self.Reify(req, context))
        return resp

    def ReloadWeights(self, request, context):
        # `source` is a checkpoint path (.safetensors) if it exists, else a
        # string to re-seed random weights from.
        src = request.source
        with self._lock:
            self._reloads += 1
            if src and src.endswith(".safetensors") and os.path.exists(src):
                self._load(0, checkpoint=src)
            else:
                self._load(abs(hash(src)) % (2**31) if src else 0)
        return model_pb2.ReloadWeightsResponse(loaded_version=self.model_version)

    def Health(self, request, context):
        return model_pb2.HealthResponse(
            model_version=self.model_version,
            ready=True,
            game_id=self.game_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="playground model server")
    parser.add_argument("--port", type=int, default=7200)
    parser.add_argument("--game", required=True, choices=["connect4", "go9", "amazons", "arimaa"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1)

    servicer = ModelServicer(args.game, seed=args.seed, checkpoint=args.checkpoint)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    model_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info("model server listening on :%d (game=%s, %s)", args.port, args.game, servicer.model_version)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
