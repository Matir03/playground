"""ModelService server. Stub — fill in when we have a first NN architecture.

Run with `playground-model-server --port 7200 --game connect4`.
"""

from __future__ import annotations

import argparse
import logging
from concurrent import futures

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="playground model server")
    parser.add_argument("--port", type=int, default=7200)
    parser.add_argument("--game", required=True, choices=["connect4", "go9", "amazons", "arimaa"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info("model server stub: game=%s port=%d checkpoint=%s", args.game, args.port, args.checkpoint)

    # TODO(M2): wire up grpcio server and register the generated ModelService servicer.
    #   from playground.gen.playground.model.v1 import model_pb2_grpc
    #   server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    #   model_pb2_grpc.add_ModelServiceServicer_to_server(ModelServicer(...), server)
    #   server.add_insecure_port(f"[::]:{args.port}")
    #   server.start()
    #   server.wait_for_termination()
    raise SystemExit("model server is a stub — implement at M2")


if __name__ == "__main__":
    main()
