/**
 * Connect-ES transports + typed service clients.
 *
 * Engine servers (Rust, tonic + tonic-web) speak gRPC-Web. The Connect-ES
 * gRPC-Web transport is what we use on the wire; the Connect developer
 * experience (typed clients, streaming, nice errors) is what we wanted.
 *
 * Generated types come from schemas/proto via `scripts/gen-proto.sh`.
 */

import { createClient } from "@connectrpc/connect";
import { createGrpcWebTransport } from "@connectrpc/connect-web";

// These imports resolve once codegen has been run.
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — generated module only exists after `scripts/gen-proto.sh`.
import { EngineService } from "../gen/playground/engine/v1/engine_pb";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — generated module only exists after `scripts/gen-proto.sh`.
import { OrchestratorService } from "../gen/playground/orchestrator/v1/orchestrator_pb";

export function engineClientFor(baseUrl: string) {
  const transport = createGrpcWebTransport({ baseUrl });
  return createClient(EngineService, transport);
}

export function orchestratorClient(baseUrl: string = "http://127.0.0.1:7300") {
  const transport = createGrpcWebTransport({ baseUrl });
  return createClient(OrchestratorService, transport);
}
