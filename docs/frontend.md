# Frontend

## Scope

One React SPA serves:
- Game viewer + human-playable board for each game.
- Player-slot configuration: human, model, or external engine on either side.
- Live MCTS visualization during model moves (tree, visit counts,
  reification events).
- Run/eval dashboards (driven by `OrchestratorService`).

## Stack

- **Vite + React + TypeScript.**
- **@connectrpc/connect** + **@connectrpc/connect-web** for RPC. gRPC-Web
  transport so we can hit `tonic-web`-wrapped Rust engines directly. Same
  client flavor used against the Python orchestrator.
- Generated types + clients from `schemas/proto` into `web/src/gen/`.
- Board rendering: plain `<canvas>` or SVG per game. Start with SVG; drop to
  canvas only if we hit perf issues (unlikely for turn-based games).

## Project structure

```
web/
  package.json
  vite.config.ts
  tsconfig.json
  src/
    main.tsx          # app entry
    App.tsx           # top-level router
    lib/
      client.ts       # Connect transport + service clients
    gen/              # generated from schemas/proto (gitignored)
    games/
      connect4/       # per-game UI module (board, piece style, SGF-ish)
      go9/
      amazons/
      arimaa/
    pages/
      play/           # interactive play surface
      runs/           # orchestrator dashboards
      eval/           # ELO curves, tournament browser
```

## Why not plain WebSockets?

We already pay the protobuf tax for on-disk records and the Python side.
Getting typed clients + streaming + proper error semantics for free from
Connect-ES is a better deal than hand-rolling a WS envelope. See
[decisions.md](decisions.md#d4-grpc-web-from-the-browser).
