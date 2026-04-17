# playground

Research playground for a custom MuZero variant. We validate the algorithm on a
sequence of progressively richer 2-player abstract strategy games
(Connect4 → 9×9 Go → Amazons → Arimaa) before applying it to the target game.

- Rust engines per game with shared core libs (MCTS, recording, server).
- Python owns everything model-related: training, model server, orchestrator.
- Protobuf schemas via `buf`, typed clients on all three sides.
- ConnectRPC-compatible wire (gRPC-Web from Rust `tonic-web`).
- React web frontend for viewing and playing games with a human or model on either side.

See [docs/](docs/) for the full design. Start with [docs/overview.md](docs/overview.md).
