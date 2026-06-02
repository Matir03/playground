/**
 * Connect 4 play surface. Hot-seat human vs human; the Rust engine owns the
 * rules (legality, win detection) and is the single source of truth for state.
 *
 * Wire format (must match crates/games/connect4/src/game.rs):
 *   GameState.payload = 42 cell bytes (row-major, row 0 = bottom) + 2 ply bytes.
 *   cell value: 0 empty, 1 first (yellow), 2 second (red).
 *   Move.payload   = 1 byte: the column index.
 */

import { useCallback, useEffect, useState } from "react";
import { create } from "@bufbuild/protobuf";
import type { Client } from "@connectrpc/connect";
import { ConnectError } from "@connectrpc/connect";
import {
  GameId,
  type GameHandle,
  type GameState,
  MoveSchema,
  Outcome,
  Player,
} from "../../gen/playground/game/game_pb";
import {
  type EngineService,
  NewGameRequestSchema,
  ApplyMoveRequestSchema,
} from "../../gen/playground/engine/engine_pb";

const ROWS = 6;
const COLS = 7;

type Cell = 0 | 1 | 2;

/** Decode the engine's opaque payload into a [row][col] grid (row 0 = bottom). */
function decodeBoard(payload: Uint8Array): Cell[][] {
  const grid: Cell[][] = Array.from({ length: ROWS }, () =>
    Array.from({ length: COLS }, () => 0 as Cell),
  );
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      grid[r][c] = (payload[r * COLS + c] ?? 0) as Cell;
    }
  }
  return grid;
}

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case Outcome.FIRST_WINS:
      return "Yellow wins";
    case Outcome.SECOND_WINS:
      return "Red wins";
    case Outcome.DRAW:
      return "Draw";
    default:
      return "";
  }
}

export function Connect4Board({
  client,
}: {
  client: Client<typeof EngineService>;
}) {
  const [handle, setHandle] = useState<GameHandle | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const newGame = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const h = await client.newGame(
        create(NewGameRequestSchema, { gameId: GameId.CONNECT4 }),
      );
      setHandle(h);
      const s = await client.getState(h);
      setState(s);
    } catch (e) {
      setError(e instanceof ConnectError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [client]);

  useEffect(() => {
    void newGame();
  }, [newGame]);

  const drop = useCallback(
    async (col: number) => {
      if (!handle || !state || busy) return;
      if (state.outcome !== Outcome.IN_PROGRESS) return;
      setError(null);
      setBusy(true);
      try {
        const next = await client.applyMove(
          create(ApplyMoveRequestSchema, {
            handle,
            move: create(MoveSchema, {
              gameId: GameId.CONNECT4,
              payload: new Uint8Array([col]),
            }),
          }),
        );
        setState(next);
      } catch (e) {
        setError(e instanceof ConnectError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [client, handle, state, busy],
  );

  const board = state ? decodeBoard(state.payload) : null;
  const inProgress = state?.outcome === Outcome.IN_PROGRESS;
  const toMove = state?.toMove === Player.FIRST ? "Yellow" : "Red";
  const colFull = (c: number) => !!board && board[ROWS - 1][c] !== 0;

  return (
    <section>
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.75rem" }}>
        <strong>
          {state == null
            ? "Loading…"
            : inProgress
              ? `${toMove} to move`
              : outcomeLabel(state.outcome)}
        </strong>
        <button onClick={() => void newGame()} disabled={busy}>
          New game
        </button>
      </div>

      <div
        style={{
          display: "inline-block",
          background: "#1d4ed8",
          padding: 8,
          borderRadius: 8,
        }}
      >
        {/* Render top row first (row index ROWS-1 .. 0). */}
        {board &&
          Array.from({ length: ROWS }, (_, i) => ROWS - 1 - i).map((r) => (
            <div key={r} style={{ display: "flex" }}>
              {Array.from({ length: COLS }, (_, c) => (
                <Disc key={c} v={board[r][c]} />
              ))}
            </div>
          ))}

        {/* Column drop buttons. */}
        <div style={{ display: "flex" }}>
          {Array.from({ length: COLS }, (_, c) => (
            <button
              key={c}
              onClick={() => void drop(c)}
              disabled={busy || !inProgress || colFull(c)}
              title={`Drop in column ${c + 1}`}
              style={{
                width: 44,
                height: 28,
                margin: 2,
                cursor: busy || !inProgress || colFull(c) ? "default" : "pointer",
              }}
            >
              ↓
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p style={{ color: "crimson", marginTop: "0.75rem" }}>Error: {error}</p>
      )}
    </section>
  );
}

function Disc({ v }: { v: Cell }) {
  const color = v === 1 ? "#facc15" : v === 2 ? "#ef4444" : "#0b2a8a";
  return (
    <div
      style={{
        width: 44,
        height: 44,
        margin: 2,
        borderRadius: "50%",
        background: color,
        boxShadow: "inset 0 2px 4px rgba(0,0,0,0.35)",
      }}
    />
  );
}
