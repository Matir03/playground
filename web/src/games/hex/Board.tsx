/**
 * Hex 13×13 play surface. The Rust engine owns the rules (connection detection,
 * legality, terminal state); each seat is a human or one of its movers:
 *
 *   - "Placeholder" (EnginePlayer.SOLVER) — centre, then the first empty cell.
 *     Not a player; it just makes the seat answerable with nothing configured.
 *   - "MoHex 2.0" (EnginePlayer.EXTERNAL) — the reference Hex engine, driven
 *     over GTP by the engine process. Only available when it was started with
 *     `--mohex-bin`; otherwise the request fails and the error shows below the
 *     board. "Seconds/move" becomes MoHex's per-move time budget.
 *
 * There is no Hex net yet — that seat arrives with the AZ run.
 *
 * Wire format (must match crates/games/hex/src/game.rs):
 *   GameState.payload = 169 cell bytes (row-major; 0 empty, 1 black, 2 white),
 *     then to_move, winner, and 2 little-endian ply bytes.
 *   Move.payload = [1, r, c] to play at row r, col c ([0] is swap, which the
 *     engine only accepts under ALLOW_SWAP).
 *
 * Geometry: pointy-top hexagons, row r shifted right by half a cell, which
 * makes the on-screen neighbourhood exactly the engine's six offsets
 * (r,c±1) (r-1,c) (r-1,c+1) (r+1,c-1) (r+1,c). Black owns the top and bottom
 * edges, White the left and right ones.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { create } from "@bufbuild/protobuf";
import type { Client } from "@connectrpc/connect";
import { ConnectError } from "@connectrpc/connect";
import {
  GameId,
  type GameHandle,
  type GameState,
  type Move,
  MoveSchema,
  Outcome,
  Player,
} from "../../gen/playground/game/game_pb";
import {
  type EngineService,
  EnginePlayer,
  NewGameRequestSchema,
  ApplyMoveRequestSchema,
  RequestMoveRequestSchema,
  SearchBudgetSchema,
  type SearchStats,
} from "../../gen/playground/engine/engine_pb";

const N = 13;

type Cell = 0 | 1 | 2;
type Seat = "human" | "engine" | "mohex";

const SEAT_LABELS: Record<Seat, string> = {
  human: "Human",
  engine: "Placeholder",
  mohex: "MoHex 2.0",
};

// MoHex only answers if the engine was started with `--mohex-bin`; otherwise
// the RPC fails with a failed-precondition that we surface as-is.
const SEAT_PLAYER: Record<Exclude<Seat, "human">, EnginePlayer> = {
  engine: EnginePlayer.SOLVER,
  mohex: EnginePlayer.EXTERNAL,
};

const BLACK = "#1f2937";
const WHITE = "#f8fafc";
const EMPTY_FILL = "#e8dcc0";
const GRID = "#a1866f";

/** Decode the engine payload into a [row][col] grid plus to_move / winner. */
function decode(payload: Uint8Array): { board: Cell[][]; toMove: number; winner: number } {
  const board: Cell[][] = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => (payload[r * N + c] ?? 0) as Cell),
  );
  return { board, toMove: payload[N * N] ?? 1, winner: payload[N * N + 1] ?? 0 };
}

function cellName(r: number, c: number): string {
  return `${String.fromCharCode(97 + c)}${r + 1}`;
}

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case Outcome.FIRST_WINS: return "Black wins — top to bottom";
    case Outcome.SECOND_WINS: return "White wins — left to right";
    default: return "";
  }
}

// ── Hexagon geometry (pointy-top, circumradius R) ────────────────────────────

const R = 17;
const HEX_W = Math.sqrt(3) * R; // full width of a hexagon
const ROW_H = 1.5 * R; // vertical distance between rows
const PAD = 22;

/** Centre of cell (r, c). */
function centre(r: number, c: number): [number, number] {
  return [PAD + HEX_W / 2 + c * HEX_W + r * (HEX_W / 2), PAD + R + r * ROW_H];
}

/** The six corners, starting at the top and going clockwise. */
function corners(cx: number, cy: number): [number, number][] {
  const h = HEX_W / 2;
  return [
    [cx, cy - R],       // 0 top
    [cx + h, cy - R / 2], // 1 upper-right
    [cx + h, cy + R / 2], // 2 lower-right
    [cx, cy + R],       // 3 bottom
    [cx - h, cy + R / 2], // 4 lower-left
    [cx - h, cy - R / 2], // 5 upper-left
  ];
}

function hexPoints(r: number, c: number): string {
  const [cx, cy] = centre(r, c);
  return corners(cx, cy).map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
}

/** Edge k of a cell's hexagon, as a line from corner k to corner k+1. */
function edge(r: number, c: number, k: number) {
  const [cx, cy] = centre(r, c);
  const v = corners(cx, cy);
  const a = v[k]!;
  const b = v[(k + 1) % 6]!;
  return { x1: a[0], y1: a[1], x2: b[0], y2: b[1] };
}

// Which hexagon edges face off the board, per border. Corner 0 is the top, so
// edge 5 (upper-left→top) faces the missing (r-1,c) neighbour, edge 0 faces
// (r-1,c+1), edge 1 faces (r,c+1), and so on around.
const TOP_EDGES = [5, 0];
const BOTTOM_EDGES = [2, 3];
const LEFT_EDGES = [3, 4];
const RIGHT_EDGES = [0, 1];

const SVG_W = PAD * 2 + HEX_W * (N + (N - 1) / 2);
const SVG_H = PAD * 2 + ROW_H * (N - 1) + 2 * R;

export function HexBoard({ client }: { client: Client<typeof EngineService> }) {
  const [handle, setHandle] = useState<GameHandle | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [blackSeat, setBlackSeat] = useState<Seat>("human");
  const [whiteSeat, setWhiteSeat] = useState<Seat>("engine");
  const [thinking, setThinking] = useState(false);
  const [seconds, setSeconds] = useState(1);
  const [lastStats, setLastStats] = useState<SearchStats | null>(null);
  const [lastMove, setLastMove] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const engineTurnRef = useRef<string | null>(null);
  const prevBoardRef = useRef<Cell[][] | null>(null);

  const newGame = useCallback(async () => {
    setError(null);
    setBusy(true);
    setLastStats(null);
    setLastMove(null);
    prevBoardRef.current = null;
    try {
      const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.HEX }));
      setHandle(h);
      setState(await client.getState(h));
    } catch (e) {
      setError(e instanceof ConnectError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [client]);

  useEffect(() => {
    void newGame();
  }, [newGame]);

  const seatToMove = (s: GameState): Seat => (s.toMove === Player.FIRST ? blackSeat : whiteSeat);

  const sendMove = useCallback(
    async (payload: Uint8Array) => {
      if (!handle || !state || busy) return;
      if (state.outcome !== Outcome.IN_PROGRESS) return;
      // Whose turn it is already gates this; deliberately *not* gated on
      // `thinking`, which stays true for a tick after the engine's move lands
      // and would otherwise swallow an immediate reply.
      if (seatToMove(state) !== "human") return;
      setError(null);
      setBusy(true);
      try {
        const next = await client.applyMove(
          create(ApplyMoveRequestSchema, {
            handle,
            move: create(MoveSchema, { gameId: GameId.HEX, payload }),
          }),
        );
        setState(next);
      } catch (e) {
        setError(e instanceof ConnectError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [client, handle, state, busy, blackSeat, whiteSeat],
  );

  const play = (r: number, c: number) => void sendMove(new Uint8Array([1, r, c]));

  // Engine auto-move when it's the engine seat's turn.
  useEffect(() => {
    if (!handle || !state) return;
    if (state.outcome !== Outcome.IN_PROGRESS) return;
    const seat = seatToMove(state);
    if (seat === "human") return;

    const tag = `${handle.handleId}:${state.ply}`;
    if (engineTurnRef.current === tag) return;
    engineTurnRef.current = tag;

    const fromPly = state.ply;
    (async () => {
      setThinking(true);
      setError(null);
      await new Promise((r) => setTimeout(r, 150));
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          let chosen: Move | undefined;
          try {
            for await (const prog of client.requestMove(
              create(RequestMoveRequestSchema, {
                handle,
                player: SEAT_PLAYER[seat],
                budget: create(SearchBudgetSchema, {
                  maxMillis: Math.max(1, Math.round(seconds * 1000)),
                }),
              }),
              { signal: AbortSignal.timeout(30000) },
            )) {
              if (prog.kind.case === "stats") setLastStats(prog.kind.value);
              else if (prog.kind.case === "chosen") chosen = prog.kind.value;
            }
          } catch (streamErr) {
            if (!chosen) throw streamErr;
          }
          if (!chosen) throw new Error("engine returned no move");
          const next = await client.applyMove(
            create(ApplyMoveRequestSchema, { handle, move: chosen }),
            { signal: AbortSignal.timeout(30000) },
          );
          setState(next);
          break;
        } catch (e) {
          const cur = await client.getState(handle, { signal: AbortSignal.timeout(8000) }).catch(() => null);
          if (cur && cur.ply > fromPly) {
            setState(cur);
            break;
          }
          if (attempt === 3) setError(e instanceof ConnectError ? e.message : String(e));
        }
      }
      setThinking(false);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handle, state, blackSeat, whiteSeat, client]);

  const dec = state ? decode(state.payload) : null;
  const board = dec?.board ?? null;

  // The engine sends positions, not moves; diff against the previous board to
  // mark the stone that just appeared.
  useEffect(() => {
    if (!board) return;
    const prev = prevBoardRef.current;
    if (prev) {
      outer: for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          if (prev[r]![c] === 0 && board[r]![c] !== 0) {
            setLastMove([r, c]);
            break outer;
          }
        }
      }
    }
    prevBoardRef.current = board;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const inProgress = state?.outcome === Outcome.IN_PROGRESS;
  const humanTurn = !!state && inProgress && seatToMove(state) === "human";
  const engineSeatLabel = state ? SEAT_LABELS[seatToMove(state)] : "";
  const toMove = state?.toMove === Player.FIRST ? "Black" : "White";

  return (
    <section>
      <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <SeatPicker label="Black (first) — top↕bottom" seat={blackSeat} onChange={setBlackSeat} />
        <SeatPicker label="White (second) — left↔right" seat={whiteSeat} onChange={setWhiteSeat} />
        <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }}>
          Seconds/move
          <input type="number" min={0.1} step={0.1} value={seconds}
            onChange={(e) => setSeconds(Number(e.target.value))} style={{ width: 70 }} />
        </label>
        <button onClick={() => void newGame()} disabled={busy}>New game</button>
      </div>

      <div style={{ marginBottom: "0.5rem", minHeight: "1.5em" }}>
        <strong>
          {state == null ? "Loading…"
            : inProgress ? (thinking ? `${toMove} — ${engineSeatLabel} thinking…` : `${toMove} to move`)
            : outcomeLabel(state.outcome)}
        </strong>
        {lastMove && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>
            last: {cellName(lastMove[0], lastMove[1])}
          </span>
        )}
        {lastStats && lastStats.visits > 0 && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>
            {lastStats.visits.toLocaleString()} visits, eval {lastStats.bestValue.toFixed(2)}
          </span>
        )}
      </div>

      <svg width={SVG_W} height={SVG_H} style={{ maxWidth: "100%" }}>
        {/* Cells. */}
        {board?.flatMap((row, r) =>
          row.map((v, c) => (
            <polygon
              key={`${r}-${c}`}
              points={hexPoints(r, c)}
              fill={v === 1 ? BLACK : v === 2 ? WHITE : EMPTY_FILL}
              stroke={GRID}
              strokeWidth={1}
              style={{ cursor: v === 0 && humanTurn ? "pointer" : "default" }}
              onClick={() => v === 0 && humanTurn && play(r, c)}
            >
              <title>{cellName(r, c)}</title>
            </polygon>
          )),
        )}

        {/* Owned edges: White first so Black wins the shared corner strokes. */}
        {Array.from({ length: N }, (_, i) => (
          <g key={`edge-${i}`} strokeLinecap="round">
            {LEFT_EDGES.map((k) => <line key={`l${k}`} {...edge(i, 0, k)} stroke={BLACK} strokeWidth={5} />)}
            {RIGHT_EDGES.map((k) => <line key={`r${k}`} {...edge(i, N - 1, k)} stroke={BLACK} strokeWidth={5} />)}
            {LEFT_EDGES.map((k) => <line key={`lw${k}`} {...edge(i, 0, k)} stroke={WHITE} strokeWidth={3} />)}
            {RIGHT_EDGES.map((k) => <line key={`rw${k}`} {...edge(i, N - 1, k)} stroke={WHITE} strokeWidth={3} />)}
            {TOP_EDGES.map((k) => <line key={`t${k}`} {...edge(0, i, k)} stroke={BLACK} strokeWidth={5} />)}
            {BOTTOM_EDGES.map((k) => <line key={`b${k}`} {...edge(N - 1, i, k)} stroke={BLACK} strokeWidth={5} />)}
          </g>
        ))}

        {/* Last-move marker. */}
        {lastMove && board && (() => {
          const [r, c] = lastMove;
          const [cx, cy] = centre(r, c);
          return <circle cx={cx} cy={cy} r={4} fill={board[r]![c] === 1 ? WHITE : BLACK} opacity={0.85} />;
        })()}

        {/* Coordinates: column letters above, row numbers to the left. */}
        {Array.from({ length: N }, (_, i) => {
          const [tx, ty] = centre(0, i);
          const [lx, ly] = centre(i, 0);
          return (
            <g key={`coord-${i}`} fill="#8a8a8a" fontSize={10} fontFamily="system-ui, sans-serif">
              <text x={tx} y={ty - R - 6} textAnchor="middle">{String.fromCharCode(97 + i)}</text>
              <text x={lx - HEX_W / 2 - 6} y={ly + 3} textAnchor="end">{i + 1}</text>
            </g>
          );
        })}
      </svg>

      {error && (
        <p style={{ color: "crimson", marginTop: "0.75rem" }}>
          Error: {error}
          {error.includes("no external engine") && (
            <>
              <br />
              Start it with <code>hex-engine serve --mohex-bin $(which mohex)</code> (build MoHex
              first: <code>scripts/fetch-mohex.sh</code>).
            </>
          )}
        </p>
      )}
    </section>
  );
}

function SeatPicker({ label, seat, onChange }: { label: string; seat: Seat; onChange: (s: Seat) => void }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }}>
      {label}
      <select value={seat} onChange={(e) => onChange(e.target.value as Seat)}>
        {(Object.keys(SEAT_LABELS) as Seat[]).map((s) => (
          <option key={s} value={s}>{SEAT_LABELS[s]}</option>
        ))}
      </select>
    </label>
  );
}
