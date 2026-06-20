/**
 * 10×10 Game of the Amazons play surface. Each seat is a human or the in-process
 * AZ net (gen71); the Rust engine owns the rules and, for the NN seat, picks the
 * move via MCTS over the embedded candle net.
 *
 * Wire format (must match crates/games/amazons/src/game.rs):
 *   GameState.payload = 51 bytes: first u128 (16 LE) | second u128 (16) |
 *     arrows u128 (16) | to_move (1) | ply (2 LE). Bit r*10+c (row-major).
 *     first = player-1 queens, second = player-2 queens, arrows = burned squares.
 *   Move.payload = [from_r, from_c, to_r, to_c, arrow_r, arrow_c] (6 bytes).
 *
 * A turn is three clicks: pick one of your queens, pick its (queen-move)
 * destination, then pick the square to shoot the arrow into.
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

const N = 10;
const DIRS: [number, number][] = [
  [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1],
];

type Seat = "human" | "nn";
const SEAT_LABELS: Record<Seat, string> = { human: "Human", nn: "NN — gen71" };

interface Decoded {
  first: boolean[][]; // player-1 queens
  second: boolean[][]; // player-2 queens
  arrows: boolean[][]; // burned squares
  toMove: number; // 1 first, 2 second
}

/** Read bit (r*10+c) from the 16-byte little-endian u128 at `off` in payload. */
function bitAt(payload: Uint8Array, off: number, r: number, c: number): boolean {
  const i = r * N + c;
  return ((payload[off + (i >> 3)] >> (i & 7)) & 1) === 1;
}

function decode(payload: Uint8Array): Decoded {
  const grid = (off: number) =>
    Array.from({ length: N }, (_, r) =>
      Array.from({ length: N }, (_, c) => bitAt(payload, off, r, c)),
    );
  return { first: grid(0), second: grid(16), arrows: grid(32), toMove: payload[48] ?? 1 };
}

/** Squares reachable by a queen slide from (fr,fc) over empty cells. */
function rays(empty: (r: number, c: number) => boolean, fr: number, fc: number): [number, number][] {
  const out: [number, number][] = [];
  for (const [dr, dc] of DIRS) {
    let r = fr + dr;
    let c = fc + dc;
    while (r >= 0 && r < N && c >= 0 && c < N && empty(r, c)) {
      out.push([r, c]);
      r += dr;
      c += dc;
    }
  }
  return out;
}

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case Outcome.FIRST_WINS: return "White wins";
    case Outcome.SECOND_WINS: return "Black wins";
    case Outcome.DRAW: return "Draw";
    default: return "";
  }
}

const key = (r: number, c: number) => r * N + c;

export function AmazonsBoard({ client }: { client: Client<typeof EngineService> }) {
  const [handle, setHandle] = useState<GameHandle | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [firstSeat, setFirstSeat] = useState<Seat>("human");
  const [secondSeat, setSecondSeat] = useState<Seat>("nn");
  const [thinking, setThinking] = useState(false);
  const [seconds, setSeconds] = useState(2);
  const [lastStats, setLastStats] = useState<SearchStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Move-in-progress: queen source, then chosen destination.
  const [from, setFrom] = useState<[number, number] | null>(null);
  const [to, setTo] = useState<[number, number] | null>(null);

  const engineTurnRef = useRef<string | null>(null);

  const newGame = useCallback(async () => {
    setError(null);
    setBusy(true);
    setLastStats(null);
    setFrom(null);
    setTo(null);
    try {
      const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.AMAZONS }));
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

  const seatToMove = (s: GameState): Seat => (s.toMove === Player.FIRST ? firstSeat : secondSeat);

  const sendMove = useCallback(
    async (payload: Uint8Array) => {
      if (!handle) return;
      setError(null);
      setBusy(true);
      try {
        const next = await client.applyMove(
          create(ApplyMoveRequestSchema, {
            handle,
            move: create(MoveSchema, { gameId: GameId.AMAZONS, payload }),
          }),
        );
        setState(next);
        setFrom(null);
        setTo(null);
      } catch (e) {
        setError(e instanceof ConnectError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [client, handle],
  );

  // Engine auto-move when it's the NN seat's turn (mirrors go9/Board.tsx).
  useEffect(() => {
    if (!handle || !state) return;
    if (state.outcome !== Outcome.IN_PROGRESS) return;
    if (seatToMove(state) !== "nn") return;

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
                player: EnginePlayer.NN,
                model: "nn",
                budget: create(SearchBudgetSchema, {
                  maxMillis: Math.max(1, Math.round(seconds * 1000)),
                }),
              }),
              { signal: AbortSignal.timeout(60000) },
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
            { signal: AbortSignal.timeout(60000) },
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
  }, [handle, state, firstSeat, secondSeat, client]);

  const dec = state ? decode(state.payload) : null;
  const inProgress = state?.outcome === Outcome.IN_PROGRESS;
  const humanTurn = !!state && inProgress && seatToMove(state) === "human" && !busy && !thinking;
  const toMoveLabel = state?.toMove === Player.FIRST ? "White (first)" : "Black (second)";

  // Occupancy + the side-to-move's queens.
  const occupied = (r: number, c: number) =>
    !!dec && (dec.first[r][c] || dec.second[r][c] || dec.arrows[r][c]);
  const myQueens = dec ? (dec.toMove === 1 ? dec.first : dec.second) : null;

  // Highlighted targets for the current move phase.
  let targets = new Set<number>();
  if (humanTurn && dec) {
    if (from && !to) {
      for (const [r, c] of rays((rr, cc) => !occupied(rr, cc), from[0], from[1])) targets.add(key(r, c));
    } else if (from && to) {
      // Arrow targets: queen has moved from `from` to `to`.
      const emptyForArrow = (r: number, c: number) => {
        if (r === from[0] && c === from[1]) return true; // vacated
        if (r === to[0] && c === to[1]) return false; // queen now here
        return !occupied(r, c);
      };
      for (const [r, c] of rays(emptyForArrow, to[0], to[1])) targets.add(key(r, c));
    }
  }

  const onCell = (r: number, c: number) => {
    if (!humanTurn || !dec || !myQueens) return;
    if (!from) {
      if (myQueens[r][c]) setFrom([r, c]);
      return;
    }
    if (from && !to) {
      if (myQueens[r][c]) { setFrom([r, c]); return; } // reselect a queen
      if (targets.has(key(r, c))) setTo([r, c]);
      return;
    }
    if (from && to) {
      if (targets.has(key(r, c))) {
        void sendMove(new Uint8Array([from[0], from[1], to[0], to[1], r, c]));
      }
    }
  };

  // SVG geometry.
  const cell = 42;
  const pad = 18;
  const size = pad * 2 + cell * N;
  const cx = (c: number) => pad + c * cell + cell / 2;
  const cy = (r: number) => pad + r * cell + cell / 2;

  return (
    <section>
      <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <SeatPicker label="White (first)" seat={firstSeat} onChange={setFirstSeat} />
        <SeatPicker label="Black (second)" seat={secondSeat} onChange={setSecondSeat} />
        <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }}>
          NN seconds/move
          <input type="number" min={0.1} step={0.5} value={seconds}
            onChange={(e) => setSeconds(Number(e.target.value))} style={{ width: 70 }} />
        </label>
        <button onClick={() => void newGame()} disabled={busy}>New game</button>
        {(from || to) && <button onClick={() => { setFrom(null); setTo(null); }} disabled={!humanTurn}>Cancel move</button>}
      </div>

      <div style={{ marginBottom: "0.5rem", minHeight: "1.5em" }}>
        <strong>
          {state == null ? "Loading…"
            : inProgress ? (thinking ? `${toMoveLabel} (engine) thinking…` : `${toMoveLabel} to move`)
            : outcomeLabel(state.outcome)}
        </strong>
        {humanTurn && from && !to && <span style={{ color: "#888", marginLeft: "0.75rem" }}>pick a destination</span>}
        {humanTurn && from && to && <span style={{ color: "#888", marginLeft: "0.75rem" }}>pick where to shoot the arrow</span>}
        {lastStats && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>
            last search: {lastStats.visits.toLocaleString()} visits, eval {lastStats.bestValue.toFixed(2)}
          </span>
        )}
      </div>

      <svg width={size} height={size} style={{ background: "#b58863", borderRadius: 4 }}>
        {/* checkerboard */}
        {Array.from({ length: N }, (_, r) =>
          Array.from({ length: N }, (_, c) => (
            <rect key={`bg-${r}-${c}`} x={pad + c * cell} y={pad + r * cell} width={cell} height={cell}
              fill={(r + c) % 2 === 0 ? "#f0d9b5" : "#b58863"} />
          )),
        )}
        {/* arrows (burned squares) */}
        {dec && dec.arrows.flatMap((row, r) =>
          row.map((v, c) => v && (
            <rect key={`a-${r}-${c}`} x={pad + c * cell + 3} y={pad + r * cell + 3}
              width={cell - 6} height={cell - 6} fill="#5a3a22" rx={3} />
          )),
        )}
        {/* targets / clickable cells */}
        {dec && Array.from({ length: N }, (_, r) =>
          Array.from({ length: N }, (_, c) => {
            const isTarget = targets.has(key(r, c));
            const isFrom = from && from[0] === r && from[1] === c;
            const isTo = to && to[0] === r && to[1] === c;
            return (
              <g key={`t-${r}-${c}`}>
                {isFrom && <rect x={pad + c * cell} y={pad + r * cell} width={cell} height={cell} fill="rgba(80,160,80,0.45)" />}
                {isTo && <rect x={pad + c * cell} y={pad + r * cell} width={cell} height={cell} fill="rgba(80,120,200,0.45)" />}
                {isTarget && <circle cx={cx(c)} cy={cy(r)} r={6} fill="rgba(40,90,40,0.55)" style={{ pointerEvents: "none" }} />}
              </g>
            );
          }),
        )}
        {/* queens (visual only — clicks pass through to the click layer below) */}
        {dec && [0, 1].flatMap((side) => {
          const g = side === 0 ? dec.first : dec.second;
          return g.flatMap((row, r) =>
            row.map((v, c) => v && (
              <circle key={`q-${side}-${r}-${c}`} cx={cx(c)} cy={cy(r)} r={cell / 2 - 5}
                fill={side === 0 ? "#fafafa" : "#1b1b1b"} stroke={side === 0 ? "#999" : "#000"} strokeWidth={1.5}
                style={{ pointerEvents: "none" }} />
            )),
          );
        })}
        {/* click layer: ON TOP so queens/arrows never intercept selection */}
        {dec && humanTurn && Array.from({ length: N }, (_, r) =>
          Array.from({ length: N }, (_, c) => (
            <rect key={`click-${r}-${c}`} x={pad + c * cell} y={pad + r * cell}
              width={cell} height={cell} fill="transparent" style={{ cursor: "pointer" }}
              onClick={() => onCell(r, c)} />
          )),
        )}
      </svg>

      {error && <p style={{ color: "crimson", marginTop: "0.75rem" }}>Error: {error}</p>}
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
