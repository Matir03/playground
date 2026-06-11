/**
 * 9×9 Go play surface. Each seat is a human or the in-process NN; the Rust
 * engine owns the rules (legality, ko, scoring) and, for the NN seat, picks the
 * move via the reification MCTS over the embedded candle net.
 *
 * Wire format (must match crates/games/go9/src/game.rs):
 *   GameState.payload = 81 board bytes (row-major; 0 empty, 1 black, 2 white),
 *     then 1 has-prev byte (+81 prev-board bytes if 1), then to_move,
 *     consecutive_passes, and 2 little-endian ply bytes.
 *   Move.payload = [0] for pass, or [1, r, c] to play at row r, col c.
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

const N = 9;
const STARS = [
  [2, 2], [2, 6], [6, 2], [6, 6], [4, 4],
]; // hoshi for 9×9

type Cell = 0 | 1 | 2;
type Seat = "human" | "nn";

const SEAT_LABELS: Record<Seat, string> = { human: "Human", nn: "NN — self-play" };

/** Decode the engine payload into a [row][col] grid plus to_move / passes. */
function decode(payload: Uint8Array): { board: Cell[][]; toMove: number; passes: number } {
  const board: Cell[][] = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => (payload[r * N + c] ?? 0) as Cell),
  );
  let i = N * N;
  const hasPrev = payload[i] ?? 0;
  i += 1;
  if (hasPrev === 1) i += N * N;
  const toMove = payload[i] ?? 1;
  const passes = payload[i + 1] ?? 0;
  return { board, toMove, passes };
}

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case Outcome.FIRST_WINS: return "Black wins";
    case Outcome.SECOND_WINS: return "White wins";
    case Outcome.DRAW: return "Draw";
    default: return "";
  }
}

export function Go9Board({ client }: { client: Client<typeof EngineService> }) {
  const [handle, setHandle] = useState<GameHandle | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [blackSeat, setBlackSeat] = useState<Seat>("human");
  const [whiteSeat, setWhiteSeat] = useState<Seat>("nn");
  const [thinking, setThinking] = useState(false);
  const [seconds, setSeconds] = useState(1);
  const [lastStats, setLastStats] = useState<SearchStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const engineTurnRef = useRef<string | null>(null);

  const newGame = useCallback(async () => {
    setError(null);
    setBusy(true);
    setLastStats(null);
    try {
      const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.GO9 }));
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
      if (!handle || !state || busy || thinking) return;
      if (state.outcome !== Outcome.IN_PROGRESS) return;
      if (seatToMove(state) !== "human") return;
      setError(null);
      setBusy(true);
      try {
        const next = await client.applyMove(
          create(ApplyMoveRequestSchema, {
            handle,
            move: create(MoveSchema, { gameId: GameId.GO9, payload }),
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
    [client, handle, state, busy, thinking, blackSeat, whiteSeat],
  );

  const play = (r: number, c: number) => void sendMove(new Uint8Array([1, r, c]));
  const pass = () => void sendMove(new Uint8Array([0]));

  // Engine auto-move when it's the NN seat's turn.
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
      await new Promise((r) => setTimeout(r, 200));
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
  const inProgress = state?.outcome === Outcome.IN_PROGRESS;
  const humanTurn = !!state && inProgress && seatToMove(state) === "human";
  const toMove = state?.toMove === Player.FIRST ? "Black" : "White";

  // SVG geometry.
  const P = 24;
  const cell = 36;
  const size = P * 2 + cell * (N - 1);
  const xy = (i: number) => P + i * cell;

  return (
    <section>
      <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <SeatPicker label="Black (first)" seat={blackSeat} onChange={setBlackSeat} />
        <SeatPicker label="White (second)" seat={whiteSeat} onChange={setWhiteSeat} />
        <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }}>
          NN seconds/move
          <input type="number" min={0.1} step={0.1} value={seconds}
            onChange={(e) => setSeconds(Number(e.target.value))} style={{ width: 70 }} />
        </label>
        <button onClick={() => void newGame()} disabled={busy}>New game</button>
        <button onClick={pass} disabled={!humanTurn || busy}>Pass</button>
      </div>

      <div style={{ marginBottom: "0.5rem", minHeight: "1.5em" }}>
        <strong>
          {state == null ? "Loading…"
            : inProgress ? (thinking ? `${toMove} (engine) thinking…` : `${toMove} to move`)
            : outcomeLabel(state.outcome)}
        </strong>
        {dec && inProgress && dec.passes > 0 && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>{dec.passes} consecutive pass(es)</span>
        )}
        {lastStats && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>
            last search: {lastStats.visits.toLocaleString()} visits
            {lastStats.reifications > 0 && `, ${lastStats.reifications} reified`}
            , eval {lastStats.bestValue.toFixed(2)}
          </span>
        )}
      </div>

      <svg width={size} height={size} style={{ background: "#dcb35c", borderRadius: 4 }}>
        {/* grid lines */}
        {Array.from({ length: N }, (_, i) => (
          <g key={i}>
            <line x1={xy(0)} y1={xy(i)} x2={xy(N - 1)} y2={xy(i)} stroke="#000" strokeWidth={1} />
            <line x1={xy(i)} y1={xy(0)} x2={xy(i)} y2={xy(N - 1)} stroke="#000" strokeWidth={1} />
          </g>
        ))}
        {/* star points */}
        {STARS.map(([r, c]) => (
          <circle key={`${r}-${c}`} cx={xy(c)} cy={xy(r)} r={3} fill="#000" />
        ))}
        {/* stones + click targets */}
        {board &&
          board.flatMap((row, r) =>
            row.map((v, c) => {
              const cx = xy(c);
              const cy = xy(r);
              return (
                <g key={`${r}-${c}`}>
                  {v !== 0 && (
                    <circle cx={cx} cy={cy} r={cell / 2 - 2}
                      fill={v === 1 ? "#111" : "#f7f7f7"}
                      stroke={v === 1 ? "#000" : "#999"} strokeWidth={1} />
                  )}
                  {v === 0 && humanTurn && (
                    <circle cx={cx} cy={cy} r={cell / 2 - 2} fill="transparent"
                      style={{ cursor: "pointer" }}
                      onClick={() => play(r, c)}
                      onMouseEnter={(e) => (e.currentTarget.setAttribute("fill", "rgba(0,0,0,0.15)"))}
                      onMouseLeave={(e) => (e.currentTarget.setAttribute("fill", "transparent"))} />
                  )}
                </g>
              );
            }),
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
