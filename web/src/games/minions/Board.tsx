/**
 * Simplified Minions on one 7×7 hex board. Each seat is a human or the
 * engine's greedy built-in mover; the Rust engine owns the rules and validates
 * every move, so everything computed here is highlighting only.
 *
 * Wire format (must match crates/games/minions/src/game.rs):
 *   GameState.payload = 133 bytes (N=7)
 *     [0,CELLS)        cells   — 0 empty, else 1 + unit + 6*side
 *     [CELLS,2*CELLS)  pstate  — bit0 moved, bit1 attacked, bit2 exhausted, bits3-5 damage
 *     [98,102) money   — 2 × u16 LE
 *     [102,112) reinforcements — side0[5] then side1[5], zombie..ghost
 *     [112,124) nw_hist — 6 × i16 LE (net-worth diff, oldest first)
 *     124 side · 125 phase · [126,128) turn u16 LE · [128,132) ply u32 LE · 132 winner
 *   (offsets shown for N=7; the constants below derive them from `CELLS`)
 *   Move.payload = [0,from,to] move · [1,from,to] attack · [2,unit] buy
 *                  [3,unit,at] spawn · [4] end phase
 *
 * A *turn* is a sequence of moves, not one move: attack phase (walk and strike
 * with each piece), then spawn phase (buy units, place them next to a
 * spawner), then "End turn". The engine seat is therefore asked for a move once
 * per ply, not once per turn.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as H from "./history";
import { UNITS, BUYABLE } from "./units";
import { MoveList } from "./MoveList";
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
  AnalyzeRequestSchema,
  ApplyTurnRequestSchema,
  type AnalyzeResponse,
} from "../../gen/playground/engine/engine_pb";
import { AnalysisTree, KernelMatrix, displayOrder, type SortKey } from "./Analysis";
import { SearchConfigPanel } from "./SearchConfig";

/** Whose turn a position is in, which is what the record's turn boundaries
 *  are made of. Module level, so every render shares one function. */
const turnInfo: H.TurnInfo = (st) => {
  const dd = decode(st.payload);
  return { turn: dd.turn, side: dd.side };
};

// ── Rules mirror ──────────────────────────────────────────────────────────────

const N = 7;
const CELLS = N * N;
const PHASE_ATTACK = 0;

const DIRS: [number, number][] = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]];

const cellX = (i: number) => i % N;
const cellY = (i: number) => Math.floor(i / N);
const cellIdx = (x: number, y: number) => y * N + x;
const cellName = (i: number) => `${String.fromCharCode(97 + cellX(i))}${cellY(i)}`;

function hexDist(a: number, b: number): number {
  const dx = cellX(a) - cellX(b);
  const dy = cellY(a) - cellY(b);
  return Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dx + dy));
}

const NEIGHBORS: number[][] = Array.from({ length: CELLS }, (_, i) => {
  const out: number[] = [];
  for (const [dx, dy] of DIRS) {
    const x = cellX(i) + dx;
    const y = cellY(i) + dy;
    if (x >= 0 && x < N && y >= 0 && y < N) out.push(cellIdx(x, y));
  }
  return out;
});

/** Must match map::GRAVEYARD_XY. */
const GRAVEYARD_XY: [number, number][] = [
  [1, 4], [5, 2],
  [4, 1], [2, 5],
  [0, 3], [6, 3],
];
const IS_GRAVEYARD = (() => {
  const m = new Array<boolean>(CELLS).fill(false);
  for (const [x, y] of GRAVEYARD_XY) m[cellIdx(x, y)] = true;
  return m;
})();
const GRAVEYARDS_TO_WIN = 4;


interface Decoded {
  cells: Uint8Array;
  pstate: Uint8Array;
  money: [number, number];
  reinforcements: [number[], number[]];
  side: number;
  phase: number;
  turn: number;
  winner: number;
}

// Offsets are derived from the board size rather than written out, so resizing
// the board cannot leave the UI silently reading the wrong bytes. Field order
// must match `Minions::encode_state` in crates/games/minions/src/game.rs:
// cells, pstate, money, reinforcements, nw_hist, side, phase, turn, ply, winner.
const ADJ_WINDOW = 6;  // must match game::ADJ_WINDOW
const O_MONEY = 2 * CELLS;
const O_REINF = O_MONEY + 4;
const O_NW_HIST = O_REINF + 10;
const O_SIDE = O_NW_HIST + 2 * ADJ_WINDOW;

function decode(p: Uint8Array): Decoded {
  const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
  return {
    cells: p.subarray(0, CELLS),
    pstate: p.subarray(CELLS, 2 * CELLS),
    money: [dv.getUint16(O_MONEY, true), dv.getUint16(O_MONEY + 2, true)],
    reinforcements: [
      Array.from(p.subarray(O_REINF, O_REINF + 5)),
      Array.from(p.subarray(O_REINF + 5, O_REINF + 10)),
    ],
    side: p[O_SIDE],
    phase: p[O_SIDE + 1],
    turn: dv.getUint16(O_SIDE + 2, true),
    winner: p[O_SIDE + 8],
  };
}

const pieceSide = (c: number) => Math.floor((c - 1) / 6);
const pieceUnit = (c: number) => (c - 1) % 6;
const hasMoved = (p: number) => (p & 1) !== 0;
const hasAttacked = (p: number) => (p & 2) !== 0;
const isExhausted = (p: number) => (p & 4) !== 0;
const damageOf = (p: number) => (p >> 3) & 7;

function canMovePiece(d: Decoded, i: number): boolean {
  return (d.pstate[i] & 0b111) === 0;
}

function canAttackPiece(d: Decoded, i: number): boolean {
  const p = d.pstate[i];
  if (isExhausted(p) || hasAttacked(p)) return false;
  return !(UNITS[pieceUnit(d.cells[i])].lumbering && hasMoved(p));
}

/** Empty cells reachable within speed; enemies block the path unless flying. */
function moveTargets(d: Decoded, from: number): number[] {
  const c = d.cells[from];
  if (c === 0) return [];
  const side = pieceSide(c);
  const u = UNITS[pieceUnit(c)];
  const seen = new Array<boolean>(CELLS).fill(false);
  seen[from] = true;
  let frontier = [from];
  const out: number[] = [];
  for (let step = 0; step < u.speed && frontier.length; step++) {
    const next: number[] = [];
    for (const cur of frontier) {
      for (const n of NEIGHBORS[cur]) {
        if (seen[n]) continue;
        const occ = d.cells[n];
        if (occ !== 0 && !u.flying && pieceSide(occ) !== side) continue;
        seen[n] = true;
        if (occ === 0) out.push(n);
        next.push(n);
      }
    }
    frontier = next;
  }
  return out;
}

function attackTargets(d: Decoded, from: number): number[] {
  const c = d.cells[from];
  if (c === 0 || !canAttackPiece(d, from)) return [];
  const side = pieceSide(c);
  const range = UNITS[pieceUnit(c)].range;
  const out: number[] = [];
  for (let i = 0; i < CELLS; i++) {
    const t = d.cells[i];
    if (t !== 0 && pieceSide(t) !== side && hexDist(from, i) <= range) out.push(i);
  }
  return out;
}

function spawnCells(d: Decoded, side: number): number[] {
  const seen = new Array<boolean>(CELLS).fill(false);
  const out: number[] = [];
  for (let i = 0; i < CELLS; i++) {
    const c = d.cells[i];
    if (c === 0 || pieceSide(c) !== side || !UNITS[pieceUnit(c)].spawn) continue;
    for (const n of NEIGHBORS[i]) {
      if (d.cells[n] === 0 && !seen[n]) {
        seen[n] = true;
        out.push(n);
      }
    }
  }
  return out;
}

function graveyardsHeld(d: Decoded, side: number): number {
  let n = 0;
  for (let i = 0; i < CELLS; i++) {
    if (IS_GRAVEYARD[i] && d.cells[i] !== 0 && pieceSide(d.cells[i]) === side) n++;
  }
  return n;
}

// ── Presentation ──────────────────────────────────────────────────────────────

type Seat = "human" | "solver" | "net";
const SEAT_LABELS: Record<Seat, string> = {
  human: "Human",
  solver: "Greedy bot",
  net: "Trained net",
};
// The net is a *turn*-level policy, so it goes through EXTERNAL (the engine
// drives it in-process from local weights) rather than NN, which is the model
// server's per-move interface and cannot express a whole-turn action.
const SEAT_PLAYER: Record<"solver" | "net", EnginePlayer> = {
  solver: EnginePlayer.SOLVER,
  net: EnginePlayer.EXTERNAL,
};

const SIDE_NAME = ["Yellow (first)", "Blue (second)"];
const SIDE_FILL = ["#e0a72c", "#3f74d8"];
const SIDE_INK = ["#4a3506", "#ffffff"];

/**
 * A side's net worth: gold plus every unit it owns at cost, on the board and in
 * reinforcements. Mirrors `net_worth` in crates/games/minions/src/game.rs — the
 * cap is adjudicated on this, so the UI has to explain the result in the same
 * terms the engine decided it in.
 */
function netWorth(d: Decoded, side: number): number {
  let total = d.money[side];
  for (const c of d.cells) {
    if (c !== 0 && pieceSide(c) === side) total += UNITS[pieceUnit(c)].cost;
  }
  // Reinforcement slots follow BUYABLE: zombie, skeleton, serpent, warg, ghost,
  // i.e. unit indices 1..5.
  d.reinforcements[side].forEach((n, slot) => {
    total += n * UNITS[slot + 1].cost;
  });
  return total;
}

/** Which of the three endings this was, read back off the final position. */
function endReason(d: Decoded, o: Outcome): string {
  if (o === Outcome.DRAW)
    return `turn ${d.turn} cap reached with net worth level at $${netWorth(d, 0)}`;
  const loser = o === Outcome.FIRST_WINS ? 1 : 0;
  const necroAlive = Array.from(d.cells).some(
    (c) => c !== 0 && pieceSide(c) === loser && pieceUnit(c) === 0,
  );
  const winner = 1 - loser;
  if (!necroAlive) return `${SIDE_NAME[loser]}'s necromancer was destroyed`;
  if (graveyardsHeld(d, winner) >= GRAVEYARDS_TO_WIN) {
    return `${SIDE_NAME[winner]} held ${graveyardsHeld(d, winner)} of the 6 graveyards at the end of ${SIDE_NAME[loser]}'s turn`;
  }
  return `turn ${d.turn} cap reached — ${SIDE_NAME[winner]} ahead on net worth, $${netWorth(d, winner)} to $${netWorth(d, loser)}`;
}

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case Outcome.FIRST_WINS: return "Yellow takes the board";
    case Outcome.SECOND_WINS: return "Blue takes the board";
    case Outcome.DRAW: return "Draw — level on net worth at the cap";
    default: return "";
  }
}

// Flat-top hex layout. Axial (x, y) → pixels with +x running right and +y
// running straight up, which keeps the engine's six neighbour offsets adjacent:
// (±1,0) are the up-right/down-left diagonals, (0,±1) are straight up/down, and
// (+1,-1)/(-1,+1) are the other diagonal pair.
const R = 21;
const HEX_H = Math.sqrt(3) * R; // flat-to-flat height of one hex
const PAD = 10;
/** Rows of "half-height steps" spanned by the whole rhombus: (N-1)/2 + (N-1). */
const V_SPAN = (N - 1) / 2 + (N - 1);
const px = (i: number) => PAD + R + 1.5 * R * cellX(i);
const py = (i: number) => PAD + HEX_H / 2 + HEX_H * (V_SPAN - (cellX(i) / 2 + cellY(i)));
const SVG_W = PAD * 2 + 2 * R + 1.5 * R * (N - 1);
const SVG_H = PAD * 2 + HEX_H * (V_SPAN + 1);

const HEX_POINTS = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 180) * 60 * k;
  return `${(R * Math.cos(a)).toFixed(2)},${(R * Math.sin(a)).toFixed(2)}`;
}).join(" ");

export function MinionsBoard({ client }: { client: Client<typeof EngineService> }) {
  const [handle, setHandle] = useState<GameHandle | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [firstSeat, setFirstSeat] = useState<Seat>("human");
  const [secondSeat, setSecondSeat] = useState<Seat>("solver");
  const [thinking, setThinking] = useState(false);
  /** Bot eval carried across turns so the delta a move caused stays readable.
   *
   * Keyed by *game turn*, not ply: a Minions turn is a sequence of actions, so
   * the engine seat answers many plies per turn and the stats stream arrives
   * several times within one of them. Only the last value of a turn is kept,
   * along with the side that produced it — with two engine seats the sign flips
   * between them, and comparing across sides would be meaningless. */
  const [evalTurn, setEvalTurn] = useState<{ turn: number; side: number; value: number } | null>(null);
  const [prevEval, setPrevEval] = useState<{ turn: number; side: number; value: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // ── Analysis mode ──────────────────────────────────────────────────────────
  //
  // Looking without playing. `analysis` is the search on the game's current
  // position; `walk` is the variation being viewed, which never touches the
  // game — the engine applies it to a copy and the board just renders what
  // comes back.
  const [analysisOn, setAnalysisOn] = useState(false);
  // Which of self-play's two searches to analyse with. Both run under the
  // engine's search config (the panel in the analysis view), which starts as
  // self-play's recorded search.
  const [fast, setFast] = useState(false);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analysing, setAnalysing] = useState(false);
  const [walk, setWalk] = useState<{ line: Uint8Array[]; labels: string[]; at: AnalyzeResponse } | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<SortKey>("visits");
  const [overlay, setOverlay] = useState<"none" | "hex0" | "hex1">("none");

  /** Attack phase: the selected piece. Spawn phase: the unit picked to place. */
  const [selected, setSelected] = useState<number | null>(null);
  const [spawnUnit, setSpawnUnit] = useState<number | null>(null);

  const engineTurnRef = useRef<string | null>(null);

  // ── Record ────────────────────────────────────────────────────────────────
  //
  // Every move that reaches the server is recorded here, whoever played it.
  // The tree owns the positions; `handle` is just whichever one the server
  // happens to be sitting on, tracked by `handleAt` so a sync can be skipped
  // when it would be a no-op. The server has no rewind, so moving to an
  // earlier node means replaying from a fresh game -- `ensureAt` does that,
  // and only when a position is actually used rather than merely looked at.
  const [hist, setHist] = useState<H.History | null>(null);
  const handleAt = useRef<number | null>(null);
  /** Suspends the engine seat without leaving analysis or losing the record.
   *  An in-flight search is aborted too, or "pause" would only take effect
   *  after the move it was meant to stop. */
  const [paused, setPaused] = useState(false);
  const engineAbort = useRef<AbortController | null>(null);
  const [syncing, setSyncing] = useState(false);

  const newGame = useCallback(async () => {
    setError(null);
    setBusy(true);
    setEvalTurn(null);
    setPrevEval(null);
    setSelected(null);
    setSpawnUnit(null);
    try {
      const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.MINIONS }));
      setHandle(h);
      const st = await client.getState(h);
      setState(st);
      setHist(H.newHistory(st));
      handleAt.current = 0;
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

  /** Move the board to a recorded position.
   *
   * Purely local: the state came back from the server when the move was played
   * and is kept verbatim, so browsing never waits on a round trip. The handle
   * is left where it was and re-synced only if the position gets used. Any
   * variation being walked in the analyser is dropped, since it was rooted at
   * the position we just left. */
  const goto = useCallback(
    (id: number) => {
      if (!hist) return;
      // Whole turns are the only places to stop; anything else snaps back to
      // the start of the turn it sits in.
      const to = H.isStop(hist, id, turnInfo) ? id : H.turnStart(hist, id, turnInfo);
      const n = H.node(hist, to);
      if (!n) return;
      setHist({ ...hist, current: to });
      setState(n.state);
      setSelected(null);
      setSpawnUnit(null);
      setWalk(null);
      setAnalysis(null);
      setExpanded(new Set());
    },
    [hist],
  );

  /** Arrow keys walk the record. Skipped while a text field has focus, so the
   *  search config's inputs still take arrow keys as arrow keys. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (!hist) return;
      const to =
        e.key === "ArrowLeft" ? H.prevId(hist, hist.current, turnInfo) : H.nextId(hist, hist.current, turnInfo);
      if (to === null) return;
      goto(to);
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hist, goto]);

  const atTip = !!hist && hist.current === H.tipOf(hist, hist.current);
  const canBack = !!hist && H.prevId(hist, hist.current, turnInfo) !== null;
  const canFwd = !!hist && H.nextId(hist, hist.current, turnInfo) !== null;
  /** Analysis, and committing a turn from it, start only from a whole turn. A
   *  tip partway through a turn is where the game is, but no search or net is
   *  ever asked about a half-played turn. */
  const canAnalyse = !!hist && H.isBoundary(hist, hist.current, turnInfo);

  /** Bring the server to `id`, replaying from a fresh game when it must.
   *
   * Cheap in the common case -- playing on at the tip leaves the handle where
   * it already is. Stepping back and playing there is the expensive one, and
   * it is a full replay because the engine holds one position per handle and
   * cannot rewind. Returns the handle to use, or null if the sync failed;
   * callers must not fall back to the stale handle, which is pointing at a
   * different position entirely. */
  const ensureAt = useCallback(
    async (id: number): Promise<GameHandle | null> => {
      if (!hist) return null;
      if (handle && handleAt.current === id) return handle;
      setSyncing(true);
      try {
        const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.MINIONS }));
        for (const step of H.pathMoves(hist, id)) {
          if (step.kind === "turn") {
            await client.applyTurn(create(ApplyTurnRequestSchema, { handle: h, turn: step.move }));
          } else {
            await client.applyMove(
              create(ApplyMoveRequestSchema, {
                handle: h,
                move: create(MoveSchema, { gameId: GameId.MINIONS, payload: step.move }),
              }),
            );
          }
        }
        setHandle(h);
        handleAt.current = id;
        return h;
      } catch (e) {
        setError(e instanceof ConnectError ? e.message : String(e));
        return null;
      } finally {
        setSyncing(false);
      }
    },
    [client, handle, hist],
  );

  const sendMove = useCallback(
    async (payload: Uint8Array) => {
      if (!hist) return;
      setError(null);
      setBusy(true);
      try {
        const h = await ensureAt(hist.current);
        if (!h) return;
        const next = await client.applyMove(
          create(ApplyMoveRequestSchema, {
            handle: h,
            move: create(MoveSchema, { gameId: GameId.MINIONS, payload }),
          }),
        );
        setState(next);
        setHist((cur) => {
          if (!cur) return cur;
          const after = H.push(cur, payload, next);
          handleAt.current = after.current;
          return after;
        });
        setSelected(null);
        // One action puts the board partway through a turn, where nothing the
        // analyser showed applies any more.
        setAnalysis(null);
        setWalk(null);
        setExpanded(new Set());
      } catch (e) {
        setError(e instanceof ConnectError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [client, hist, ensureAt],
  );

  // The engine seat answers one *action* at a time, so this fires once per ply
  // until the turn passes back. Mirrors amazons/Board.tsx.
  useEffect(() => {
    if (!handle || !state) return;
    // Analysis mode suspends the engine seat. The whole point is to look at a
    // position without committing to it, and a seat that keeps answering would
    // move the game out from under the analysis being read.
    if (analysisOn) return;
    // Paused is separate from analysis: it stops the seat while leaving the
    // board, the record and the analyser exactly as they are.
    if (paused) return;
    // Only ever play on at the tip of the line being viewed. Parked on an
    // earlier node, an engine seat would append moves to a position the user
    // is reading, which is the one thing browsing must not do.
    if (hist && hist.current !== H.tipOf(hist, hist.current)) return;
    if (state.outcome !== Outcome.IN_PROGRESS) return;
    const mover = seatToMove(state);
    if (mover === "human") return;

    const tag = `${handle.handleId}:${state.ply}`;
    if (engineTurnRef.current === tag) return;
    engineTurnRef.current = tag;

    const fromPly = state.ply;
    (async () => {
      setThinking(true);
      setError(null);
      const ac = new AbortController();
      engineAbort.current = ac;
      await new Promise((r) => setTimeout(r, 60));
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          let chosen: Move | undefined;
          try {
            for await (const prog of client.requestMove(
              create(RequestMoveRequestSchema, {
                handle,
                player: SEAT_PLAYER[mover],
                budget: create(SearchBudgetSchema, {
                  maxMillis: mover === "net" ? 20000 : 1000,
                }),
              }),
              { signal: AbortSignal.any([ac.signal, AbortSignal.timeout(30000)]) },
            )) {
              if (prog.kind.case === "stats") {
                const st = prog.kind.value;
                const dec = decode(state.payload);
                const { turn, side } = dec;
                setEvalTurn((cur) => {
                  if (cur && (cur.turn !== turn || cur.side !== side)) setPrevEval(cur);
                  return { turn, side, value: st.bestValue };
                });
              }
              else if (prog.kind.case === "chosen") chosen = prog.kind.value;
            }
          } catch (streamErr) {
            if (!chosen) throw streamErr;
          }
          if (ac.signal.aborted) break;
          if (!chosen) throw new Error("engine returned no move");
          const next = await client.applyMove(
            create(ApplyMoveRequestSchema, { handle, move: chosen }),
            { signal: AbortSignal.timeout(30000) },
          );
          setState(next);
          setHist((cur) => {
            if (!cur) return cur;
            const after = H.push(cur, chosen!.payload, next);
            handleAt.current = after.current;
            return after;
          });
          break;
        } catch (e) {
          if (ac.signal.aborted) break;
          const cur = await client.getState(handle, { signal: AbortSignal.timeout(8000) }).catch(() => null);
          if (cur && cur.ply > fromPly) {
            setState(cur);
            break;
          }
          if (attempt === 2) setError(e instanceof ConnectError ? e.message : String(e));
        }
      }
      if (engineAbort.current === ac) engineAbort.current = null;
      setThinking(false);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // `analysisOn` belongs here: leaving analysis must let the seat resume on
    // the position the game is actually in, and without the dependency the
    // effect would not re-run until something else changed.
  }, [handle, state, firstSeat, secondSeat, client, analysisOn, paused, hist]);

  // The board shows the variation when one is being walked, and the game
  // otherwise. Rendering the walked position rather than overlaying arrows on
  // the real one is the whole point of analysis mode: you see the position the
  // numbers describe.
  const shownState = walk?.at.state ?? state;
  const d = useMemo(() => (shownState ? decode(shownState.payload) : null), [shownState]);

  /** Aux for whatever position is on the board. */
  const aux = (walk?.at ?? analysis)?.aux;
  /** Value for the position on the board, from its side to move. */
  const shownValue = (walk?.at ?? analysis)?.value;

  const runAnalysis = useCallback(
    async (line: Uint8Array[], labels: string[], evalOnly: boolean) => {
      if (!hist || !H.isBoundary(hist, hist.current, turnInfo)) return;
      setAnalysing(true);
      setError(null);
      try {
        // Analysis is rooted wherever the board is, including an earlier node
        // or a variation, so the server has to be brought there first. This is
        // what makes "analyse from any position" work at all: `line` walks
        // forward from the handle and can never walk back.
        const h = await ensureAt(hist.current);
        if (!h) return;
        const res = await client.analyze(
          create(AnalyzeRequestSchema, {
            handle: h,
            line,
            // No budget, candidate count or kernel of its own: the engine's
            // search config decides all of them.
            evalOnly,
            fast,
          }),
          { signal: AbortSignal.timeout(120000) },
        );
        if (line.length === 0) {
          setAnalysis(res);
          setWalk(null);
          setExpanded(new Set());
        } else {
          setWalk({ line, labels, at: res });
        }
      } catch (err) {
        setError(err instanceof ConnectError ? err.message : String(err));
      } finally {
        setAnalysing(false);
      }
    },
    [client, hist, ensureAt, fast],
  );

  /** Play a root candidate for real, then re-root the analysis on the result.
   *
   * Distinct from walking a line: that reads a position the game is not in,
   * this advances the game. The re-analysis is a full search, not `eval_only`,
   * because the new position is now the one being studied. */
  const commitTurn = useCallback(
    async (turn: Uint8Array, label?: string) => {
      if (!hist || !H.isBoundary(hist, hist.current, turnInfo)) return;
      setAnalysing(true);
      setError(null);
      try {
        const h = await ensureAt(hist.current);
        if (!h) return;
        const next = await client.applyTurn(
          create(ApplyTurnRequestSchema, { handle: h, turn }),
          { signal: AbortSignal.timeout(30000) },
        );
        setState(next);
        // Recorded as a `turn` node: committing from the analyser off an
        // earlier position is exactly how a variation gets made, and the
        // replay has to know this one needs ApplyTurn rather than ApplyMove.
        setHist((cur) => {
          if (!cur) return cur;
          const after = H.push(cur, turn, next, "turn", label);
          handleAt.current = after.current;
          return after;
        });
        setWalk(null);
        setExpanded(new Set());
        setAnalysis(null);
        // The engine seat stays suspended while analysis is on, so this does
        // not hand the game to it — the new position simply becomes the root.
        setSelected(null);
        setSpawnUnit(null);
      } catch (err) {
        setError(err instanceof ConnectError ? err.message : String(err));
      } finally {
        setAnalysing(false);
      }
    },
    [client, hist, ensureAt],
  );
  const inProgress = state?.outcome === Outcome.IN_PROGRESS;
  const humanTurn = !!state && inProgress && seatToMove(state) === "human" && !busy && !thinking;

  // Highlights for the current selection.
  const moveSet = useMemo(
    () => new Set(d && selected != null && d.phase === PHASE_ATTACK && canMovePiece(d, selected) ? moveTargets(d, selected) : []),
    [d, selected],
  );
  const attackSet = useMemo(
    () => new Set(d && selected != null && d.phase === PHASE_ATTACK ? attackTargets(d, selected) : []),
    [d, selected],
  );
  const spawnHeld = d && spawnUnit != null ? d.reinforcements[d.side][spawnUnit - 1] : 0;
  const spawnSet = useMemo(
    () => new Set(d && d.phase !== PHASE_ATTACK && spawnUnit != null && spawnHeld > 0 ? spawnCells(d, d.side) : []),
    [d, spawnUnit, spawnHeld],
  );

  // The spawn-phase unit picker resets whenever the phase or side changes, or
  // when the last copy of the picked unit has been placed.
  useEffect(() => {
    setSpawnUnit(null);
    setSelected(null);
  }, [d?.phase, d?.side]);
  useEffect(() => {
    if (spawnUnit != null && spawnHeld === 0) setSpawnUnit(null);
  }, [spawnUnit, spawnHeld]);

  const onCell = (i: number) => {
    if (!humanTurn || !d) return;
    if (d.phase === PHASE_ATTACK) {
      if (moveSet.has(i) && selected != null) {
        void sendMove(new Uint8Array([0, selected, i]));
        return;
      }
      if (attackSet.has(i) && selected != null) {
        void sendMove(new Uint8Array([1, selected, i]));
        return;
      }
      const c = d.cells[i];
      setSelected(c !== 0 && pieceSide(c) === d.side ? i : null);
    } else if (spawnUnit != null && spawnSet.has(i)) {
      void sendMove(new Uint8Array([3, spawnUnit, i]));
    }
  };

  const endPhase = () => void sendMove(new Uint8Array([4]));
  const buy = (u: number) => void sendMove(new Uint8Array([2, u]));

  const me = d?.side ?? 0;
  const endLabel =
    d == null ? "End" : d.phase !== PHASE_ATTACK || d.turn === 0 ? "End turn" : "End attack phase";

  return (
    <section>
      <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <SeatPicker label={SIDE_NAME[0]} seat={firstSeat} onChange={setFirstSeat} />
        <SeatPicker label={SIDE_NAME[1]} seat={secondSeat} onChange={setSecondSeat} />
        <button onClick={() => void newGame()} disabled={busy}>New game</button>
        {inProgress && <button onClick={endPhase} disabled={!humanTurn}>{endLabel}</button>}
        {hist && (
          <span style={{ display: "flex", gap: 2, alignItems: "center" }} title="move through the record">
            <button onClick={() => goto(hist.root)} disabled={!canBack} title="start">⏮</button>
            <button onClick={() => goto(H.prevId(hist, hist.current, turnInfo)!)} disabled={!canBack} title="back one turn (←)">◀</button>
            <button onClick={() => goto(H.nextId(hist, hist.current, turnInfo)!)} disabled={!canFwd} title="forward one turn (→)">▶</button>
            <button onClick={() => goto(H.tipOf(hist, hist.current))} disabled={!canFwd} title="end of this line">⏭</button>
          </span>
        )}
        {/* Pause is not analysis: the seats stay as they are, the record stays
            as it is, and an in-flight search is cut off rather than allowed to
            land the move it was already committed to. */}
        <button
          onClick={() => {
            setPaused((v) => {
              if (!v) engineAbort.current?.abort();
              return !v;
            });
          }}
          title={paused ? "let the engine seats play again" : "stop the engine seats where they are"}
        >
          {paused ? "▶ resume engine" : "⏸ pause engine"}
        </button>
        {/* Analysis is a *view*, so leaving it drops the variation and puts the
            real game back on the board — the alternative is a board showing a
            position the game is not in with no obvious way back. */}
        <button
          onClick={() => {
            setAnalysisOn((on) => {
              if (on) {
                setWalk(null);
                setAnalysis(null);
                setExpanded(new Set());
              }
              return !on;
            });
          }}
          style={{ marginLeft: "auto" }}
          title="run the net on this position without playing it"
        >
          {analysisOn ? "exit analysis" : "analysis"}
        </button>
      </div>

      {state != null && d != null && !inProgress && (
        <div
          style={{
            display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem",
            margin: "0 0 0.75rem", padding: "0.75rem 1rem", borderRadius: 10,
            background: state.outcome === Outcome.DRAW ? "#4a4f57" : SIDE_FILL[state.outcome === Outcome.FIRST_WINS ? 0 : 1],
            color: state.outcome === Outcome.FIRST_WINS ? SIDE_INK[0] : "#fff",
          }}
        >
          <div>
            <div style={{ fontSize: "1.15rem", fontWeight: 700 }}>
              Game over — {outcomeLabel(state.outcome)}
            </div>
            <div style={{ fontSize: "0.85rem", opacity: 0.85 }}>{endReason(d, state.outcome)}</div>
          </div>
          <button onClick={() => void newGame()} disabled={busy} style={{ font: "inherit", padding: "0.4rem 0.9rem" }}>
            New game
          </button>
        </div>
      )}

      <div style={{ marginBottom: "0.5rem", minHeight: "1.5em" }}>
        <strong>
          {state == null || d == null ? "Loading…"
            : inProgress
              ? `${SIDE_NAME[d.side]} — ${d.phase === PHASE_ATTACK ? "attack" : "spawn"} phase, turn ${d.turn}${thinking ? " (engine thinking…)" : ""}`
              : outcomeLabel(state.outcome)}
        </strong>
        {evalTurn != null && (
          <span style={{ color: "#888", marginLeft: "0.75rem" }}>
            bot eval {evalTurn.value.toFixed(2)}
            {/* The previous turn's number is only comparable when the same side
                produced it — `bestValue` is from the mover's perspective. */}
            {prevEval != null && prevEval.side === evalTurn.side && (
              <>
                {" "}
                <span title={`turn ${prevEval.turn}`}>
                  (prev {prevEval.value.toFixed(2)}
                  {(() => {
                    const dv = evalTurn.value - prevEval.value;
                    const col = dv > 0.005 ? "#6ea86e" : dv < -0.005 ? "#c07070" : "#888";
                    return (
                      <span style={{ color: col }}>
                        {" "}{dv >= 0 ? "+" : "\u2212"}{Math.abs(dv).toFixed(2)}
                      </span>
                    );
                  })()}
                  )
                </span>
              </>
            )}
          </span>
        )}
      </div>

      {d && (
        <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", alignItems: "flex-start" }}>
          {/* Board and game state are one group: the analysis panel sits
              beside them as a third column rather than wrapping underneath,
              so reading the tree never means scrolling away from the board
              whose aux overlay it is driving. */}
          <div
            style={{
              display: "flex",
              gap: "1.5rem",
              flexWrap: "wrap",
              alignItems: "flex-start",
              flex: "0 1 auto",
              // Bounded, or the side panel's long lines (a whole turn in the
              // record) set the group's max-content width to the whole page
              // and push the analysis panel underneath anyway.
              maxWidth: 720,
            }}
          >
            <svg width={SVG_W} height={SVG_H} style={{ background: "#20242b", borderRadius: 8, flex: "0 0 auto" }}>
              {Array.from({ length: CELLS }, (_, i) => {
                const c = d.cells[i];
                const isMove = moveSet.has(i);
                const isAttack = attackSet.has(i);
                const isSpawn = spawnSet.has(i);
                const isSel = selected === i;
                const grave = IS_GRAVEYARD[i];
                let fill = grave ? "#4a4438" : "#333942";
                if (isSpawn) fill = "#2f5c3a";
                return (
                  <g key={i} transform={`translate(${px(i)},${py(i)})`}>
                    <polygon
                      points={HEX_POINTS}
                      fill={fill}
                      stroke={isSel ? "#f4f4f4" : isAttack ? "#d4553f" : "#161a20"}
                      strokeWidth={isSel || isAttack ? 2.5 : 1}
                    />
                    {grave && (
                      // An inner ring, not just the glyph, so a graveyard stays
                      // legible under a piece or a highlight dot.
                      <polygon points={HEX_POINTS} transform="scale(0.8)" fill="none" stroke="#c8b06a" strokeWidth={1.5} style={{ pointerEvents: "none" }} />
                    )}
                    {overlay !== "none" && aux && (() => {
                      // Per-cell control probability from the aux head, as a
                      // filled overlay. Drawn *under* the piece and the graveyard
                      // ring so it tints the cell without hiding what is on it.
                      //
                      // The head's channels are [mover, opponent], not
                      // [yellow, blue] — so which channel a colour wants depends
                      // on whose turn it is in the position being shown. Painting
                      // channel 0 yellow throughout swaps the sides on every
                      // blue-to-move position, which is what it did.
                      const side = overlay === "hex0" ? 0 : 1;
                      const chan = side === d.side ? aux.hexControlMover : aux.hexControlOpponent;
                      const v = chan[i];
                      if (v === undefined) return null;
                      const rgb = side === 0 ? "224,167,44" : "63,116,216";
                      return (
                        <polygon
                          points={HEX_POINTS}
                          fill={`rgba(${rgb},${(v * 0.72).toFixed(3)})`}
                          style={{ pointerEvents: "none" }}
                        />
                      );
                    })()}
                    {grave && c === 0 && !isMove && !isSpawn && (
                      <text textAnchor="middle" dy="0.35em" fontSize={15} fill="#9a8f74" style={{ pointerEvents: "none" }}>†</text>
                    )}
                    {c !== 0 && <Piece cell={c} pstate={d.pstate[i]} />}
                    {isMove && <circle r={5} fill="rgba(120,220,140,0.85)" style={{ pointerEvents: "none" }} />}
                    {isSpawn && <circle r={5} fill="rgba(150,230,170,0.9)" style={{ pointerEvents: "none" }} />}
                    {humanTurn && (
                      <polygon points={HEX_POINTS} fill="transparent" style={{ cursor: "pointer" }} onClick={() => onCell(i)} />
                    )}
                  </g>
                );
              })}
              {!inProgress && state != null && (
                <g style={{ pointerEvents: "none" }}>
                  <rect x={0} y={0} width={SVG_W} height={SVG_H} fill="rgba(16,18,22,0.74)" rx={8} />
                  <text x={SVG_W / 2} y={SVG_H / 2 - 6} textAnchor="middle" fontSize={20} fontWeight={700} fill="#fff">
                    Game over
                  </text>
                  <text x={SVG_W / 2} y={SVG_H / 2 + 20} textAnchor="middle" fontSize={15} fill="#e6e9ee">
                    {outcomeLabel(state.outcome)}
                  </text>
                </g>
              )}
            </svg>

            <div style={{ flex: "1 1 220px", minWidth: 220, fontSize: "0.85rem" }}>
              {[0, 1].map((side) => (
                <div key={side} style={{ marginBottom: "0.75rem", padding: "0.5rem 0.65rem", border: "1px solid #ddd", borderRadius: 8, background: side === me ? "#fbfbf5" : "#fff" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <strong style={{ color: SIDE_FILL[side] }}>{SIDE_NAME[side]}</strong>
                    <span style={{ color: "#666" }}>${d.money[side]}</span>
                  </div>
                  <div style={{ color: "#666", marginTop: 2 }}>
                    graveyards {graveyardsHeld(d, side)}/10 (win at {GRAVEYARDS_TO_WIN}) · income{" "}
                    ${graveyardsHeld(d, side) + 3}/turn
                  </div>
                  <div style={{ color: "#666", marginTop: 2 }}>
                    reinforcements:{" "}
                    {d.reinforcements[side].some((n) => n > 0)
                      ? BUYABLE.filter((u) => d.reinforcements[side][u - 1] > 0)
                          .map((u) => `${UNITS[u].name}×${d.reinforcements[side][u - 1]}`)
                          .join(", ")
                      : "—"}
                  </div>
                </div>
              ))}

              {inProgress && d.phase !== PHASE_ATTACK && (
                <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: "0.5rem 0.65rem" }}>
                  <strong>Spawn phase</strong>
                  <div style={{ color: "#666", margin: "2px 0 6px" }}>
                    Buy into reinforcements, then pick a unit and click a highlighted cell
                    next to one of your spawners.
                  </div>
                  {BUYABLE.map((u) => {
                    const def = UNITS[u];
                    const held = d.reinforcements[me][u - 1];
                    const affordable = d.money[me] >= def.cost;
                    return (
                      <div key={u} style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
                        <button onClick={() => buy(u)} disabled={!humanTurn || !affordable} style={{ width: 52 }}>
                          ${def.cost}
                        </button>
                        <button
                          onClick={() => setSpawnUnit(spawnUnit === u ? null : u)}
                          disabled={!humanTurn || held === 0}
                          style={{
                            flex: 1,
                            textAlign: "left",
                            fontWeight: spawnUnit === u ? 600 : 400,
                            background: spawnUnit === u ? "#e8f5ec" : undefined,
                          }}
                        >
                          {def.name} ×{held}
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}

              {inProgress && d.phase === PHASE_ATTACK && (
                <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: "0.5rem 0.65rem", color: "#666" }}>
                  <strong style={{ color: "#222" }}>Attack phase</strong>
                  <div style={{ marginTop: 2 }}>
                    Click one of your pieces, then a green dot to walk or a red-edged
                    hex to strike. Each piece moves once and attacks once; lumbering
                    pieces cannot do both.
                  </div>
                  {selected != null && d.cells[selected] !== 0 && (
                    <UnitCard cell={d.cells[selected]} pstate={d.pstate[selected]} at={selected} />
                  )}
                </div>
              )}

              {hist && (
                <div style={{ marginTop: "0.75rem" }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "baseline",
                      justifyContent: "space-between",
                      marginBottom: 4,
                    }}
                  >
                    <strong style={{ fontSize: "0.85rem" }}>Record</strong>
                    <span style={{ fontSize: "0.75rem", color: "#888" }}>
                      {syncing ? "syncing…" : atTip ? "at the end" : "viewing history"}
                    </span>
                  </div>
                  <MoveList
                    hist={hist}
                    info={turnInfo}
                    onGoto={goto}
                    onPromote={(id) => setHist((cur) => (cur ? H.promote(cur, id) : cur))}
                    onDrop={(id) => {
                      const dropped = H.drop(hist, id);
                      // Dropping a variation can leave `current` on its branch
                      // point, which may sit partway through a turn.
                      const after = {
                        ...dropped,
                        current: H.isStop(dropped, dropped.current, turnInfo)
                          ? dropped.current
                          : H.turnStart(dropped, dropped.current, turnInfo),
                      };
                      const n = H.node(after, after.current);
                      // The handle may have been sitting inside the subtree we
                      // just removed; forget it so the next use resyncs rather
                      // than trusting a position the record no longer has.
                      if (handleAt.current !== null && !after.nodes.has(handleAt.current)) {
                        handleAt.current = null;
                      }
                      setHist(after);
                      if (n) setState(n.state);
                    }}
                    sideFill={SIDE_FILL}
                  />
                  {!atTip && (
                    <div style={{ fontSize: "0.75rem", color: "#7a4ea8", marginTop: 4 }}>
                      Playing or committing a turn from here starts a variation. Engine
                      seats stay put until you return to the end of a line.
                    </div>
                  )}
                  <div style={{ marginTop: 6 }}>
                    <button
                      onClick={() => {
                        setAnalysisOn(true);
                        void runAnalysis([], [], false);
                      }}
                      disabled={analysing || syncing || !canAnalyse}
                      style={{ font: "inherit", padding: "0.25rem 0.7rem" }}
                      title={
                        canAnalyse
                          ? "search this position, wherever it sits in the record"
                          : "finish the turn first: analysis starts from whole turns"
                      }
                    >
                      {analysing ? "analysing…" : "analyse this position"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {analysisOn && (
            <div style={{ flex: "1 1 520px", minWidth: 460, fontSize: "0.85rem" }}>
              <div
                style={{
                  display: "flex",
                  gap: "0.6rem",
                  alignItems: "center",
                  flexWrap: "wrap",
                  marginBottom: "0.4rem",
                }}
              >
                <button
                  onClick={() => void runAnalysis([], [], false)}
                  disabled={analysing || !handle || !canAnalyse}
                  style={{ font: "inherit", padding: "0.25rem 0.7rem" }}
                >
                  {analysing ? "analysing…" : analysis ? "re-analyse" : "analyse"}
                </button>
                {!canAnalyse && (
                  <span style={{ color: "#b08030" }}>
                    partway through a turn — finish it, or step back (◀), to analyse
                  </span>
                )}
                <label
                  style={{ color: "#999" }}
                  title="Self-play searches a turn with its full search at full_search_prob and with its fast one otherwise. Both run under the search config below."
                >
                  search{" "}
                  <select
                    value={fast ? "fast" : "full"}
                    onChange={(e) => setFast(e.target.value === "fast")}
                    style={{ font: "inherit" }}
                  >
                    <option value="full">full</option>
                    <option value="fast">fast</option>
                  </select>
                </label>
                <label style={{ color: "#999" }}>
                  sort{" "}
                  <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} style={{ font: "inherit" }}>
                    <option value="visits">visits</option>
                    <option value="prior">prior</option>
                    <option value="value">value</option>
                  </select>
                </label>
                <label style={{ color: "#999" }}>
                  overlay{" "}
                  <select
                    value={overlay}
                    onChange={(e) => setOverlay(e.target.value as "none" | "hex0" | "hex1")}
                    style={{ font: "inherit" }}
                  >
                    <option value="none">none</option>
                    <option value="hex0">yellow control</option>
                    <option value="hex1">blue control</option>
                  </select>
                </label>
              </div>
              <SearchConfigPanel client={client} />

              {/* The position on the board, and what the nets say about it. The
                  value is always from *that* position's side to move, so it
                  flips sign down a variation — labelling the side avoids
                  reading a sign flip as a swing. */}
              {(analysis || walk) && (
                <div style={{ marginBottom: "0.45rem", color: "#bbb" }}>
                  {shownValue !== undefined && d && (
                    <span>
                      value{" "}
                      <strong style={{ color: SIDE_FILL[d.side] }}>{shownValue.toFixed(3)}</strong>{" "}
                      <span style={{ color: "#777" }}>for {SIDE_NAME[d.side]}</span>
                    </span>
                  )}
                  {aux && (
                    <span
                      style={{ marginLeft: "0.9rem", color: "#888" }}
                      title="predicted terminal score in points, from the side to move: net-worth differential plus 10 per turn the game ends before the cap. Unlike value, this keeps discriminating once a win is already certain."
                    >
                      score{" "}
                      <strong style={{ color: aux.score >= 0 ? "#6a9" : "#c77" }}>
                        {aux.score >= 0 ? "+" : ""}{aux.score.toFixed(0)}
                      </strong>
                    </span>
                  )}
                  {aux && (
                    <span style={{ marginLeft: "0.9rem", color: "#888" }} title="trajectory-balance log-partition estimate">
                      log Z {aux.logZ.toFixed(2)}
                    </span>
                  )}
                  {analysis && analysis.nodes.length > 0 && (
                    <span style={{ marginLeft: "0.9rem", color: "#888" }}>
                      {analysis.nodes.length} nodes · root {analysis.nodes[0].totalVisits} visits
                    </span>
                  )}
                </div>
              )}

              {/* Necromancer survival over the aux head's horizon: one bar per
                  side-turn ahead, so a line that gets a necromancer killed
                  shows as the curve collapsing rather than as a number. */}
              {aux && aux.necroHazardMover.length > 0 && d && (
                <div style={{ marginBottom: "0.5rem" }}>
                  <div style={{ color: "#777", fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    necromancer risk <span style={{ textTransform: "none" }}>(taller = more likely dead by then)</span>
                  </div>
                  {/* Hazard, not survival: the target stores `dead`, so a tall
                      bar is a necromancer in trouble. And the channels are
                      [mover, opponent], so each row picks its channel by
                      whether that side is the one to move here. */}
                  {[0, 1].map((side) => {
                    const curve =
                      side === d.side ? aux.necroHazardMover : aux.necroHazardOpponent;
                    return (
                      <div key={side} style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 26, marginTop: 2 }}>
                        <span style={{ color: SIDE_FILL[side], fontSize: "0.7rem", width: "2.6rem" }}>
                          {SIDE_NAME[side]}
                        </span>
                        {curve.map((p, i) => (
                          <div
                            key={i}
                            title={`dead within +${i + 1} side-turns: ${(p * 100).toFixed(0)}%`}
                            style={{
                              width: 10,
                              height: `${Math.max(2, p * 24)}px`,
                              background: SIDE_FILL[side],
                              opacity: 0.35 + 0.65 * p,
                            }}
                          />
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Breadcrumb of the variation being viewed. Clicking a step
                  truncates to it, so backing out of a line is one click rather
                  than a re-analyse. */}
              {walk && (
                <div style={{ marginBottom: "0.4rem", fontSize: "0.78rem" }}>
                  <span
                    onClick={() => setWalk(null)}
                    style={{ cursor: "pointer", color: "#8ab", marginRight: "0.3rem" }}
                  >
                    root
                  </span>
                  {walk.labels.map((l, i) => (
                    <span key={i}>
                      <span style={{ color: "#555" }}>→</span>{" "}
                      <span
                        onClick={() =>
                          void runAnalysis(walk.line.slice(0, i + 1), walk.labels.slice(0, i + 1), true)
                        }
                        style={{
                          cursor: "pointer",
                          color: i === walk.labels.length - 1 ? "#e0a72c" : "#8ab",
                          marginRight: "0.3rem",
                        }}
                        title={l}
                      >
                        {l.length > 22 ? `${l.slice(0, 22)}…` : l}
                      </span>
                    </span>
                  ))}
                </div>
              )}

              {analysis && analysis.nodes.length > 0 && (
                analysis.nodes[0].kernel.length > 0 ? (
                  <KernelMatrix
                    node={analysis.nodes[0]}
                    order={displayOrder(analysis.nodes[0], sort)}
                  />
                ) : (
                  <div style={{ color: "#666", fontSize: "0.72rem", marginBottom: "0.45rem" }}>
                    kernel off (kernel_sigma = 0) — candidates are scored
                    independently, as in self-play. Raise kernel_sigma in the search
                    config to pool statistics across similar turns and see the
                    influence matrix.
                  </div>
                )
              )}

              {analysis && analysis.nodes.length > 0 ? (
                <AnalysisTree
                  nodes={analysis.nodes}
                  chosen={analysis.chosen}
                  sort={sort}
                  expanded={expanded}
                  onToggle={(k) =>
                    setExpanded((prev) => {
                      const next = new Set(prev);
                      if (next.has(k)) next.delete(k);
                      else next.add(k);
                      return next;
                    })
                  }
                  onPlay={(line, labels) => void runAnalysis(line, labels, true)}
                  shown={walk?.line}
                  onCommit={(turn, label) => void commitTurn(turn, label)}
                  maxDepth={12}
                />
              ) : (
                <div style={{ color: "#777" }}>
                  {analysing ? "running the search…" : "press analyse to search this position"}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {error && <p style={{ color: "crimson", marginTop: "0.75rem" }}>Error: {error}</p>}
    </section>
  );
}

function Piece({ cell, pstate }: { cell: number; pstate: number }) {
  const side = pieceSide(cell);
  const def = UNITS[pieceUnit(cell)];
  const dmg = damageOf(pstate);
  const spent = isExhausted(pstate) || (hasMoved(pstate) && hasAttacked(pstate));
  return (
    <g style={{ pointerEvents: "none", opacity: spent ? 0.55 : 1 }}>
      <circle
        r={R - 5}
        fill={SIDE_FILL[side]}
        stroke={def.necromancer ? "#fff" : "rgba(0,0,0,0.35)"}
        strokeWidth={def.necromancer ? 2.5 : 1}
      />
      <text textAnchor="middle" dy="0.35em" fontSize={15} fontWeight={700} fill={SIDE_INK[side]}>
        {def.ch}
      </text>
      {dmg > 0 && (
        <>
          <circle cx={R - 7} cy={-(R - 8)} r={7} fill="#c0392b" />
          <text x={R - 7} y={-(R - 8)} textAnchor="middle" dy="0.35em" fontSize={9} fontWeight={700} fill="#fff">
            {dmg}
          </text>
        </>
      )}
    </g>
  );
}

function UnitCard({ cell, pstate, at }: { cell: number; pstate: number; at: number }) {
  const def = UNITS[pieceUnit(cell)];
  const keywords = [
    def.necromancer && "necromancer",
    def.lumbering && "lumbering",
    def.flying && "flying",
    def.persistent && "persistent",
    def.spawn && "spawn",
  ].filter(Boolean);
  return (
    <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid #eee", color: "#333" }}>
      <strong>{def.name}</strong> at {cellName(at)}
      <div style={{ color: "#666" }}>
        atk {def.attack} · def {def.defense}
        {damageOf(pstate) > 0 && ` (−${damageOf(pstate)})`} · spd {def.speed} · rng {def.range}
      </div>
      {keywords.length > 0 && <div style={{ color: "#888" }}>{keywords.join(", ")}</div>}
      <div style={{ color: "#888" }}>
        {hasMoved(pstate) ? "moved" : "can move"} · {hasAttacked(pstate) ? "attacked" : "can attack"}
        {isExhausted(pstate) && " · just spawned"}
      </div>
    </div>
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
