/** Replay viewer for matches recorded by `eval-match --record-file`.
 *
 * Deliberately engine-free. The recording carries the same
 * `GameState.payload` bytes the live board already decodes, so a match can be
 * reviewed with nothing running — which is the normal case here, since matches
 * are played while the ladder is paused and the engine is shut down
 * afterwards.
 *
 * Nothing here re-implements the board. `decode`, the hex geometry and
 * `Piece` are imported from `Board.tsx`, and the record is a `History` so the
 * existing `MoveList` does the navigating. The 133-byte state format is
 * already hand-duplicated between Rust and TypeScript; a second copy in this
 * file would be a third place to get it wrong.
 *
 * Produce a recording with:
 *   minions-engine eval-match --a <dir> --b <dir> --games 80 \
 *     --record-file web/public/recordings/<name>.json
 * then add it to web/public/recordings/index.json.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { create } from "@bufbuild/protobuf";
import {
  GameId,
  GameStateSchema,
  Outcome,
  type GameState,
} from "../../gen/playground/game/game_pb";
import { decode, px, py, HEX_POINTS, SVG_W, SVG_H, Piece, SIDE_FILL } from "./Board";
import { MoveList } from "./MoveList";
import * as H from "./history";

// ── the on-disk record (crates/games/minions/src/eval.rs) ────────────────────

interface GameRecord {
  game: number;
  bid: number;
  a_first: boolean;
  turns: number;
  /** From A's seat. */
  result: "a" | "b" | "draw";
  /** Index 0 is the opening; each later entry is the position after one turn.
   *  Length is NOT always `turns + 1` — a game ends the instant a necromancer
   *  dies, part-way through a turn — so this is walked, never indexed by turn. */
  states: string[];
}

interface MatchRecord {
  games: number;
  a_wins: number;
  b_wins: number;
  draws: number;
  a_score: number;
  a_label: string;
  b_label: string;
  records: GameRecord[];
}

interface IndexEntry {
  file: string;
  label: string;
  note?: string;
}

// ── record → History ─────────────────────────────────────────────────────────

function hexToBytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.substr(i * 2, 2), 16);
  return out;
}

/** The proto outcome for a finished game, from A's seat plus who moved first.
 *  `FIRST_WINS` is about yellow, not about seat A, so both fields are needed. */
function terminalOutcome(r: GameRecord): Outcome {
  if (r.result === "draw") return Outcome.DRAW;
  const aWon = r.result === "a";
  return aWon === r.a_first ? Outcome.FIRST_WINS : Outcome.SECOND_WINS;
}

function mkState(hex: string, outcome: Outcome): GameState {
  return create(GameStateSchema, {
    gameId: GameId.MINIONS,
    payload: hexToBytes(hex),
    outcome,
  });
}

/** Whose turn a position is, read off the payload — the same function the live
 *  board hands `MoveList`. */
const turnInfo: H.TurnInfo = (st) => {
  const d = decode(st.payload);
  return { turn: d.turn, side: d.side };
};

function buildHistory(r: GameRecord): H.History {
  const last = r.states.length - 1;
  // Only the final position is terminal. This matters: `isBoundary` treats any
  // non-IN_PROGRESS node as a boundary, so marking them all would collapse the
  // move list into one chip per position.
  const states = r.states.map((hx, k) =>
    mkState(hx, k === last ? terminalOutcome(r) : Outcome.IN_PROGRESS),
  );
  let h = H.newHistory(states[0]);
  for (let k = 1; k < states.length; k++) {
    // A distinct synthetic move per turn: `push` merges siblings that share
    // (kind, move), so a constant payload would fold the whole game into one
    // node the moment any two turns hung off the same parent.
    const mv = new Uint8Array([k & 0xff, (k >> 8) & 0xff]);
    h = H.push(h, mv, states[k], "turn", `turn ${decode(states[k].payload).turn}`);
  }
  return { ...h, current: h.root };
}

// ── board ────────────────────────────────────────────────────────────────────

function Position({ state }: { state: GameState }) {
  const d = decode(state.payload);
  return (
    <svg width={SVG_W} height={SVG_H} style={{ background: "#20242b", borderRadius: 8, flex: "0 0 auto" }}>
      {Array.from({ length: d.cells.length }, (_, i) => (
        <g key={i} transform={`translate(${px(i)},${py(i)})`}>
          <polygon points={HEX_POINTS} fill="#2c313a" stroke="#454b55" strokeWidth={1} />
          {d.cells[i] !== 0 && <Piece cell={d.cells[i]} pstate={d.pstate[i]} />}
        </g>
      ))}
    </svg>
  );
}

// ── view ─────────────────────────────────────────────────────────────────────

export function Recordings({ onBack }: { onBack: () => void }) {
  const [index, setIndex] = useState<IndexEntry[] | null>(null);
  const [file, setFile] = useState<string | null>(null);
  const [match, setMatch] = useState<MatchRecord | null>(null);
  const [gameIdx, setGameIdx] = useState(0);
  const [hist, setHist] = useState<H.History | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/recordings/index.json");
        if (!r.ok) throw new Error(`index.json: ${r.status}`);
        const idx = (await r.json()) as IndexEntry[];
        setIndex(idx);
        if (idx.length > 0) setFile(idx[0].file);
      } catch (e) {
        setError((e as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    if (file == null) return;
    (async () => {
      try {
        setError(null);
        const r = await fetch(`/recordings/${file}`);
        if (!r.ok) throw new Error(`${file}: ${r.status}`);
        const m = (await r.json()) as MatchRecord;
        setMatch(m);
        setGameIdx(0);
      } catch (e) {
        setMatch(null);
        setError((e as Error).message);
      }
    })();
  }, [file]);

  const game = match?.records[gameIdx] ?? null;
  useEffect(() => {
    setHist(game ? buildHistory(game) : null);
  }, [game]);

  const goto = useCallback((id: number) => {
    setHist((h) => (h ? { ...h, current: id } : h));
  }, []);

  const step = useCallback(
    (dir: -1 | 1) => {
      setHist((h) => {
        if (!h) return h;
        const id = dir < 0 ? H.prevId(h, h.current, turnInfo) : H.nextId(h, h.current, turnInfo);
        return id == null ? h : { ...h, current: id };
      });
    },
    [],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  const cur = hist ? H.node(hist, hist.current) : undefined;
  const info = useMemo(() => (cur ? decode(cur.state.payload) : null), [cur]);

  return (
    <>
      <button
        onClick={onBack}
        style={{ marginBottom: "1rem", background: "none", border: "none", color: "#1d4ed8", cursor: "pointer", font: "inherit", padding: 0 }}
      >
        ← Menu
      </button>
      <h1 style={{ marginTop: 0, marginBottom: "0.5rem" }}>Recorded matches</h1>

      {error && (
        <p style={{ color: "crimson" }}>
          {error}. Record one with <code>eval-match --record-file
          web/public/recordings/&lt;name&gt;.json</code>, then list it in{" "}
          <code>web/public/recordings/index.json</code>.
        </p>
      )}

      {index && index.length > 0 && (
        <div style={{ display: "flex", gap: "1rem", alignItems: "baseline", flexWrap: "wrap" }}>
          <label style={{ fontSize: "0.9rem", color: "#555" }}>
            Match{" "}
            <select value={file ?? ""} onChange={(e) => setFile(e.target.value)}>
              {index.map((e) => (
                <option key={e.file} value={e.file}>{e.label}</option>
              ))}
            </select>
          </label>
          {match && (
            <span style={{ fontSize: "0.9rem", color: "#555" }}>
              A <strong>{match.a_label}</strong> {match.a_wins} — {match.b_wins}{" "}
              <strong>{match.b_label}</strong> B
              {match.draws > 0 && ` (${match.draws} drawn)`} · A scored {match.a_score.toFixed(3)}
            </span>
          )}
        </div>
      )}

      {match && game && (
        <>
          <div style={{ display: "flex", gap: "1rem", alignItems: "baseline", margin: "0.75rem 0", flexWrap: "wrap" }}>
            <label style={{ fontSize: "0.9rem", color: "#555" }}>
              Game{" "}
              <select value={gameIdx} onChange={(e) => setGameIdx(Number(e.target.value))}>
                {match.records.map((r, i) => (
                  <option key={r.game} value={i}>
                    #{r.game} — {r.result === "draw" ? "drawn" : `${r.result.toUpperCase()} won`} in {r.turns} turns
                  </option>
                ))}
              </select>
            </label>
            <span style={{ fontSize: "0.9rem", color: "#555" }}>
              handicap {game.bid} · {game.a_first ? "A" : "B"} had yellow
            </span>
            <span style={{ display: "flex", gap: "0.25rem" }}>
              <button onClick={() => step(-1)}>←</button>
              <button onClick={() => step(1)}>→</button>
            </span>
          </div>

          <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start", flexWrap: "wrap" }}>
            {cur && <Position state={cur.state} />}
            <div style={{ minWidth: 260 }}>
              {info && (
                <div style={{ fontSize: "0.9rem", color: "#444", marginBottom: "0.5rem" }}>
                  <div>
                    turn {info.turn} · to move{" "}
                    <span style={{ color: SIDE_FILL[info.side], fontWeight: 700 }}>
                      {info.side === 0 ? "yellow" : "blue"}
                    </span>
                  </div>
                  <div style={{ color: "#666" }}>
                    gold {info.money[0]} / {info.money[1]}
                  </div>
                </div>
              )}
              {hist && (
                <MoveList
                  hist={hist}
                  info={turnInfo}
                  onGoto={goto}
                  onPromote={() => {}}
                  onDrop={() => {}}
                  sideFill={SIDE_FILL}
                />
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
