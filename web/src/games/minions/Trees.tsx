/** Viewer for self-play trees saved by `steps-selfplay --save-trees`
 * (crates/games/minions/src/tree_save.rs).
 *
 * Engine-free, like `Recordings`: the files are static JSON under
 * `/trees/<run>/<batch>/`. A batch is one shared search tree with several
 * games walking through it. For each move and game, self-play wrote the
 * subtree under the game's position *as it stood when the move was made* —
 * its most visited expanded nodes and every live edge of each — which this
 * page renders with the analysis window's own `AnalysisTree`. Clicking a line
 * shows the board at the end of it: the turn so far projected (fated enemies
 * gone, decided units moved), as in the live step analysis.
 *
 * The full tree (`nodes.bin`, every node the batch created) is far too big
 * for a browser; read it with `python/minions_steps_trees.py`.
 *
 * Publish a run with: `steps-selfplay ... --games 16 --save-trees
 * data/models/minions/steps/trees/<run>` and list it in
 * `data/models/minions/steps/trees/index.json` (`web/public/trees` links
 * there).
 */

import { useEffect, useMemo, useState } from "react";
import { create } from "@bufbuild/protobuf";
import { GameId, GameStateSchema, Outcome, type GameState } from "../../gen/playground/game/game_pb";
import {
  AnalyzeEdgeSchema,
  AnalyzeNodeSchema,
  type AnalyzeNode,
} from "../../gen/playground/engine/engine_pb";
import { decode, SIDE_FILL } from "./Board";
import { AnalysisTree, lineKeys, nodeAt, type SortKey } from "./Analysis";
import { Position } from "./Recordings";

// ── the files (tree_save.rs) ─────────────────────────────────────────────────

interface IndexEntry {
  run: string;
  label: string;
  batches: string[];
  note?: string;
}

interface GameMove {
  game: number;
  turn: number;
  from: number;
  to: number | null;
  view: string;
}

interface Meta {
  batch: number;
  games: number;
  start: string;
  salt: number;
  moves: { games: GameMove[]; nodes_written: number }[];
  nodes: number;
  ended: string;
}

/** `[label, visits, value | null, prior value, child | -1, played]`, values in
 *  [-1, 1] for the parent's mover. */
type ViewEdge = [string, number, number | null, number, number, boolean];

interface ViewNode {
  p: number;
  n: number;
  q: number;
  pq: number;
  t: boolean;
  b: string;
  e: ViewEdge[];
}

interface View {
  batch: number;
  game: number;
  move: number;
  turn: number;
  value: number;
  tree_nodes: number;
  nodes: ViewNode[];
  chosen: number | null;
}

// ── conversion ───────────────────────────────────────────────────────────────

function hexToBytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.substr(i * 2, 2), 16);
  return out;
}

const board = (hex: string): GameState =>
  create(GameStateSchema, { gameId: GameId.MINIONS, payload: hexToBytes(hex), outcome: Outcome.IN_PROGRESS });

/** An edge's `turn`: its node and edge index, which is all a line needs to
 *  identify it here (there is no engine to send it to). */
function edgeId(node: number, edge: number): Uint8Array {
  return new Uint8Array([node & 0xff, (node >> 8) & 0xff, (node >> 16) & 0xff, edge & 0xff, (edge >> 8) & 0xff]);
}

function toAnalysis(v: View): AnalyzeNode[] {
  return v.nodes.map((n, ni) => {
    const live = Math.max(n.e.length, 1);
    return create(AnalyzeNodeSchema, {
      parent: n.p >= 0 ? n.p : undefined,
      totalVisits: n.n,
      value: n.q,
      terminal: n.t,
      termValue: n.q,
      priorValue: n.pq,
      kernel: [],
      edges: n.e.map(([label, visits, value, pq, child, played], ei) =>
        create(AnalyzeEdgeSchema, {
          turn: edgeId(ni, ei),
          // Every option is enumerated and statically evaluated: no policy.
          prior: 1 / live,
          energy: 0,
          visits,
          value: value ?? undefined,
          child: child >= 0 ? child : undefined,
          explore: false,
          forced: 0,
          label: played ? `${label}  ◀ played` : label,
          draws: 1,
          priorValue: pq,
          manual: false,
        }),
      ),
    });
  });
}

/** The played line, as deep as the view has it, and its labels. */
function playedLine(v: View): { line: Uint8Array[]; labels: string[] } {
  const line: Uint8Array[] = [];
  const labels: string[] = [];
  let cur = 0;
  for (;;) {
    const ei = v.nodes[cur].e.findIndex((e) => e[5]);
    if (ei < 0) break;
    line.push(edgeId(cur, ei));
    labels.push(v.nodes[cur].e[ei][0]);
    const child = v.nodes[cur].e[ei][4];
    if (child < 0) break;
    cur = child;
  }
  return { line, labels };
}

// ── view ─────────────────────────────────────────────────────────────────────

const muted = { fontSize: "0.9rem", color: "#555" } as const;

export function Trees({ onBack }: { onBack: () => void }) {
  const [index, setIndex] = useState<IndexEntry[] | null>(null);
  const [run, setRun] = useState<string | null>(null);
  const [batch, setBatch] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [game, setGame] = useState(0);
  const [move, setMove] = useState(0);
  const [view, setView] = useState<View | null>(null);
  const [line, setLine] = useState<Uint8Array[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<SortKey>("visits");
  const [error, setError] = useState<string | null>(null);

  const entry = index?.find((e) => e.run === run) ?? null;

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/trees/index.json");
        if (!r.ok) throw new Error(`trees/index.json: ${r.status}`);
        const idx = (await r.json()) as IndexEntry[];
        setIndex(idx);
        if (idx.length > 0) {
          setRun(idx[0].run);
          setBatch(idx[0].batches[0] ?? null);
        }
      } catch (e) {
        setError((e as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    if (run == null || batch == null) return;
    (async () => {
      try {
        setError(null);
        const r = await fetch(`/trees/${run}/${batch}/meta.json`);
        if (!r.ok) throw new Error(`${run}/${batch}/meta.json: ${r.status}`);
        setMeta((await r.json()) as Meta);
        setGame(0);
        setMove(0);
      } catch (e) {
        setMeta(null);
        setError((e as Error).message);
      }
    })();
  }, [run, batch]);

  /** This game's moves: index into `meta.moves`, and its entry there. */
  const gameMoves = useMemo(() => {
    if (!meta) return [];
    const out: { mi: number; gm: GameMove }[] = [];
    meta.moves.forEach((m, mi) => {
      const gm = m.games.find((g) => g.game === game);
      if (gm) out.push({ mi, gm });
    });
    return out;
  }, [meta, game]);

  const current = gameMoves[Math.min(move, gameMoves.length - 1)];

  useEffect(() => {
    if (!current || run == null || batch == null) {
      setView(null);
      return;
    }
    (async () => {
      try {
        const r = await fetch(`/trees/${run}/${batch}/${current.gm.view}`);
        if (!r.ok) throw new Error(`${current.gm.view}: ${r.status}`);
        const v = (await r.json()) as View;
        setView(v);
        // Open the played line, and show where it ends.
        const pl = playedLine(v).line;
        const nodes = toAnalysis(v);
        setExpanded(new Set(lineKeys(nodes, pl)));
        setLine([]);
        setLabels([]);
      } catch (e) {
        setView(null);
        setError((e as Error).message);
      }
    })();
  }, [current?.gm.view, run, batch]); // eslint-disable-line react-hooks/exhaustive-deps

  const nodes = useMemo(() => (view ? toAnalysis(view) : []), [view]);

  // The board at the end of the shown line: the node it reaches, or, past the
  // view's cut, the deepest node on it (edges below it carry no board).
  const shownNode = useMemo(() => {
    if (!view) return undefined;
    for (let k = line.length; k >= 0; k--) {
      const n = nodeAt(nodes, line.slice(0, k));
      if (n !== undefined) return { n, exact: k === line.length };
    }
    return undefined;
  }, [view, nodes, line]);

  const shown = view && shownNode ? board(view.nodes[shownNode.n].b) : null;
  const info = shown ? decode(shown.payload) : null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); setMove((m) => Math.max(0, m - 1)); }
      if (e.key === "ArrowRight") { e.preventDefault(); setMove((m) => Math.min(gameMoves.length - 1, m + 1)); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [gameMoves.length]);

  const played = view ? playedLine(view) : { line: [], labels: [] };

  return (
    <>
      <button
        onClick={onBack}
        style={{ marginBottom: "1rem", background: "none", border: "none", color: "#1d4ed8", cursor: "pointer", font: "inherit", padding: 0 }}
      >
        ← Menu
      </button>
      <h1 style={{ marginTop: 0, marginBottom: "0.5rem" }}>Self-play trees</h1>

      {error && (
        <p style={{ color: "crimson" }}>
          {error}. Save a run with <code>steps-selfplay --games 16 --save-trees
          data/models/minions/steps/trees/&lt;run&gt;</code> and list it in{" "}
          <code>data/models/minions/steps/trees/index.json</code>.
        </p>
      )}

      {index && index.length > 0 && (
        <div style={{ display: "flex", gap: "1rem", alignItems: "baseline", flexWrap: "wrap" }}>
          <label style={muted}>
            Run{" "}
            <select
              value={run ?? ""}
              onChange={(e) => {
                const ent = index.find((x) => x.run === e.target.value);
                setRun(e.target.value);
                setBatch(ent?.batches[0] ?? null);
              }}
            >
              {index.map((e) => (
                <option key={e.run} value={e.run}>{e.label}</option>
              ))}
            </select>
          </label>
          {entry && entry.batches.length > 1 && (
            <label style={muted}>
              Tree{" "}
              <select value={batch ?? ""} onChange={(e) => setBatch(e.target.value)}>
                {entry.batches.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </label>
          )}
          {meta && (
            <span style={muted}>
              {meta.games} games in one tree · {meta.moves.length} moves ·{" "}
              {meta.nodes.toLocaleString()} nodes in all · ended: {meta.ended}
            </span>
          )}
        </div>
      )}
      {entry?.note && <p style={{ ...muted, marginTop: "0.35rem" }}>{entry.note}</p>}

      {meta && (
        <div style={{ display: "flex", gap: "1rem", alignItems: "baseline", margin: "0.75rem 0", flexWrap: "wrap" }}>
          <label style={muted}>
            Game{" "}
            <select
              value={game}
              onChange={(e) => {
                setGame(Number(e.target.value));
                setMove(0);
              }}
            >
              {Array.from({ length: meta.games }, (_, g) => (
                <option key={g} value={g}>#{g}</option>
              ))}
            </select>
          </label>
          <span style={{ display: "flex", gap: "0.25rem", alignItems: "baseline" }}>
            <button onClick={() => setMove((m) => Math.max(0, m - 1))}>←</button>
            <span style={muted}>
              move {Math.min(move, gameMoves.length - 1) + 1} / {gameMoves.length}
              {current && <> · turn {current.gm.turn}</>}
            </span>
            <button onClick={() => setMove((m) => Math.min(gameMoves.length - 1, m + 1))}>→</button>
          </span>
          {current && current.gm.to == null && <span style={{ color: "crimson" }}>walk got stuck</span>}
        </div>
      )}

      {view && (
        <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ flex: "0 0 auto" }}>
            {shown && <Position state={shown} />}
            {info && (
              <div style={{ fontSize: "0.9rem", color: "#444", marginTop: "0.5rem" }}>
                turn {info.turn} · to move{" "}
                <span style={{ color: SIDE_FILL[info.side], fontWeight: 700 }}>
                  {info.side === 0 ? "yellow" : "blue"}
                </span>{" "}
                · gold {info.money[0]} / {info.money[1]}
              </div>
            )}
            <div style={{ ...muted, marginTop: "0.35rem", maxWidth: 420 }}>
              {labels.length === 0 ? "The position the move was searched from." : labels.join(" → ")}
              {shownNode && !shownNode.exact && " (past the saved view: showing its deepest saved node)"}
            </div>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
              <button onClick={() => { setLine([]); setLabels([]); }}>root</button>
              <button
                onClick={() => {
                  setLine(played.line);
                  setLabels(played.labels);
                  setExpanded((prev) => new Set([...prev, ...lineKeys(nodes, played.line)]));
                }}
              >
                played line
              </button>
            </div>
          </div>
          <div style={{ flex: "1 1 520px", minWidth: 0 }}>
            <div style={{ display: "flex", gap: "1rem", alignItems: "baseline", flexWrap: "wrap", marginBottom: "0.35rem" }}>
              <span style={muted}>
                {view.nodes[0].n.toLocaleString()} descents · search {view.value.toFixed(3)} · net{" "}
                {view.nodes[0].pq.toFixed(3)} · {view.nodes.length} of the subtree's expanded nodes shown ·{" "}
                {view.tree_nodes.toLocaleString()} nodes in the shared tree
              </span>
              <label style={muted}>
                sort{" "}
                <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
                  <option value="visits">visits</option>
                  <option value="value">value</option>
                  <option value="prior">net</option>
                </select>
              </label>
            </div>
            <div style={{ background: "#20242b", borderRadius: 8, padding: "0.35rem 0.5rem", color: "#ccc", maxHeight: "75vh", overflow: "auto" }}>
              <AnalysisTree
                nodes={nodes}
                chosen={view.chosen ?? undefined}
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
                onPlay={(l, ls) => {
                  setLine(l);
                  setLabels(ls);
                }}
                maxDepth={256}
                shown={line}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

