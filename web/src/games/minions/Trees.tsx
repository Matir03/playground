/** Viewer for self-play trees saved by `steps-selfplay --save-trees`
 * (crates/games/minions/src/tree_save.rs).
 *
 * A batch is one shared search tree with several games walking through it;
 * self-play saved the whole tree at every round, as it stood when that round's
 * moves were made. A round's tree has millions of nodes, so they are fetched
 * as rows are opened, from `minions-engine steps-tree-serve` (proxied under
 * `/treeapi`), which replays each node's state for its labels and boards.
 * Nothing is cut off: every expanded node can be opened, at any depth.
 *
 * The first runs (format 1) saved only each move's 300 most visited nodes as
 * static JSON; those are shown as they are, without the server.
 *
 * The rows are the analysis window's own `AnalysisTree`. Clicking one shows
 * the board at the end of its line: the turn so far projected (fated enemies
 * gone, decided units moved), as in the live step analysis.
 *
 * Runs live in `data/models/minions/steps/trees` (`web/public/trees` links
 * there), listed in its `index.json`; the cycle saves every generation.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { create } from "@bufbuild/protobuf";
import { GameId, GameStateSchema, Outcome, type GameState } from "../../gen/playground/game/game_pb";
import {
  AnalyzeEdgeSchema,
  AnalyzeNodeSchema,
  type AnalyzeNode,
} from "../../gen/playground/engine/engine_pb";
import { decode, SIDE_FILL } from "./Board";
import { AnalysisTree, lineKeys, type SortKey } from "./Analysis";
import { Position } from "./Recordings";

// ── the files ────────────────────────────────────────────────────────────────

interface IndexEntry {
  run: string;
  label: string;
  batches: string[];
  note?: string;
}

/** Format 2 (`tree_save.rs`): the whole tree per round, served node by node. */
interface MetaV2 {
  version: 2;
  games: number;
  rounds: { file: string; nodes: number; games: { game: number; turn: number; from: number; to: number | null }[] }[];
  distinct: number;
  ended: string;
}

/** Format 1: static per-move views. */
interface MetaV1 {
  games: number;
  moves: { games: { game: number; turn: number; to: number | null; view: string }[] }[];
  nodes: number;
  ended: string;
}

/** A format-1 move view. */
interface V1View {
  tree_nodes: number;
  nodes: { p: number; n: number; q: number; pq: number; t: boolean; b: string; e: Edge[] }[];
}

type Meta = MetaV2 | MetaV1;
const isV2 = (m: Meta): m is MetaV2 => (m as MetaV2).version === 2;

/** `[label, visits, value | null, static value, child, played, expandable,
 *  child board]`, values for the node's mover. Format 1 has only the first
 *  six, with `child` -1 past the view's cut. */
type Edge = [string, number, number | null, number, number, boolean, boolean?, string?];

interface SNode {
  /** The node's id: its index in the round's snapshot (format 2) or in the
   *  view (format 1). */
  i: number;
  n: number;
  q: number;
  pq: number;
  t: boolean;
  /** The server's replay offered exactly the saved children. */
  ok?: boolean;
  b: string;
  e: Edge[];
}

// ── the node store ───────────────────────────────────────────────────────────

/** Loaded nodes by id, plus the local indices `AnalysisTree` sees. Local
 *  indices only ever grow, so the tree's expand keys (`c<local>`) stay valid
 *  as nodes arrive; a node that is not loaded yet shows as an empty
 *  placeholder until it is opened. */
interface Store {
  nodes: Map<number, SNode>;
  order: number[];
  local: Map<number, number>;
}

const emptyStore = (): Store => ({ nodes: new Map(), order: [], local: new Map() });

function expandable(e: Edge): boolean {
  return e[6] ?? e[4] >= 0;
}

function addNodes(s: Store, add: SNode[]): Store {
  const nodes = new Map(s.nodes);
  const order = [...s.order];
  const local = new Map(s.local);
  const alloc = (id: number) => {
    if (!local.has(id)) {
      local.set(id, order.length);
      order.push(id);
    }
  };
  for (const n of add) {
    nodes.set(n.i, n);
    alloc(n.i);
  }
  for (const n of add) for (const e of n.e) if (expandable(e)) alloc(e[4]);
  return { nodes, order, local };
}

/** A format-1 view as store nodes (children's boards filled in). */
function fromView(v: V1View): SNode[] {
  return v.nodes.map((n, i) => ({
    i,
    n: n.n,
    q: n.q,
    pq: n.pq,
    t: n.t,
    b: n.b,
    e: n.e.map((e) => [e[0], e[1], e[2], e[3], e[4], e[5], e[4] >= 0, e[4] >= 0 ? v.nodes[e[4]].b : undefined] as Edge),
  }));
}

function hexToBytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.substr(i * 2, 2), 16);
  return out;
}

const board = (hex: string): GameState =>
  create(GameStateSchema, { gameId: GameId.MINIONS, payload: hexToBytes(hex), outcome: Outcome.IN_PROGRESS });

/** An edge's `turn`: its node's local index and the edge's index. There is no
 *  engine to send it to; it only has to identify the edge. */
function edgeId(node: number, edge: number): Uint8Array {
  return new Uint8Array([node & 0xff, (node >> 8) & 0xff, (node >> 16) & 0xff, (node >> 24) & 0xff, edge & 0xff, (edge >> 8) & 0xff]);
}

function parseEdgeId(t: Uint8Array): [number, number] {
  return [t[0] | (t[1] << 8) | (t[2] << 16) | (t[3] << 24), t[4] | (t[5] << 8)];
}

function toAnalysis(s: Store): AnalyzeNode[] {
  return s.order.map((id) => {
    const n = s.nodes.get(id);
    if (!n) {
      // Not loaded yet: opening it fetches it.
      return create(AnalyzeNodeSchema, { totalVisits: 0, terminal: false, termValue: 0, edges: [], kernel: [] });
    }
    const live = Math.max(n.e.length, 1);
    return create(AnalyzeNodeSchema, {
      totalVisits: n.n,
      value: n.q,
      terminal: n.t,
      termValue: n.q,
      priorValue: n.pq,
      kernel: [],
      edges: n.e.map((e, ei) =>
        create(AnalyzeEdgeSchema, {
          turn: edgeId(s.local.get(id)!, ei),
          // Every option is enumerated and statically evaluated: no policy.
          prior: 1 / live,
          energy: 0,
          visits: e[1],
          value: e[2] ?? undefined,
          child: expandable(e) ? s.local.get(e[4]) : undefined,
          explore: false,
          forced: 0,
          label: e[5] ? `${e[0]}  ◀ played` : e[0],
          draws: 1,
          priorValue: e[3],
          manual: false,
        }),
      ),
    });
  });
}

/** The played line from the root through the loaded nodes, and its labels. */
function playedLine(s: Store, root: number): { line: Uint8Array[]; labels: string[] } {
  const line: Uint8Array[] = [];
  const labels: string[] = [];
  let cur = s.nodes.get(root);
  while (cur) {
    const ei = cur.e.findIndex((e) => e[5]);
    if (ei < 0) break;
    line.push(edgeId(s.local.get(cur.i)!, ei));
    labels.push(cur.e[ei][0]);
    cur = expandable(cur.e[ei]) ? s.nodes.get(cur.e[ei][4]) : undefined;
  }
  return { line, labels };
}

// ── view ─────────────────────────────────────────────────────────────────────

const muted = { fontSize: "0.9rem", color: "#555" } as const;

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  const body = await r.json().catch(() => null);
  if (!r.ok) {
    const why = body && typeof body === "object" && "error" in body ? String(body.error) : `${r.status}`;
    throw new Error(`${url}: ${why}`);
  }
  return body as T;
}

export function Trees({ onBack }: { onBack: () => void }) {
  const [index, setIndex] = useState<IndexEntry[] | null>(null);
  const [run, setRun] = useState<string | null>(null);
  const [batch, setBatch] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [game, setGame] = useState(0);
  const [move, setMove] = useState(0);
  const [store, setStore] = useState<Store>(emptyStore);
  const [root, setRoot] = useState<number | null>(null);
  const [treeNodes, setTreeNodes] = useState(0);
  const [line, setLine] = useState<Uint8Array[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<SortKey>("visits");
  const [loading, setLoading] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const entry = index?.find((e) => e.run === run) ?? null;

  useEffect(() => {
    getJson<IndexEntry[]>("/trees/index.json")
      .then((idx) => {
        setIndex(idx);
        if (idx.length > 0) {
          setRun(idx[0].run);
          setBatch(idx[0].batches[0] ?? null);
        }
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  useEffect(() => {
    if (run == null || batch == null) return;
    setError(null);
    getJson<Meta>(`/trees/${run}/${batch}/meta.json`)
      .then((m) => {
        setMeta(m);
        setGame(0);
        setMove(0);
      })
      .catch((e) => {
        setMeta(null);
        setError((e as Error).message);
      });
  }, [run, batch]);

  /** This game's moves: the round (format 2) or move (format 1) and its turn. */
  const gameMoves = useMemo(() => {
    if (!meta) return [];
    const out: { round: number; turn: number; view?: string; stuck: boolean }[] = [];
    if (isV2(meta)) {
      meta.rounds.forEach((r, ri) => {
        const g = r.games.find((x) => x.game === game);
        if (g) out.push({ round: ri, turn: g.turn, stuck: g.to == null });
      });
    } else {
      meta.moves.forEach((m, mi) => {
        const g = m.games.find((x) => x.game === game);
        if (g) out.push({ round: mi, turn: g.turn, view: g.view, stuck: g.to == null });
      });
    }
    return out;
  }, [meta, game]);

  const current = gameMoves[Math.min(move, gameMoves.length - 1)];
  const query = current && run && batch ? `run=${run}&batch=${batch}&round=${current.round}&game=${game}` : null;

  useEffect(() => {
    if (!current || !meta || run == null || batch == null) {
      setStore(emptyStore());
      setRoot(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        setError(null);
        setLoading((x) => x + 1);
        let nodes: SNode[];
        if (isV2(meta)) {
          const r = await getJson<{ nodes: SNode[]; tree_nodes: number }>(`/treeapi/line?${query}`);
          nodes = r.nodes;
          setTreeNodes(r.tree_nodes);
        } else {
          const v = await getJson<V1View>(`/trees/${run}/${batch}/${current.view}`);
          nodes = fromView(v);
          setTreeNodes(v.tree_nodes);
        }
        if (cancelled) return;
        const s = addNodes(emptyStore(), nodes);
        setStore(s);
        setRoot(nodes[0]?.i ?? null);
        // Open the played line.
        const pl = playedLine(s, nodes[0].i).line;
        setExpanded(new Set(lineKeys(toAnalysis(s), pl)));
        setLine([]);
        setLabels([]);
      } catch (e) {
        if (!cancelled) {
          setStore(emptyStore());
          setRoot(null);
          setError((e as Error).message);
        }
      } finally {
        setLoading((x) => x - 1);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [query, meta]); // eslint-disable-line react-hooks/exhaustive-deps

  const nodes = useMemo(() => toAnalysis(store), [store]);

  /** Open (or close) a row; opening a node not loaded yet fetches it. */
  const toggle = useCallback(
    (key: string) => {
      const opening = !expanded.has(key);
      setExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key);
        else next.add(key);
        return next;
      });
      if (!opening || !query || !meta || !isV2(meta) || !key.startsWith("c")) return;
      const id = store.order[Number(key.slice(1))];
      if (id === undefined || store.nodes.has(id)) return;
      setLoading((x) => x + 1);
      getJson<{ nodes: SNode[] }>(`/treeapi/nodes?${query}&idx=${id}`)
        .then((r) => setStore((s) => addNodes(s, r.nodes)))
        .catch((e) => setError((e as Error).message))
        .finally(() => setLoading((x) => x - 1));
    },
    [expanded, query, meta, store],
  );

  // The board at the end of the shown line: the last edge's own board.
  const shown = useMemo(() => {
    if (root == null) return null;
    if (line.length === 0) return board(store.nodes.get(root)!.b);
    const [li, ei] = parseEdgeId(line[line.length - 1]);
    const n = store.nodes.get(store.order[li]);
    const e = n?.e[ei];
    return e ? board(e[7] ?? n!.b) : null;
  }, [store, root, line]);
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

  const rootNode = root != null ? store.nodes.get(root) : undefined;
  const played = root != null ? playedLine(store, root) : { line: [], labels: [] };
  const chosen = rootNode ? rootNode.e.findIndex((e) => e[5]) : -1;
  const drift = [...store.nodes.values()].filter((n) => n.ok === false).length;
  const v2 = meta != null && isV2(meta);

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
          {error}
          {error.includes("/treeapi/") && (
            <>
              {" "}
              — start the tree server: <code>minions-engine steps-tree-serve --dir data/models/minions/steps/trees</code>
            </>
          )}
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
              {meta.games} games in one tree ·{" "}
              {isV2(meta)
                ? `${meta.rounds.filter((r) => r.games.length > 0).length} rounds · ${meta.distinct.toLocaleString()} distinct nodes`
                : `${meta.moves.length} moves · ${meta.nodes.toLocaleString()} nodes · top 300 per move only (old format)`}
              {meta.ended ? ` · ended: ${meta.ended}` : " · still running"}
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
              {current && <> · turn {current.turn}</>}
            </span>
            <button onClick={() => setMove((m) => Math.min(gameMoves.length - 1, m + 1))}>→</button>
          </span>
          {current?.stuck && <span style={{ color: "crimson" }}>walk got stuck</span>}
          {loading > 0 && <span style={muted}>loading…</span>}
        </div>
      )}

      {rootNode && (
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
                {rootNode.n.toLocaleString()} descents · search {rootNode.q.toFixed(3)} · net {rootNode.pq.toFixed(3)} ·{" "}
                {treeNodes.toLocaleString()} nodes in the shared tree ·{" "}
                {v2 ? `${store.nodes.size.toLocaleString()} loaded (open any row for more)` : `${store.nodes.size} saved`}
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
            {drift > 0 && (
              <p style={{ color: "crimson", fontSize: "0.85rem", margin: "0 0 0.35rem" }}>
                {drift} loaded node(s) replayed with different options than the saved tree: their labels and boards may be wrong.
              </p>
            )}
            <div style={{ background: "#20242b", borderRadius: 8, padding: "0.35rem 0.5rem", color: "#ccc", maxHeight: "75vh", overflow: "auto" }}>
              <AnalysisTree
                nodes={nodes}
                chosen={chosen >= 0 ? chosen : undefined}
                sort={sort}
                expanded={expanded}
                onToggle={toggle}
                onPlay={(l, ls) => {
                  setLine(l);
                  setLabels(ls);
                }}
                maxDepth={1024}
                shown={line}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
