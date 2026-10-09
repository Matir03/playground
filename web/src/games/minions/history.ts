/** Game record for the Minions board: a move tree, not a move list.
 *
 * The board needs four things the engine does not provide, because the engine
 * holds exactly one position per handle and has no rewind:
 *
 *   - every move that was played, in order;
 *   - the ability to look at any earlier position;
 *   - variations, so a line can be tried from an earlier position without
 *     destroying what actually happened;
 *   - a position to hand the analyser, from anywhere in the tree.
 *
 * So the record keeps the `GameState` the server returned for every move. That
 * makes browsing free — no server round trip to look at an old position, which
 * matters because a Minions turn is a sequence of actions and a game runs to
 * a few hundred of them. The server only has to be involved when a position is
 * *used*: to play from it, or to analyse it. `pathMoves` produces the replay
 * for that, and `Board.tsx` caches which node the live handle is sitting on.
 *
 * Nodes are addressed by a numeric id and held in a flat map rather than by
 * object reference, so React state updates are a shallow copy of the map and
 * a node can be named in a URL, a key, or a ref without carrying the subtree.
 */

import { type GameState, Outcome } from "../../gen/playground/game/game_pb";
import { UNITS } from "./units";

export interface HNode {
  id: number;
  parent: number | null;
  /** The move that produced this node. Null only at the root. */
  move: Uint8Array | null;
  /** Which RPC replays `move`. Minions has both: the board plays one action at
   *  a time through ApplyMove, while the analyser commits a whole TurnSpec
   *  through ApplyTurn. The two encodings are unrelated, so a replay that
   *  guessed would fail on any line containing a committed turn. */
  kind: "move" | "turn";
  /** Display text for a `turn` node, whose payload is opaque here. */
  label?: string;
  /** The position *after* `move` — exactly what the server returned. */
  state: GameState;
  /** Children in play order. Index 0 is the main continuation; the rest are
   *  variations, in the order they were first tried. */
  children: number[];
}

export interface History {
  nodes: Map<number, HNode>;
  root: number;
  /** The node being shown. Not necessarily a leaf, and not necessarily on the
   *  main line. */
  current: number;
  /** Monotonic id source. Never reused, so a stale id is always detectably
   *  absent rather than silently pointing at a different node. */
  nextId: number;
}

export function newHistory(state: GameState): History {
  const root: HNode = { id: 0, parent: null, move: null, kind: "move", state, children: [] };
  return { nodes: new Map([[0, root]]), root: 0, current: 0, nextId: 1 };
}

export function node(h: History, id: number): HNode | undefined {
  return h.nodes.get(id);
}

function sameMove(a: Uint8Array | null, b: Uint8Array | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Record `move` played from `h.current`, and make the result current.
 *
 * Replaying a move that already exists as a child follows it instead of
 * duplicating it — stepping back and playing the same move again should walk
 * the line you already have, not fork a variation identical to it. Any other
 * move from a node that already has children becomes a new variation.
 */
export function push(
  h: History,
  move: Uint8Array,
  state: GameState,
  kind: "move" | "turn" = "move",
  label?: string,
): History {
  const parent = h.nodes.get(h.current);
  if (!parent) return h;

  for (const cid of parent.children) {
    const c = h.nodes.get(cid);
    if (c && c.kind === kind && sameMove(c.move, move)) {
      // Refresh the stored state: identical moves must reach identical
      // positions, but the server is the authority and a mismatch here would
      // be a bug worth showing rather than hiding behind the cached copy.
      const nodes = new Map(h.nodes);
      nodes.set(cid, { ...c, state });
      return { ...h, nodes, current: cid };
    }
  }

  const id = h.nextId;
  const nodes = new Map(h.nodes);
  nodes.set(id, { id, parent: parent.id, move, kind, label, state, children: [] });
  nodes.set(parent.id, { ...parent, children: [...parent.children, id] });
  return { ...h, nodes, current: id, nextId: id + 1 };
}

/** Root → `id`, inclusive of both ends. Empty if `id` is unknown. */
export function path(h: History, id: number): HNode[] {
  const out: HNode[] = [];
  let cur = h.nodes.get(id);
  while (cur) {
    out.push(cur);
    cur = cur.parent === null ? undefined : h.nodes.get(cur.parent);
  }
  return out.reverse();
}

/** The steps to replay from a fresh game to reach `id`, each tagged with the
 *  RPC that applies it. */
export function pathMoves(h: History, id: number): { move: Uint8Array; kind: "move" | "turn" }[] {
  return path(h, id)
    .filter((n) => n.move !== null)
    .map((n) => ({ move: n.move as Uint8Array, kind: n.kind }));
}

/** True when `anc` is `id` or one of its ancestors. */
export function isAncestor(h: History, anc: number, id: number): boolean {
  let cur: number | null = id;
  while (cur !== null) {
    if (cur === anc) return true;
    cur = h.nodes.get(cur)?.parent ?? null;
  }
  return false;
}

/** The main line: root, then first child all the way down. */
export function mainLine(h: History): HNode[] {
  const out: HNode[] = [];
  let cur = h.nodes.get(h.root);
  while (cur) {
    out.push(cur);
    cur = cur.children.length ? h.nodes.get(cur.children[0]) : undefined;
  }
  return out;
}

/** The deepest node reachable from `id` by first children. */
export function tipOf(h: History, id: number): number {
  let cur = h.nodes.get(id);
  while (cur && cur.children.length) {
    const next = h.nodes.get(cur.children[0]);
    if (!next) break;
    cur = next;
  }
  return cur ? cur.id : id;
}

// ── turn boundaries ─────────────────────────────────────────────────────────
//
// The unit of the record is a whole turn. A position partway through a turn is
// not a place to stop, look, or analyse from: the search, the nets and the
// analyser all work in whole TurnSpecs, and a half-played turn is a position
// none of them is ever asked about. The one exception is the tip of a line
// whose last turn is still being played, because that is where the game is.

/** Whose turn a position is in; the caller owns decoding. */
export type TurnInfo = (s: GameState) => { turn: number; side: number };

/** True for the root, a finished game, and the first position of each turn. */
export function isBoundary(h: History, id: number, info: TurnInfo): boolean {
  const n = h.nodes.get(id);
  if (!n) return false;
  if (n.parent === null || n.kind === "turn") return true;
  if (n.state.outcome !== Outcome.IN_PROGRESS) return true;
  const p = h.nodes.get(n.parent);
  if (!p) return true;
  const a = info(p.state);
  const b = info(n.state);
  return a.turn !== b.turn || a.side !== b.side;
}

/** A position the record lets you stop on: a boundary, or a line's tip. */
export function isStop(h: History, id: number, info: TurnInfo): boolean {
  const n = h.nodes.get(id);
  return !!n && (n.children.length === 0 || isBoundary(h, id, info));
}

/** The nearest boundary at or above `id`: the start of the turn it sits in. */
export function turnStart(h: History, id: number, info: TurnInfo): number {
  let cur: number | null = id;
  while (cur !== null) {
    if (isBoundary(h, cur, info)) return cur;
    cur = h.nodes.get(cur)?.parent ?? null;
  }
  return h.root;
}

/** One turn back: the start of the turn before the one `id` ends, or of the
 *  turn in progress when `id` is a tip partway through it. */
export function prevId(h: History, id: number, info: TurnInfo): number | null {
  const p = h.nodes.get(id)?.parent ?? null;
  return p === null ? null : turnStart(h, p, info);
}

/** One turn forward along the main continuation. A line that ends partway
 *  through a turn stops at its tip. */
export function nextId(h: History, id: number, info: TurnInfo): number | null {
  let cur = h.nodes.get(id);
  while (cur && cur.children.length) {
    const next = h.nodes.get(cur.children[0]);
    if (!next) break;
    if (isBoundary(h, next.id, info)) return next.id;
    cur = next;
  }
  return cur && cur.id !== id ? cur.id : null;
}

/** Make the variation containing `id` the main line at every branch above it.
 *
 * Promotion is per-branch rather than whole-tree: only the choices on the path
 * to `id` change, so promoting a deep variation does not silently reorder
 * branches it never passed through.
 */
export function promote(h: History, id: number): History {
  const nodes = new Map(h.nodes);
  let cur = nodes.get(id);
  while (cur && cur.parent !== null) {
    const p = nodes.get(cur.parent);
    if (!p) break;
    if (p.children[0] !== cur.id) {
      nodes.set(p.id, {
        ...p,
        children: [cur.id, ...p.children.filter((c) => c !== cur!.id)],
      });
    }
    cur = nodes.get(p.id);
  }
  return { ...h, nodes };
}

/** Delete `id` and everything under it. The root cannot be deleted.
 *
 * If the current node was inside the deleted subtree, it moves to the parent —
 * leaving `current` pointing at a node that no longer exists would render an
 * empty board with no way back.
 */
export function drop(h: History, id: number): History {
  const target = h.nodes.get(id);
  if (!target || target.parent === null) return h;

  const nodes = new Map(h.nodes);
  const stack = [id];
  while (stack.length) {
    const cur = nodes.get(stack.pop()!);
    if (!cur) continue;
    stack.push(...cur.children);
    nodes.delete(cur.id);
  }
  const p = nodes.get(target.parent);
  if (p) nodes.set(p.id, { ...p, children: p.children.filter((c) => c !== id) });

  const current = h.nodes.has(h.current) && isAncestor(h, id, h.current) ? target.parent : h.current;
  return { ...h, nodes, current: nodes.has(current) ? current : h.root };
}

// ── move rendering ──────────────────────────────────────────────────────────

const FILES = "abcdefg";

/** Cell index to a board coordinate, e.g. 0 -> "a1". Row-major over the 7x7. */
export function cellName(i: number): string {
  if (i < 0 || i >= 49) return "??";
  return `${FILES[i % 7]}${Math.floor(i / 7) + 1}`;
}

/** A unit's one-letter symbol (Z, S, P, W, G, N), from the shared table
 *  rather than a copy of it: the first copy of this list invented a unit and
 *  shifted every index above zombie by one, which the record then displayed
 *  as fact. */
function unitCh(u: number): string {
  return UNITS[u]?.ch ?? `?${u}`;
}

/** A move as it should read in the record. Mirrors `MinionsMove` in game.rs;
 *  an unknown encoding renders as its bytes rather than throwing, so a record
 *  written by a newer engine still displays. */
export function moveLabel(m: Uint8Array | null, n?: HNode): string {
  if (n?.kind === "turn") return n.label && n.label.length ? n.label : "⟨turn⟩";
  if (!m || m.length === 0) return "start";
  switch (m[0]) {
    case 0:
      return `${cellName(m[1])}–${cellName(m[2])}`;
    case 1:
      return `${cellName(m[1])}×${cellName(m[2])}`;
    case 2:
      return `+${unitCh(m[1])}`;
    case 3:
      return `${unitCh(m[1])}@${cellName(m[2])}`;
    case 4:
      return "end";
    case 5:
      // A cyclic move: each unit to the next one's cell, in cycle notation.
      return `(${Array.from(m.subarray(1)).map(cellName).join(" ")})`;
    default:
      return `[${Array.from(m).join(" ")}]`;
  }
}
