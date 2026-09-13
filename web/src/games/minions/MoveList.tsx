/** The game record, rendered as a clickable tree of whole turns.
 *
 * Grouped by turn rather than by ply. A Minions turn is a sequence of actions —
 * several moves, some attacks, a buy, a spawn, an end — so a flat ply list is
 * unreadable and hides the only boundary that matters: whose turn it was and
 * what the whole turn amounted to.
 *
 * The turn is also the unit you select. Its actions are listed inside one chip
 * that goes to the position after the turn; the positions between its actions
 * are not offered, because nothing — search, nets, analyser — is ever asked
 * about a half-played turn. See `isBoundary` in history.ts.
 *
 * Variations hang under the turn they branch from, indented with a rule. The
 * main line is always the first child, so promoting a variation is what makes
 * it "what happened" — nothing else in the record distinguishes them.
 */

import type { GameState } from "../../gen/playground/game/game_pb";
import { type HNode, type History, isBoundary, moveLabel, node } from "./history";

export interface MoveListProps {
  hist: History;
  /** Turn number and side to move for a position; the caller owns decoding. */
  info: (s: GameState) => { turn: number; side: number };
  onGoto: (id: number) => void;
  onPromote: (id: number) => void;
  onDrop: (id: number) => void;
  sideFill: string[];
}

const CHIP: React.CSSProperties = {
  font: "inherit",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  fontSize: "0.78rem",
  padding: "1px 5px",
  border: "1px solid transparent",
  borderRadius: 4,
  background: "none",
  cursor: "pointer",
  color: "#333",
  lineHeight: 1.5,
  textAlign: "left",
};

/** The actions of one turn, plus any branches off them. */
interface Block {
  turn: number;
  side: number;
  nodes: HNode[];
  /** Starts partway through a turn: a variation that shares the turn's first
   *  actions with the line it branched from. */
  partial: boolean;
}

/** Walk the line from `startId` (exclusive) and group it into turns. The first
 *  step follows `first` when given, which is how a variation is rendered:
 *  from its branch point, down its own child rather than the main one. */
function blocksFrom(hist: History, startId: number, info: MoveListProps["info"], first?: number): Block[] {
  const out: Block[] = [];
  let cur = node(hist, startId);
  let next = first ?? cur?.children[0];
  while (cur && next !== undefined) {
    const child = node(hist, next);
    if (!child) break;
    const last = out[out.length - 1];
    // A new turn starts exactly where the position it is played from is a
    // boundary, so the blocks and the selectable positions cannot disagree.
    if (last && !isBoundary(hist, cur.id, info)) last.nodes.push(child);
    else {
      // Labelled by the position the turn was played *from*: it belongs to
      // the side that was to move before it, not the side to move after.
      const { turn, side } = info(cur.state);
      out.push({ turn, side, nodes: [child], partial: !isBoundary(hist, cur.id, info) });
    }
    cur = child;
    next = cur.children[0];
  }
  return out;
}

/** A turn as one line of text. Phase ends are left out: which phase an action
 *  belongs to already shows in the action itself. An unfinished turn with no
 *  actions yet is not a pass, so it gets no body at all. */
function turnLabel(b: Block, done: boolean): string {
  const parts = b.nodes
    .filter((n) => !(n.kind === "move" && n.move?.[0] === 4))
    .map((n) => moveLabel(n.move, n));
  const body = parts.length ? parts.join("  ") : done ? "pass" : "";
  const text = b.partial ? `… ${body}` : body;
  return done ? text : `${text} …`.trim();
}

function Line({
  startId,
  first,
  depth,
  p,
}: {
  startId: number;
  first?: number;
  depth: number;
  p: MoveListProps;
}) {
  const blocks = blocksFrom(p.hist, startId, p.info, first);
  if (!blocks.length) return null;
  const accent = depth > 0 ? "#7a4ea8" : "#2c6fb8";

  return (
    <>
      {blocks.map((b) => {
        const end = b.nodes[b.nodes.length - 1];
        const done = isBoundary(p.hist, end.id, p.info);
        const here = b.nodes.some((n) => n.id === p.hist.current);
        return (
          <div
            key={`${b.turn}:${b.side}:${b.nodes[0].id}`}
            style={{ display: "flex", gap: 6, alignItems: "baseline", padding: "1px 0" }}
          >
            <span
              style={{
                flex: "0 0 auto",
                minWidth: "2.6rem",
                fontFamily: "ui-monospace, monospace",
                fontSize: "0.72rem",
                color: "#999",
                textAlign: "right",
              }}
            >
              <span style={{ color: p.sideFill[b.side], fontWeight: 700 }}>●</span> {b.turn}
            </span>
            <button
              onClick={() => p.onGoto(end.id)}
              style={{
                ...CHIP,
                background: here ? accent : "none",
                color: here ? "#fff" : depth > 0 ? "#7a4ea8" : "#333",
                fontWeight: here ? 600 : 400,
                fontStyle: done ? "normal" : "italic",
              }}
              title={
                done
                  ? "go to the position after this turn"
                  : "this turn is still being played — the position the game is in"
              }
            >
              {turnLabel(b, done)}
            </button>
          </div>
        );
      })}
      {/* Branches, emitted after the line they leave so the main line reads
          uninterrupted. Each is rendered from its branch point, down its own
          child. */}
      {blocks.flatMap((b) =>
        b.nodes.flatMap((n) => {
          const parent = node(p.hist, n.parent!);
          if (!parent || parent.children.length < 2 || parent.children[0] !== n.id) return [];
          return parent.children.slice(1).map((vid) => (
            <div
              key={vid}
              style={{
                marginLeft: 12 + depth * 8,
                paddingLeft: 8,
                borderLeft: "2px solid #d9cdea",
                margin: "2px 0 4px",
              }}
            >
              <Line startId={parent.id} first={vid} depth={depth + 1} p={p} />
              <div style={{ display: "flex", gap: 4, paddingLeft: "2.9rem" }}>
                <button
                  onClick={() => p.onPromote(vid)}
                  style={{ ...CHIP, color: "#666", fontSize: "0.68rem" }}
                  title="make this the main line"
                >
                  ↑main
                </button>
                <button
                  onClick={() => p.onDrop(vid)}
                  style={{ ...CHIP, color: "#a33", fontSize: "0.68rem" }}
                  title="delete this variation"
                >
                  ✕
                </button>
              </div>
            </div>
          ));
        }),
      )}
    </>
  );
}

export function MoveList(props: MoveListProps) {
  const { hist } = props;
  const empty = (node(hist, hist.root)?.children.length ?? 0) === 0;
  return (
    <div
      style={{
        border: "1px solid #ddd",
        borderRadius: 8,
        padding: "0.5rem 0.6rem",
        maxHeight: 300,
        overflowY: "auto",
        overflowX: "auto",
        background: "#fff",
      }}
    >
      <button
        onClick={() => props.onGoto(hist.root)}
        style={{
          ...CHIP,
          color: hist.current === hist.root ? "#fff" : "#666",
          background: hist.current === hist.root ? "#2c6fb8" : "none",
          marginBottom: 2,
        }}
      >
        start
      </button>
      {empty ? (
        <div style={{ color: "#999", fontSize: "0.8rem", padding: "2px 4px" }}>
          no moves yet
        </div>
      ) : (
        <Line startId={hist.root} depth={0} p={props} />
      )}
    </div>
  );
}
