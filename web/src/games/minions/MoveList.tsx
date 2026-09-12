/** The game record, rendered as a clickable tree.
 *
 * Grouped by turn rather than by ply. A Minions turn is a sequence of actions —
 * several moves, some attacks, a buy, a spawn, an end — so a flat ply list is
 * unreadable and hides the only boundary that matters: whose turn it was and
 * what the whole turn amounted to. Actions run inline inside a turn block.
 *
 * Variations hang under the action they branch from, indented with a rule. The
 * main line is always the first child, so promoting a variation is what makes
 * it "what happened" — nothing else in the record distinguishes them.
 */

import type { GameState } from "../../gen/playground/game/game_pb";
import { type HNode, type History, moveLabel, node } from "./history";

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
};

/** One run of actions belonging to a single turn, plus any branches off them. */
interface Block {
  turn: number;
  side: number;
  nodes: HNode[];
}

/** Walk first children from `startId` (exclusive) and group into turn blocks. */
function blocksFrom(hist: History, startId: number, info: MoveListProps["info"]): Block[] {
  const out: Block[] = [];
  let cur = node(hist, startId);
  while (cur && cur.children.length) {
    const child = node(hist, cur.children[0]);
    if (!child) break;
    // A node is labelled by the position it *came from*: the action belongs to
    // the side that was to move before it, not the side to move after.
    const { turn, side } = info(cur.state);
    const last = out[out.length - 1];
    if (last && last.turn === turn && last.side === side) last.nodes.push(child);
    else out.push({ turn, side, nodes: [child] });
    cur = child;
  }
  return out;
}

function Line({
  startId,
  depth,
  p,
}: {
  startId: number;
  depth: number;
  p: MoveListProps;
}) {
  const blocks = blocksFrom(p.hist, startId, p.info);
  if (!blocks.length) return null;

  return (
    <>
      {blocks.map((b) => (
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
          <span style={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "baseline" }}>
            {b.nodes.map((n) => {
              const cur = n.id === p.hist.current;
              const branches = (node(p.hist, n.parent!)?.children.length ?? 1) > 1;
              return (
                <span key={n.id} style={{ display: "contents" }}>
                  <button
                    onClick={() => p.onGoto(n.id)}
                    style={{
                      ...CHIP,
                      background: cur ? "#2c6fb8" : "none",
                      color: cur ? "#fff" : branches ? "#7a4ea8" : "#333",
                      fontWeight: cur ? 600 : 400,
                    }}
                    title={cur ? "current position" : "go to this position"}
                  >
                    {moveLabel(n.move, n)}
                  </button>
                </span>
              );
            })}
          </span>
        </div>
      ))}
      {/* Branches, emitted after the line they leave so the main line reads
          uninterrupted. Each is rendered by recursing from its parent. */}
      {blocks.flatMap((b) =>
        b.nodes.flatMap((n) => {
          const parent = node(p.hist, n.parent!);
          if (!parent || parent.children.length < 2 || parent.children[0] !== n.id) return [];
          return parent.children.slice(1).map((vid) => {
            const v = node(p.hist, vid);
            if (!v) return null;
            return (
              <div
                key={vid}
                style={{
                  marginLeft: 12 + depth * 8,
                  paddingLeft: 8,
                  borderLeft: "2px solid #d9cdea",
                  margin: "2px 0 4px",
                }}
              >
                <div style={{ display: "flex", gap: 4, alignItems: "baseline" }}>
                  <button
                    onClick={() => p.onGoto(vid)}
                    style={{
                      ...CHIP,
                      color: vid === p.hist.current ? "#fff" : "#7a4ea8",
                      background: vid === p.hist.current ? "#7a4ea8" : "none",
                      fontWeight: vid === p.hist.current ? 600 : 400,
                    }}
                  >
                    {moveLabel(v.move, v)}
                  </button>
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
                <Line startId={vid} depth={depth + 1} p={p} />
              </div>
            );
          });
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
