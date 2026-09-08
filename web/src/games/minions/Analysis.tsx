// Analysis panel: the search tree the net built, as something you can read and
// walk.
//
// The tree arrives flattened — `nodes[0]` is the root and every edge's `child`
// indexes back into the same list — so rendering is a recursive walk over
// indices rather than a nested structure. That shape is deliberate on the wire:
// a tree with 800 nodes nests deeply, and a flat list keeps the encoding cheap
// and lets the panel expand lazily without holding the whole subtree in memory.

import type { AnalyzeNode, AnalyzeEdge } from "../../gen/playground/engine/engine_pb";

/** How a row is ranked. Visits is what the search believed; prior is what the
 *  net proposed before searching, and the disagreement between them is usually
 *  the interesting part. */
export type SortKey = "visits" | "prior" | "value";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Colour a value from the mover's perspective: green good, red bad. */
function valueColor(v: number): string {
  const t = Math.max(-1, Math.min(1, v));
  return t >= 0
    ? `rgb(${Math.round(120 - 40 * t)},${Math.round(168 + 40 * t)},${Math.round(120 - 20 * t)})`
    : `rgb(${Math.round(196 + 40 * t)},${Math.round(112 + 60 * t)},${Math.round(112 + 60 * t)})`;
}

function sortEdges(edges: AnalyzeEdge[], key: SortKey): AnalyzeEdge[] {
  const e = [...edges];
  if (key === "visits") e.sort((a, b) => b.visits - a.visits || b.prior - a.prior);
  else if (key === "prior") e.sort((a, b) => b.prior - a.prior);
  else
    e.sort((a, b) => {
      // Unvisited edges have no value and sink to the bottom rather than
      // sorting as if they were 0 — an unsearched move is not a drawn one.
      const av = a.value ?? -Infinity;
      const bv = b.value ?? -Infinity;
      return bv - av || b.visits - a.visits;
    });
  return e;
}

function EdgeRow({
  edge,
  node,
  depth,
  chosen,
  expanded,
  onToggle,
  onPlay,
  onCommit,
  path,
}: {
  edge: AnalyzeEdge;
  node: AnalyzeNode;
  depth: number;
  chosen: boolean;
  expanded: boolean;
  onToggle: () => void;
  onPlay: () => void;
  /** Absent below the root: only a root candidate is legal in the real game. */
  onCommit?: () => void;
  path: string;
}) {
  const share = node.totalVisits > 0 ? edge.visits / node.totalVisits : 0;
  const hasChild = edge.child !== undefined;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1.1rem 1.3rem 3.4rem 3.9rem 3.6rem 3.6rem 1fr",
        gap: "0.35rem",
        alignItems: "center",
        padding: "0.12rem 0.25rem",
        paddingLeft: `${0.25 + depth * 0.85}rem`,
        fontSize: "0.78rem",
        fontVariantNumeric: "tabular-nums",
        borderLeft: chosen ? "2px solid #e0a72c" : "2px solid transparent",
        background: "linear-gradient(to right, rgba(90,140,200,0.16) 0%, rgba(90,140,200,0.16) " +
          `${(share * 100).toFixed(1)}%, transparent ${(share * 100).toFixed(1)}%)`,
      }}
    >
      <span
        onClick={hasChild ? onToggle : undefined}
        title={hasChild ? (expanded ? "collapse" : "expand") : "not expanded by the search"}
        style={{ cursor: hasChild ? "pointer" : "default", color: hasChild ? "#bbb" : "#555", userSelect: "none" }}
      >
        {hasChild ? (expanded ? "▾" : "▸") : "·"}
      </span>

      {/* Commit the turn to the real game and re-root the analysis here.
          Only offered at the root: a deeper turn is legal in the variation but
          not in the position the game is actually in, so playing it would need
          every turn above it played first — which is what walking the line and
          playing from there already does, one step at a time. */}
      {onCommit ? (
        <span
          onClick={(e) => {
            e.stopPropagation();
            onCommit();
          }}
          title="play this turn on the board and re-analyse from there"
          style={{ cursor: "pointer", color: "#6a9", userSelect: "none" }}
        >
          ▶
        </span>
      ) : (
        <span />
      )}

      <span title="visits, and share of this node's total">
        {edge.visits}
        {edge.forced > 0 && (
          // Visits a floor demanded rather than PUCT choosing. Worth showing:
          // an edge whose visits are all forced was never actually preferred.
          <span style={{ color: "#8a7fb0" }} title={`${edge.forced} forced`}>
            {" "}f{edge.forced}
          </span>
        )}
      </span>

      <span title="prior probability from the energy head" style={{ color: "#9ab" }}>
        {pct(edge.prior)}
      </span>

      <span title="energy f(s,a), shifted so the best candidate reads 0" style={{ color: "#87a" }}>
        {edge.energy === -Infinity ? "−∞" : edge.energy.toFixed(2)}
      </span>

      <span
        title={edge.value === undefined ? "never visited" : "mean value over this edge's visits"}
        style={{ color: edge.value === undefined ? "#555" : valueColor(edge.value) }}
      >
        {edge.value === undefined ? "—" : edge.value.toFixed(3)}
      </span>

      <span
        onClick={onPlay}
        title={`${path}\n\nclick to play this line on the board`}
        style={{
          cursor: "pointer",
          color: chosen ? "#e0a72c" : "#ccc",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {edge.explore && (
          // Injected by forced exploration rather than proposed by the sampler.
          <span style={{ color: "#6a9", marginRight: "0.25rem" }} title="explore candidate">
            ⚑
          </span>
        )}
        {edge.label || "pass"}
      </span>
    </div>
  );
}

export function AnalysisTree({
  nodes,
  chosen,
  sort,
  expanded,
  onToggle,
  onPlay,
  onCommit,
  maxDepth,
}: {
  nodes: AnalyzeNode[];
  chosen?: number;
  sort: SortKey;
  /** Keys are `"nodeIndex:edgeIndex"`, so an edge expands independently of any
   *  other edge that happens to reach the same child. */
  expanded: Set<string>;
  onToggle: (key: string) => void;
  /** The sequence of turn encodings from the root down to the clicked edge. */
  onPlay: (line: Uint8Array[], labels: string[]) => void;
  /** Commit a root candidate to the real game. */
  onCommit: (turn: Uint8Array) => void;
  maxDepth: number;
}) {
  if (nodes.length === 0) return null;

  const rows: React.ReactNode[] = [];

  const walk = (nodeIdx: number, depth: number, line: Uint8Array[], labels: string[]) => {
    if (depth > maxDepth) return;
    const node = nodes[nodeIdx];
    if (!node) return;
    // Only edges the search actually looked at. A node holds every candidate it
    // generated, and listing the unvisited ones would bury the searched line in
    // dozens of rows the search never considered — they stay reachable by
    // switching the sort to prior.
    const shown = sortEdges(node.edges, sort).filter(
      (e) => sort === "prior" || e.visits > 0 || depth === 0,
    );
    for (const edge of shown) {
      const ei = node.edges.indexOf(edge);
      const key = `${nodeIdx}:${ei}`;
      const isOpen = expanded.has(key);
      const nextLine = [...line, edge.turn];
      const nextLabels = [...labels, edge.label || "pass"];
      rows.push(
        <EdgeRow
          key={key}
          edge={edge}
          node={node}
          depth={depth}
          chosen={depth === 0 && chosen !== undefined && ei === chosen}
          expanded={isOpen}
          onToggle={() => onToggle(key)}
          onPlay={() => onPlay(nextLine, nextLabels)}
          onCommit={depth === 0 ? () => onCommit(edge.turn) : undefined}
          path={nextLabels.join("  →  ")}
        />,
      );
      if (isOpen && edge.child !== undefined) {
        walk(edge.child, depth + 1, nextLine, nextLabels);
      }
    }
  };

  walk(0, 0, [], []);

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.1rem 1.3rem 3.4rem 3.9rem 3.6rem 3.6rem 1fr",
          gap: "0.35rem",
          padding: "0.15rem 0.25rem 0.15rem 0.25rem",
          fontSize: "0.7rem",
          color: "#777",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          borderBottom: "1px solid #333",
        }}
      >
        <span />
        <span />
        <span>visits</span>
        <span>prior</span>
        <span>energy</span>
        <span>value</span>
        <span>turn</span>
      </div>
      <div style={{ maxHeight: "26rem", overflowY: "auto" }}>{rows}</div>
      {rows.length === 0 && (
        <div style={{ padding: "0.5rem", color: "#777", fontSize: "0.8rem" }}>
          No visited candidates — raise the simulation budget.
        </div>
      )}
    </div>
  );
}
