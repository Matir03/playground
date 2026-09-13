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

/** Visits PUCT chose. `visits` counts every visit, including the ones a floor
 *  forced; those buy a value estimate and are excluded from the policy target,
 *  so they are shown in their own column rather than folded in here. */
const chosenVisits = (e: AnalyzeEdge) => e.visits - e.forced;

/** One column template for the header and every row, so they cannot drift:
 *  expand, visits, forced, draws, prior, energy, value, turn, play. */
const COLUMNS = "1.1rem 3rem 2.8rem 2.8rem 3.9rem 3.6rem 3.6rem 1fr 1.3rem";

function sameLine(a: Uint8Array[], b: Uint8Array[]): boolean {
  return (
    a.length === b.length &&
    a.every((x, i) => x.length === b[i].length && x.every((v, j) => v === b[i][j]))
  );
}

function sortEdges(edges: AnalyzeEdge[], key: SortKey): AnalyzeEdge[] {
  const e = [...edges];
  if (key === "visits")
    e.sort((a, b) => chosenVisits(b) - chosenVisits(a) || b.forced - a.forced || b.prior - a.prior);
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

/** Edge indices of `node` in the order the tree renders them, so a side panel
 *  can label its axes with the same ranks the reader is looking at. */
export function displayOrder(node: AnalyzeNode, key: SortKey): number[] {
  return sortEdges(node.edges, key).map((e) => node.edges.indexOf(e));
}

function EdgeRow({
  edge,
  node,
  energyRef,
  depth,
  chosen,
  expanded,
  onToggle,
  onPlay,
  onCommit,
  path,
  onBoard,
}: {
  edge: AnalyzeEdge;
  node: AnalyzeNode;
  /** The node's top energy. Energies are only identified up to a per-state
   *  additive constant (the loss is a softmax over the candidate set, which is
   *  shift-invariant), so the absolute level is arbitrary and not worth
   *  showing. Subtracting the best candidate's energy turns the column into
   *  `log(p / p_best)` — nats of evidence against the leader, 0 for the leader
   *  itself. */
  energyRef: number;
  depth: number;
  chosen: boolean;
  expanded: boolean;
  onToggle: () => void;
  onPlay: () => void;
  /** Absent below the root: only a root candidate is legal in the real game. */
  onCommit?: () => void;
  path: string;
  /** This row's line is the one on the board. */
  onBoard: boolean;
}) {
  // The bar is the share of the visits PUCT chose, which is the policy
  // target's shape; forced visits would flatten it toward the floor.
  const chosenTotal = node.edges.reduce((s, e) => s + chosenVisits(e), 0);
  const share = chosenTotal > 0 ? chosenVisits(edge) / chosenTotal : 0;
  const hasChild = edge.child !== undefined;
  return (
    // The whole row shows its line on the board, not just the label: clicking
    // through the tree anywhere should move the board with it.
    <div
      onClick={onPlay}
      style={{
        display: "grid",
        gridTemplateColumns: COLUMNS,
        gap: "0.35rem",
        alignItems: "center",
        cursor: "pointer",
        boxShadow: onBoard ? "inset 0 0 0 1px #e0a72c" : undefined,
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
        onClick={
          hasChild
            ? (e) => {
                // Opening a line also shows it: exploring the tree and looking
                // at the board are one action. Stopped here so the row's own
                // handler does not show it a second time.
                e.stopPropagation();
                onToggle();
                onPlay();
              }
            : undefined
        }
        title={hasChild ? (expanded ? "collapse" : "expand, and show this line on the board") : "not expanded by the search"}
        style={{ cursor: hasChild ? "pointer" : "default", color: hasChild ? "#bbb" : "#555", userSelect: "none" }}
      >
        {hasChild ? (expanded ? "▾" : "▸") : "·"}
      </span>

      <span title="visits PUCT chose — forced visits excluded; the bar is this edge's share of them">
        {chosenVisits(edge)}
      </span>

      {/* Visits a floor demanded rather than PUCT choosing. Worth its own
          column: an edge whose visits are all forced was never preferred. */}
      <span
        title={edge.forced > 0 ? `${edge.forced} visits forced by an exploration floor` : "no forced visits"}
        style={{ color: edge.forced > 0 ? "#8a7fb0" : "#555" }}
      >
        {edge.forced > 0 ? edge.forced : "·"}
      </span>

      {/* Sampler draws that landed on this turn. A node draws its `draws`
          setting's worth of proposals; repeats show here, and the shortfall
          they leave is filled with random playouts, which are no draw at all. */}
      <span
        title={
          edge.draws === 0
            ? "a fill: a random playout added because the node's draws left it short of candidates"
            : `${edge.draws} sampler draw${edge.draws === 1 ? "" : "s"} produced this turn`
        }
        style={{ color: edge.draws > 1 ? "#b08030" : edge.draws === 0 ? "#8a7fb0" : "#555" }}
      >
        {edge.draws === 0 ? "fill" : edge.draws}
      </span>

      <span title="prior probability from the energy head" style={{ color: "#9ab" }}>
        {pct(edge.prior)}
      </span>

      <span
        title={`energy f(s,a) relative to this node's best candidate, in nats: log(p/p_best). Raw f = ${edge.energy.toFixed(3)}`}
        style={{ color: "#87a" }}
      >
        {Number.isFinite(edge.energy) ? (edge.energy - energyRef).toFixed(2) : "−∞"}
      </span>

      <span
        title={edge.value === undefined ? "never visited" : "mean value over this edge's visits"}
        style={{ color: edge.value === undefined ? "#555" : valueColor(edge.value) }}
      >
        {edge.value === undefined ? "—" : edge.value.toFixed(3)}
      </span>

      <span
        title={`${path}\n\nclick the row to show this line on the board`}
        style={{
          color: onBoard || chosen ? "#e0a72c" : "#ccc",
          fontWeight: onBoard ? 600 : 400,
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
          style={{ cursor: "pointer", color: "#6a9", userSelect: "none", textAlign: "right" }}
        >
          ▶
        </span>
      ) : (
        <span />
      )}
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
  shown,
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
  onCommit: (turn: Uint8Array, label?: string) => void;
  maxDepth: number;
  /** The line currently on the board, so its row can be marked. */
  shown?: Uint8Array[];
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
    // An edge visited only by a forced floor counts as looked at: it has a
    // value estimate, and hiding it would hide what the floor bought.
    const listed = sortEdges(node.edges, sort).filter(
      (e) => sort === "prior" || e.visits > 0 || e.forced > 0 || depth === 0,
    );
    // Per node, not per tree: each node's candidate set has its own arbitrary
    // offset, so a shift shared across depths would compare incomparable scales.
    const energyRef = node.edges.reduce(
      (m, e) => (Number.isFinite(e.energy) && e.energy > m ? e.energy : m),
      -Infinity,
    );
    for (const edge of listed) {
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
          energyRef={energyRef}
          depth={depth}
          chosen={depth === 0 && chosen !== undefined && ei === chosen}
          expanded={isOpen}
          onToggle={() => onToggle(key)}
          onPlay={() => onPlay(nextLine, nextLabels)}
          onCommit={depth === 0 ? () => onCommit(edge.turn, edge.label || "pass") : undefined}
          path={nextLabels.join("  →  ")}
          onBoard={!!shown && sameLine(nextLine, shown)}
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
          gridTemplateColumns: COLUMNS,
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
        <span title="visits PUCT chose">visits</span>
        <span title="visits an exploration floor forced">forced</span>
        <span title="sampler draws that produced this turn">draws</span>
        <span>prior</span>
        <span>energy</span>
        <span>value</span>
        <span>turn</span>
        <span title="play a root turn on the board">play</span>
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

/** The KR-PUCT influence matrix for one node.
 *
 * Row `i` is what candidate `i` sees when selection scores it: `Q̂(i)` and
 * `N̂(i)` are sums over the whole set weighted by this row, so a row that is
 * bright away from the diagonal is a candidate being judged largely on its
 * neighbours' evidence rather than its own. The diagonal is 1 by construction
 * and carries no information — it is drawn only so the axes are readable.
 *
 * Empty when `kernel_sigma = 0`, which is the search self-play and eval run;
 * there the matrix is the identity and every candidate is scored alone.
 */
export function KernelMatrix({
  node,
  order,
  max = 24,
}: {
  node: AnalyzeNode;
  /** Edge indices in display order, so the axes match the tree above. */
  order: number[];
  max?: number;
}) {
  const k = node.edges.length;
  if (node.kernel.length !== k * k || k === 0) return null;

  // A 40x40 grid of 8px cells is unreadable and slow; the tail of the
  // candidate list is also the part with the least influence on anything.
  const idx = order.slice(0, max);
  const clipped = order.length - idx.length;
  const cell = idx.length > 16 ? 13 : 17;

  // Off-diagonal mass per row: how much of this candidate's score is borrowed.
  // This is the number the matrix exists to communicate, so state it directly
  // rather than leaving it to be eyeballed off the colours.
  const borrowed = idx.map((i) => {
    let s = 0;
    for (let j = 0; j < k; j++) if (j !== i) s += node.kernel[i * k + j];
    return s;
  });
  const worst = borrowed.reduce((m, x) => Math.max(m, x), 0);

  return (
    <div style={{ marginBottom: "0.5rem" }}>
      <div
        style={{
          color: "#777",
          fontSize: "0.7rem",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          marginBottom: "0.25rem",
        }}
      >
        kernel influence
        <span style={{ textTransform: "none", letterSpacing: 0, color: "#666" }}>
          {" "}— row i borrows column j; max off-diagonal {worst.toFixed(2)}
        </span>
      </div>
      <div style={{ display: "flex", gap: "0.4rem", alignItems: "flex-start", overflowX: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${idx.length}, ${cell}px)` }}>
          {idx.map((i, ri) =>
            idx.map((j, ci) => {
              const w = node.kernel[i * k + j];
              const diag = ri === ci;
              return (
                <div
                  key={`${ri}:${ci}`}
                  title={`row ${ri} (${node.edges[i].label || "pass"})\ncol ${ci} (${node.edges[j].label || "pass"})\nweight ${w.toFixed(3)}`}
                  style={{
                    width: cell,
                    height: cell,
                    background: diag
                      ? "rgba(224,167,44,0.45)"
                      : `rgba(90,170,220,${(w * 0.95).toFixed(3)})`,
                    outline: "1px solid #2a2a2a",
                    outlineOffset: -1,
                  }}
                />
              );
            }),
          )}
        </div>
        {/* Per-row borrowed mass, as a bar beside its row. */}
        <div style={{ display: "grid", gridTemplateRows: `repeat(${idx.length}, ${cell}px)` }}>
          {borrowed.map((b, ri) => (
            <div key={ri} style={{ display: "flex", alignItems: "center", height: cell }}>
              <div
                style={{
                  width: `${worst > 0 ? (b / worst) * 42 : 0}px`,
                  height: cell - 5,
                  background: "rgba(138,127,176,0.8)",
                }}
              />
            </div>
          ))}
        </div>
      </div>
      {clipped > 0 && (
        <div style={{ color: "#666", fontSize: "0.7rem", marginTop: "0.2rem" }}>
          showing the top {idx.length} of {order.length} candidates
        </div>
      )}
    </div>
  );
}
