import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
const pct = (x) => `${(x * 100).toFixed(1)}%`;
/** Colour a value from the mover's perspective: green good, red bad. */
function valueColor(v) {
    const t = Math.max(-1, Math.min(1, v));
    return t >= 0
        ? `rgb(${Math.round(120 - 40 * t)},${Math.round(168 + 40 * t)},${Math.round(120 - 20 * t)})`
        : `rgb(${Math.round(196 + 40 * t)},${Math.round(112 + 60 * t)},${Math.round(112 + 60 * t)})`;
}
function sortEdges(edges, key) {
    const e = [...edges];
    if (key === "visits")
        e.sort((a, b) => b.visits - a.visits || b.prior - a.prior);
    else if (key === "prior")
        e.sort((a, b) => b.prior - a.prior);
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
function EdgeRow({ edge, node, depth, chosen, expanded, onToggle, onPlay, path, }) {
    const share = node.totalVisits > 0 ? edge.visits / node.totalVisits : 0;
    const hasChild = edge.child !== undefined;
    return (_jsxs("div", { style: {
            display: "grid",
            gridTemplateColumns: "1.1rem 3.4rem 3.9rem 3.6rem 3.6rem 1fr",
            gap: "0.35rem",
            alignItems: "center",
            padding: "0.12rem 0.25rem",
            paddingLeft: `${0.25 + depth * 0.85}rem`,
            fontSize: "0.78rem",
            fontVariantNumeric: "tabular-nums",
            borderLeft: chosen ? "2px solid #e0a72c" : "2px solid transparent",
            background: "linear-gradient(to right, rgba(90,140,200,0.16) 0%, rgba(90,140,200,0.16) " +
                `${(share * 100).toFixed(1)}%, transparent ${(share * 100).toFixed(1)}%)`,
        }, children: [_jsx("span", { onClick: hasChild ? onToggle : undefined, title: hasChild ? (expanded ? "collapse" : "expand") : "not expanded by the search", style: { cursor: hasChild ? "pointer" : "default", color: hasChild ? "#bbb" : "#555", userSelect: "none" }, children: hasChild ? (expanded ? "▾" : "▸") : "·" }), _jsxs("span", { title: "visits, and share of this node's total", children: [edge.visits, edge.forced > 0 && (
                    // Visits a floor demanded rather than PUCT choosing. Worth showing:
                    // an edge whose visits are all forced was never actually preferred.
                    _jsxs("span", { style: { color: "#8a7fb0" }, title: `${edge.forced} forced`, children: [" ", "f", edge.forced] }))] }), _jsx("span", { title: "prior probability from the energy head", style: { color: "#9ab" }, children: pct(edge.prior) }), _jsx("span", { title: "energy f(s,a), shifted so the best candidate reads 0", style: { color: "#87a" }, children: edge.energy === -Infinity ? "−∞" : edge.energy.toFixed(2) }), _jsx("span", { title: edge.value === undefined ? "never visited" : "mean value over this edge's visits", style: { color: edge.value === undefined ? "#555" : valueColor(edge.value) }, children: edge.value === undefined ? "—" : edge.value.toFixed(3) }), _jsxs("span", { onClick: onPlay, title: `${path}\n\nclick to play this line on the board`, style: {
                    cursor: "pointer",
                    color: chosen ? "#e0a72c" : "#ccc",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                }, children: [edge.explore && (
                    // Injected by forced exploration rather than proposed by the sampler.
                    _jsx("span", { style: { color: "#6a9", marginRight: "0.25rem" }, title: "explore candidate", children: "\u2691" })), edge.label || "pass"] })] }));
}
export function AnalysisTree({ nodes, chosen, sort, expanded, onToggle, onPlay, maxDepth, }) {
    if (nodes.length === 0)
        return null;
    const rows = [];
    const walk = (nodeIdx, depth, line, labels) => {
        if (depth > maxDepth)
            return;
        const node = nodes[nodeIdx];
        if (!node)
            return;
        // Only edges the search actually looked at. A node holds every candidate it
        // generated, and listing the unvisited ones would bury the searched line in
        // dozens of rows the search never considered — they stay reachable by
        // switching the sort to prior.
        const shown = sortEdges(node.edges, sort).filter((e) => sort === "prior" || e.visits > 0 || depth === 0);
        for (const edge of shown) {
            const ei = node.edges.indexOf(edge);
            const key = `${nodeIdx}:${ei}`;
            const isOpen = expanded.has(key);
            const nextLine = [...line, edge.turn];
            const nextLabels = [...labels, edge.label || "pass"];
            rows.push(_jsx(EdgeRow, { edge: edge, node: node, depth: depth, chosen: depth === 0 && chosen !== undefined && ei === chosen, expanded: isOpen, onToggle: () => onToggle(key), onPlay: () => onPlay(nextLine, nextLabels), path: nextLabels.join("  →  ") }, key));
            if (isOpen && edge.child !== undefined) {
                walk(edge.child, depth + 1, nextLine, nextLabels);
            }
        }
    };
    walk(0, 0, [], []);
    return (_jsxs("div", { children: [_jsxs("div", { style: {
                    display: "grid",
                    gridTemplateColumns: "1.1rem 3.4rem 3.9rem 3.6rem 3.6rem 1fr",
                    gap: "0.35rem",
                    padding: "0.15rem 0.25rem 0.15rem 0.25rem",
                    fontSize: "0.7rem",
                    color: "#777",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    borderBottom: "1px solid #333",
                }, children: [_jsx("span", {}), _jsx("span", { children: "visits" }), _jsx("span", { children: "prior" }), _jsx("span", { children: "energy" }), _jsx("span", { children: "value" }), _jsx("span", { children: "turn" })] }), _jsx("div", { style: { maxHeight: "26rem", overflowY: "auto" }, children: rows }), rows.length === 0 && (_jsx("div", { style: { padding: "0.5rem", color: "#777", fontSize: "0.8rem" }, children: "No visited candidates \u2014 raise the simulation budget." }))] }));
}
