/** The lattice player's searches: tier 1 (MCTS over complete turns) and
 * tier 2 (the lattice of partial commitments searched inside each turn).
 *
 * Served by the engine's lattice seat (`minions-engine serve --lattice-net
 * <dir>`, JSON on `--lattice-http`, proxied by Vite under `/latapi`; see
 * crates/games/minions/src/lattice_ui.rs). The engine keeps its last few
 * searches whole, so any node of either tier can be opened after the move.
 *
 * Follow mode shows each search the net seat makes as it plays; "analyse"
 * searches the position on the board without playing it. Values are win
 * probabilities for the mover of the node shown.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { create } from "@bufbuild/protobuf";
import { GameId, GameStateSchema, Outcome, type GameState } from "../../gen/playground/game/game_pb";
import { Position } from "./Recordings";
import { SIDE_FILL } from "./Board";

// ── the API (lattice_ui.rs) ──────────────────────────────────────────────────

interface Summary {
  id: number;
  n1: number;
  ms: number;
  turn: number;
  side: number;
  board: string;
  kill: string | null;
  root: number | null;
  chosen: number | null;
  t1_nodes: number;
  lattice_nodes: number;
}

interface T1Child {
  t: number;
  label: string;
  n: number;
  q: number | null;
  prior: number | null;
  sv: number | null;
  terminal: boolean;
  expanded: boolean;
  board: string;
}

interface T1Node {
  t: number;
  n: number;
  q: number;
  side: number;
  turn: number;
  board: string;
  terminal: number | null;
  chosen: number | null;
  lattice: { nodes: number; expansions: number; sinks: number; descents: number } | null;
  children: T1Child[];
}

interface LatKid {
  k: number;
  var: number;
  cell: string;
  label: string;
  prior: number | null;
  v: number | null;
  n: number;
  q: number | null;
  sn: number;
  sq: number | null;
  dead: boolean;
  child: number | null;
}

interface LatNode {
  node: number;
  kind: "inner" | "sink" | "dead";
  stage: string;
  expanded: boolean;
  decided: number;
  n: number | null;
  q: number | null;
  sn: number | null;
  sq: number | null;
  value: number | null;
  board: string;
  kids: LatKid[];
}

async function api<T>(path: string): Promise<T> {
  const r = await fetch(`/latapi${path}`);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error((j as { error?: string }).error ?? `HTTP ${r.status}`);
  return j as T;
}

function hexToBytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(2 * i, 2 * i + 2), 16);
  return out;
}

const toHex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");

const boardState = (hex: string): GameState =>
  create(GameStateSchema, { gameId: GameId.MINIONS, payload: hexToBytes(hex), outcome: Outcome.IN_PROGRESS });

// ── bits ─────────────────────────────────────────────────────────────────────

const pct = (v: number | null | undefined) => (v == null ? "–" : `${(100 * v).toFixed(1)}%`);

/** A value cell: the number, hued from red (0) through amber to green (1). */
function Val({ v, title }: { v: number | null | undefined; title?: string }) {
  if (v == null) return <span style={{ color: "#999" }}>–</span>;
  const t = Math.max(0, Math.min(1, v));
  // Most values sit near even; stretch 0.3..0.7 over the full hue range.
  const h = 120 * Math.max(0, Math.min(1, (t - 0.3) / 0.4));
  return (
    <span style={{ color: `hsl(${h}, 70%, 34%)`, fontVariantNumeric: "tabular-nums", fontWeight: 500 }} title={title}>
      {pct(v)}
    </span>
  );
}

/** A horizontal bar for a share in [0, 1], with an optional second (ghost)
 *  share drawn underneath: prior under visits, say. */
function Bar({ share, ghost, color = "#5b8def", width = 90 }: { share: number; ghost?: number | null; color?: string; width?: number }) {
  return (
    <span style={{ position: "relative", display: "inline-block", width, height: 10, background: "#e6e8ec", borderRadius: 3, verticalAlign: "middle" }}>
      {ghost != null && (
        <span style={{ position: "absolute", left: 0, top: 0, height: 10, width: `${Math.min(1, ghost) * 100}%`, background: "#c9b37a", borderRadius: 3, opacity: 0.8 }} />
      )}
      <span style={{ position: "absolute", left: 0, top: 3, height: 4, width: `${Math.min(1, share) * 100}%`, background: color, borderRadius: 2 }} />
    </span>
  );
}

const th: React.CSSProperties = { textAlign: "left", fontWeight: 600, fontSize: 12, color: "#555", padding: "2px 6px", borderBottom: "1px solid #ddd", whiteSpace: "nowrap" };
const td: React.CSSProperties = { padding: "2px 6px", fontSize: 13, whiteSpace: "nowrap" };
const crumb: React.CSSProperties = { background: "none", border: "none", padding: 0, color: "#2a5db0", cursor: "pointer", font: "inherit" };

function Crumbs({ items, onPick }: { items: string[]; onPick: (i: number) => void }) {
  return (
    <div style={{ fontSize: 13, margin: "0 0 6px", lineHeight: 1.6 }}>
      {items.map((s, i) => (
        <span key={i}>
          {i > 0 && <span style={{ color: "#999" }}> › </span>}
          {i === items.length - 1 ? <strong>{s}</strong> : <button style={crumb} onClick={() => onPick(i)}>{s}</button>}
        </span>
      ))}
    </div>
  );
}

// ── the panel ────────────────────────────────────────────────────────────────

export function LatticePanel({ state }: { state: GameState | null }) {
  const [follow, setFollow] = useState(true);
  const [n1, setN1] = useState(400);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  /** Tier-1 nodes from the root to the one shown, with the turn labels that
   *  led to each. */
  const [t1Path, setT1Path] = useState<{ t: number; label: string }[]>([]);
  const [t1, setT1] = useState<T1Node | null>(null);
  /** Lattice nodes (of the shown tier-1 node's turn) from its root. */
  const [latPath, setLatPath] = useState<{ node: number; label: string }[]>([]);
  const [lat, setLat] = useState<LatNode | null>(null);
  const [preview, setPreview] = useState<{ board: string; caption: string } | null>(null);
  const [seen, setSeen] = useState(0);

  const open = useCallback(async (id: number) => {
    setError(null);
    try {
      const s = await api<Summary>(`/search?id=${id}`);
      setSummary(s);
      setT1Path(s.root == null ? [] : [{ t: 0, label: `turn ${s.turn}` }]);
      setPreview(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  // Follow the net seat: open each new search it makes.
  useEffect(() => {
    if (!follow) return;
    let stop = false;
    const tick = async () => {
      try {
        const { id } = await api<{ id: number }>("/last");
        if (!stop && id > 0 && id !== seen) {
          setSeen(id);
          await open(id);
        }
      } catch {
        // The engine may not be serving the panel; the error shows on demand.
      }
    };
    void tick();
    const h = setInterval(() => void tick(), 2000);
    return () => {
      stop = true;
      clearInterval(h);
    };
  }, [follow, seen, open]);

  const analyse = useCallback(async () => {
    if (!state) return;
    setBusy(true);
    setError(null);
    setFollow(false);
    try {
      const s = await api<Summary>(`/analyze?n1=${n1}&state=${toHex(state.payload)}`);
      setSummary(s);
      setT1Path(s.root == null ? [] : [{ t: 0, label: `turn ${s.turn}` }]);
      setPreview(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [state, n1]);

  // The tier-1 node shown, and its lattice from the root.
  const t1Id = t1Path.length ? t1Path[t1Path.length - 1].t : null;
  useEffect(() => {
    if (!summary || t1Id == null) {
      setT1(null);
      return;
    }
    let live = true;
    api<T1Node>(`/t1?id=${summary.id}&t=${t1Id}`)
      .then((n) => {
        if (!live) return;
        setT1(n);
        setLatPath(n.lattice ? [{ node: 0, label: "no commitments" }] : []);
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [summary, t1Id]);

  const latId = latPath.length ? latPath[latPath.length - 1].node : null;
  useEffect(() => {
    if (!summary || t1Id == null || latId == null) {
      setLat(null);
      return;
    }
    let live = true;
    api<LatNode>(`/lat?id=${summary.id}&t=${t1Id}&node=${latId}`)
      .then((n) => live && setLat(n))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [summary, t1Id, latId]);

  /** Bars are scaled to the largest entry, so the leaders read at a glance. */
  /** Follow forced commitments (variables with one live choice) down from
   *  the node shown, to the first node with a real decision. */
  const skipForced = useCallback(async () => {
    if (!summary || t1Id == null || !latPath.length) return;
    let path = [...latPath];
    try {
      for (let guard = 0; guard < 64; guard++) {
        const n = await api<LatNode>(`/lat?id=${summary.id}&t=${t1Id}&node=${path[path.length - 1].node}`);
        if (!n.expanded) break;
        const byVar = new Map<number, LatKid[]>();
        for (const k of n.kids) if (!k.dead) byVar.set(k.var, [...(byVar.get(k.var) ?? []), k]);
        if ([...byVar.values()].some((ks) => ks.length > 1)) break;
        const next = [...byVar.values()].map((ks) => ks[0]).find((k) => k.child != null);
        if (!next) break;
        path = [...path, { node: next.child!, label: next.label }];
      }
      setLatPath(path);
    } catch (e) {
      setError(String(e));
    }
  }, [summary, t1Id, latPath]);

  const t1Max = useMemo(() => (t1 ? Math.max(1, ...t1.children.map((c) => c.n)) : 1), [t1]);
  const t1PriorMax = useMemo(() => (t1 ? Math.max(1e-9, ...t1.children.map((c) => c.prior ?? 0)) : 1), [t1]);

  /** The lattice node's kids, one group per variable (a unit, an enemy or a
   *  spawn cell), each most likely first. */
  const groups = useMemo(() => {
    if (!lat) return [];
    const m = new Map<number, LatKid[]>();
    for (const k of lat.kids) {
      if (!m.has(k.var)) m.set(k.var, []);
      m.get(k.var)!.push(k);
    }
    const out = [...m.entries()].map(([v, ks]) => {
      ks.sort((a, b) => (b.prior ?? 0) - (a.prior ?? 0) || b.n - a.n);
      const n = ks.reduce((a, k) => a + k.n, 0);
      return { v, ks, n };
    });
    out.sort((a, b) => b.n - a.n);
    return out;
  }, [lat]);
  /** Variables with one live choice: nothing to decide, listed on one line. */
  const forced = groups.filter((g) => g.ks.filter((k) => !k.dead).length <= 1);
  const open_ = groups.filter((g) => g.ks.filter((k) => !k.dead).length > 1);

  const shown = preview ?? (lat ? { board: lat.board, caption: latPath.length > 1 ? `turn so far: ${latPath.slice(1).map((p) => p.label).join(", ")}` : "turn start" } : t1 ? { board: t1.board, caption: "turn start" } : summary ? { board: summary.board, caption: "searched position" } : null);

  return (
    <section style={{ marginTop: "1rem", border: "1px solid #d8dbe0", borderRadius: 10, padding: "0.75rem 1rem", background: "#fafbfc" }}>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
        <strong style={{ fontSize: 15 }}>Lattice search</strong>
        <label style={{ fontSize: 13 }}>
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow the net's moves
        </label>
        <span style={{ fontSize: 13 }}>
          tier-1 visits{" "}
          <input type="number" value={n1} min={10} step={50} onChange={(e) => setN1(Number(e.target.value))} style={{ width: 70 }} />
        </span>
        <button onClick={() => void analyse()} disabled={busy || !state}>
          {busy ? "searching…" : "search this position"}
        </button>
        {summary && (
          <span style={{ fontSize: 12, color: "#666" }}>
            search #{summary.id}: turn {summary.turn},{" "}
            <span style={{ color: SIDE_FILL[summary.side], fontWeight: 600 }}>{summary.side === 0 ? "yellow" : "blue"}</span> to move, n1 {summary.n1},{" "}
            {(summary.ms / 1000).toFixed(1)} s, {summary.t1_nodes} turns searched, {summary.lattice_nodes.toLocaleString()} lattice nodes
          </span>
        )}
      </div>
      {error && <div style={{ color: "#b03030", fontSize: 13, marginBottom: 6 }}>{error}</div>}
      {!summary && !error && (
        <div style={{ fontSize: 13, color: "#777" }}>
          No search yet. Seat the net (engine started with <code>serve --lattice-net</code>) and let it move, or analyse the position.
        </div>
      )}
      {summary?.kill && (
        <div style={{ fontSize: 13 }}>
          No search: the mover kills the enemy necromancer — <code>{summary.kill}</code>.
        </div>
      )}

      {summary && t1 && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1.1fr) auto minmax(360px, 1.2fr)", gap: "1rem", alignItems: "start" }}>
          {/* Tier 1 */}
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>Tier 1 — complete turns</div>
            <Crumbs items={t1Path.map((p) => p.label)} onPick={(i) => setT1Path(t1Path.slice(0, i + 1))} />
            <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>
              {t1.n} visits, value <Val v={t1.q} /> for the mover
              {t1.lattice && <> · lattice {t1.lattice.nodes.toLocaleString()} nodes, {t1.lattice.sinks} turns found</>}
            </div>
            <div style={{ maxHeight: 460, overflow: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%" }}>
                <thead>
                  <tr>
                    <th style={th}>turn</th>
                    <th style={th} title="visit share (blue) over prior (gold)">visits</th>
                    <th style={th} title="search value for this node's mover">Q</th>
                    <th style={th} title="the turn's static (net) value">net</th>
                    <th style={th}></th>
                  </tr>
                </thead>
                <tbody>
                  {t1.children.map((c) => {
                    const chosen = t1.chosen === c.t;
                    return (
                      <tr
                        key={c.t}
                        onMouseEnter={() => setPreview({ board: c.board, caption: `after: ${c.label}` })}
                        onMouseLeave={() => setPreview(null)}
                        style={{ background: chosen ? "#fff4d6" : undefined }}
                      >
                        <td style={{ ...td, whiteSpace: "normal", maxWidth: 260, fontSize: 12 }} title={c.label}>
                          {chosen && <span title="played">★ </span>}
                          {c.label}
                          {c.terminal && <span style={{ color: "#b05" }}> (ends game)</span>}
                        </td>
                        <td style={td}>
                          <Bar share={c.n / t1Max} ghost={(c.prior ?? 0) / t1PriorMax} /> <span style={{ fontSize: 12 }}>{c.n}</span>
                        </td>
                        <td style={td}><Val v={c.q} /></td>
                        <td style={td}><Val v={c.sv} /></td>
                        <td style={td}>
                          {c.expanded && (
                            <button style={{ fontSize: 11 }} onClick={() => setT1Path([...t1Path, { t: c.t, label: c.label.length > 40 ? c.label.slice(0, 40) + "…" : c.label }])} title="the opponent's replies and this position's lattice">
                              open
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* The board */}
          <div style={{ textAlign: "center" }}>
            {shown && <Position state={boardState(shown.board)} />}
            <div style={{ fontSize: 12, color: "#555", maxWidth: 330, margin: "4px auto 0" }}>{shown?.caption}</div>
          </div>

          {/* Tier 2 */}
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>Tier 2 — the lattice of this turn</div>
            {!t1.lattice && <div style={{ fontSize: 13, color: "#777" }}>This position was never searched at tier 2.</div>}
            {t1.lattice && (
              <>
                <Crumbs items={latPath.map((p) => p.label)} onPick={(i) => setLatPath(latPath.slice(0, i + 1))} />
                {lat && (
                  <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>
                    {lat.kind === "sink" ? (
                      <>complete turn, net value <Val v={lat.value} /></>
                    ) : lat.kind === "dead" ? (
                      <span style={{ color: "#b03030" }}>dead: cannot be completed to a legal turn</span>
                    ) : !lat.expanded ? (
                      <>{lat.stage}: not expanded (never visited)</>
                    ) : (
                      <>
                        deciding <strong>{lat.stage}</strong> · {lat.decided} decided · {lat.n?.toFixed(0)} visits, Q <Val v={lat.q} />, complete turns below {lat.sn?.toFixed(0)} (mean <Val v={lat.sq} />)
                      </>
                    )}
                  </div>
                )}
                <div style={{ maxHeight: 460, overflow: "auto" }}>
                  {lat?.expanded && forced.length > 0 && (
                    <div style={{ fontSize: 12, color: "#666", margin: "0 0 6px" }}>
                      {open_.length === 0 && (
                        <button style={{ fontSize: 11, marginRight: 6 }} onClick={() => void skipForced()} title="follow the forced commitments to the first real decision">
                          skip forced ▸
                        </button>
                      )}
                      forced:{" "}
                      {forced.map((g, i) => {
                        const k = g.ks.find((x) => !x.dead) ?? g.ks[0];
                        const go = k.child != null && !k.dead;
                        return (
                          <span key={g.v}>
                            {i > 0 && ", "}
                            {go ? (
                              <button style={crumb} onClick={() => setLatPath([...latPath, { node: k.child!, label: k.label }])}>{k.label}</button>
                            ) : (
                              k.label
                            )}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  {lat?.expanded &&
                    open_.map((g) => (
                      <table key={g.v} style={{ borderCollapse: "collapse", width: "100%", marginBottom: 8 }}>
                        <thead>
                          <tr>
                            <th style={th}>{g.ks[0].cell}</th>
                            <th style={th} title="visit share in this variable (blue) over the prior (gold)">prior / visits</th>
                            <th style={th} title="search value (intermediate values included)">Q</th>
                            <th style={th} title="mean of the complete turns found containing it (the training target), and how many">turns</th>
                            <th style={th} title="the net's own value for it">net</th>
                          </tr>
                        </thead>
                        <tbody>
                          {g.ks.map((k) => (
                            <tr
                              key={k.k}
                              style={{ opacity: k.dead ? 0.45 : 1, cursor: k.child != null && !k.dead ? "pointer" : "default" }}
                              onClick={() => {
                                if (k.child != null && !k.dead) setLatPath([...latPath, { node: k.child, label: k.label }]);
                              }}
                              title={k.child != null && !k.dead ? "open this commitment" : k.dead ? "refuted" : "never created"}
                            >
                              <td style={{ ...td, textDecoration: k.dead ? "line-through" : undefined }}>
                                {k.child != null && !k.dead ? "▸ " : "  "}
                                {k.label}
                              </td>
                              <td style={td}>
                                <Bar share={k.n / Math.max(1, ...g.ks.map((x) => x.n))} ghost={k.prior} width={80} /> <span style={{ fontSize: 11, color: "#666" }}>{pct(k.prior)}</span>
                              </td>
                              <td style={td}><Val v={k.q} /></td>
                              <td style={td}>
                                <Val v={k.sq} /> <span style={{ fontSize: 11, color: "#888" }}>{k.sn > 0 ? `(${k.sn.toFixed(0)})` : ""}</span>
                              </td>
                              <td style={td}><Val v={k.v} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
