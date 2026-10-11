/** Training-data explorer for the Minions turn-lattice cycle.
 *
 * Reads the logs the cycle writes (served by `web/explorer-plugin.ts`):
 * - self-play games (`lattice-gen --game-log`): each turn's position, the
 *   tier-1 search (top turns, visits, values, priors), the turn picked from
 *   it, the anchored tier-2 search the played turn was drawn from (the
 *   complete turns it reached, and its record: the commitments made and each
 *   kid's target), and the turn played;
 * - the rooted phase's tier-2 records (a sample);
 * - gate and self-check games (`steps-match --game-log`): both seats'
 *   searches, so a candidate's play can be set against the champion's.
 *
 * Two sources can be compared side by side (e.g. a gate against the one four
 * generations back), and every game replays turn by turn with the played
 * turn drawn on the board.
 */

import { useEffect, useMemo, useState } from "react";
import { decode, px, py, HEX_POINTS, SVG_W, SVG_H, Piece, IS_GRAVEYARD } from "./Board";

// ── the logs ─────────────────────────────────────────────────────────────────

interface Source {
  src: string;
  kind: "selfplay" | "match";
  label: string;
  gen: number | null;
}

interface Summary {
  i: number;
  unfinished: boolean;
  /** 0 in progress, 1 first (yellow) wins, 2 second wins, 3 draw. */
  winner: number;
  a?: string;
  b?: string;
  a_first?: boolean;
  pair?: number;
  start_turn: number | null;
  final_turn: number | null;
  n_turns: number;
  kids: number | null;
  visits: number | null;
  top_share: number | null;
  played_top: number | null;
  picked_top: number | null;
  fallback: number;
  anchored: number;
  value_dev: number | null;
  a_turns: number;
}

interface Cand {
  label: string;
  n: number;
  q: number | null;
  prior?: number | null;
  picked?: boolean;
}

interface T2Kid {
  label: string;
  n: number;
  target: number | null;
  mean: number | null;
  returns: number;
}

interface T2Rec {
  pos: string;
  turn: number;
  side: number;
  commits: string[];
  n: number;
  q: number | null;
  kids: T2Kid[];
}

interface Turn {
  pos: string;
  turn: number;
  side: number;
  mover?: "a" | "b";
  visits?: number;
  kids?: number;
  value?: number;
  secs?: number;
  cands?: Cand[];
  picked?: string;
  picked_n?: number;
  played?: string;
  necro_kill?: boolean;
  anchor?: {
    from: "anchor" | "fallback";
    reached: { label: string; n: number; v: number; played: boolean }[];
    n_reached: number;
    record: T2Rec | null;
  };
}

interface Game {
  unfinished?: boolean;
  winner: number;
  final?: string;
  final_turn?: number;
  a?: string;
  b?: string;
  a_first?: boolean;
  pair?: number;
  turns: Turn[];
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as T;
}

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return out;
}

const shortNet = (s?: string) => (s ? s.replace(/^.*\//, "").replace(/^(.*:)?/, "") : "?");
const fmt = (x: number | null | undefined, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
const pct = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(0)}%`);
const mean = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x != null && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
/** Whether seat A won a match game. */
const aWon = (g: { winner: number; a_first?: boolean }) =>
  (g.winner === 1 && g.a_first) || (g.winner === 2 && g.a_first === false);
const aLost = (g: { winner: number; a_first?: boolean }) =>
  (g.winner === 2 && g.a_first) || (g.winner === 1 && g.a_first === false);

// ── aggregate stats ──────────────────────────────────────────────────────────

interface Stat {
  key: string;
  label: string;
  value: (s: Summary[]) => string;
  hint?: string;
}

const done = (s: Summary[]) => s.filter((g) => !g.unfinished);

const STATS: Stat[] = [
  { key: "games", label: "games (finished)", value: (s) => `${s.length} (${done(s).length})` },
  {
    key: "first",
    label: "yellow / blue / draw",
    hint: "who won the finished games, by colour (yellow moves first)",
    value: (s) => {
      const d = done(s);
      const n = Math.max(1, d.length);
      const c = (w: number) => d.filter((g) => g.winner === w).length / n;
      return `${pct(c(1))} / ${pct(c(2))} / ${pct(c(3))}`;
    },
  },
  {
    key: "len",
    label: "final turn (mean)",
    value: (s) => fmt(mean(done(s).map((g) => g.final_turn)), 1),
  },
  {
    key: "early",
    label: "ended before the cap",
    hint: "finished games whose final turn is under 30: a necromancer kill or graveyard win",
    value: (s) => {
      const d = done(s);
      return pct(d.filter((g) => (g.final_turn ?? 30) < 30).length / Math.max(1, d.length));
    },
  },
  { key: "start", label: "start turn (mean)", value: (s) => fmt(mean(s.map((g) => g.start_turn)), 1) },
  { key: "kids", label: "root turns per search", value: (s) => fmt(mean(s.map((g) => g.kids)), 1) },
  { key: "visits", label: "root visits", value: (s) => fmt(mean(s.map((g) => g.visits)), 0) },
  {
    key: "top",
    label: "top turn's visit share",
    hint: "how concentrated the root search is",
    value: (s) => pct(mean(s.map((g) => g.top_share))),
  },
  {
    key: "ptop",
    label: "played = most visited",
    value: (s) => pct(mean(s.map((g) => g.played_top))),
  },
  {
    key: "pick",
    label: "picked = most visited",
    hint: "self-play: the noisy pick from root visits, before the anchored search",
    value: (s) => pct(mean(s.map((g) => g.picked_top))),
  },
  {
    key: "fb",
    label: "anchor fallbacks",
    hint: "anchored searches that reached no complete turn, so the picked turn was played",
    value: (s) => {
      const a = s.reduce((x, g) => x + g.anchored, 0);
      return a ? `${pct(s.reduce((x, g) => x + g.fallback, 0) / a)} of ${a}` : "–";
    },
  },
  {
    key: "dev",
    label: "|root value − 0.5|",
    hint: "how decided the searched positions look to the net",
    value: (s) => fmt(mean(s.map((g) => g.value_dev)), 3),
  },
  {
    key: "ascore",
    label: "A score (matches)",
    value: (s) => {
      const d = done(s).filter((g) => g.a !== undefined);
      if (!d.length) return "–";
      const w = d.filter(aWon).length;
      const l = d.filter(aLost).length;
      return `${fmt((w + 0.5 * (d.length - w - l)) / d.length, 3)} (${w}-${l})`;
    },
  },
  {
    key: "alen",
    label: "length of A's wins / losses",
    value: (s) => {
      const d = done(s).filter((g) => g.a !== undefined);
      if (!d.length) return "–";
      return `${fmt(mean(d.filter(aWon).map((g) => g.final_turn)), 1)} / ${fmt(mean(d.filter(aLost).map((g) => g.final_turn)), 1)}`;
    },
  },
];

function StatsTable({ cols }: { cols: { label: string; s: Summary[] }[] }) {
  return (
    <table style={{ borderCollapse: "collapse", fontSize: 13, marginBottom: "1rem" }}>
      <thead>
        <tr>
          <th />
          {cols.map((c) => (
            <th key={c.label} style={{ textAlign: "right", padding: "2px 10px" }}>
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {STATS.map((st) => (
          <tr key={st.key} title={st.hint}>
            <td style={{ color: "#555", padding: "2px 10px 2px 0" }}>{st.label}</td>
            {cols.map((c) => (
              <td key={c.label} style={{ textAlign: "right", padding: "2px 10px", fontVariantNumeric: "tabular-nums" }}>
                {st.value(c.s)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── the board, with a turn drawn on it ───────────────────────────────────────

/** The engine's cell names (`map::cell_name`: file a-g, row 0-6), which the
 *  logged turn labels use -- not `history.ts`'s 1-based ones. */
const cellName = (i: number) => `${"abcdefg"[i % 7]}${Math.floor(i / 7)}`;
const CELL_OF = new Map(Array.from({ length: 49 }, (_, i) => [cellName(i), i]));

interface Marks {
  moves: [number, number][];
  attacks: [number, number][];
  spawns: [number, string][];
}

/** The actions of a turn label (`Zc0→d1`, `d1✕e2`, `+Z`, `Z@b1`). */
function marksOf(label?: string): Marks {
  const m: Marks = { moves: [], attacks: [], spawns: [] };
  for (const tok of (label ?? "").split(/\s+/)) {
    let x = /^[A-Z]([a-g]\d)→([a-g]\d)$/.exec(tok);
    if (x) {
      const [a, b] = [CELL_OF.get(x[1]), CELL_OF.get(x[2])];
      if (a != null && b != null) m.moves.push([a, b]);
      continue;
    }
    x = /^([a-g]\d)✕([a-g]\d)$/.exec(tok);
    if (x) {
      const [a, b] = [CELL_OF.get(x[1]), CELL_OF.get(x[2])];
      if (a != null && b != null) m.attacks.push([a, b]);
      continue;
    }
    x = /^([A-Z])@([a-g]\d)$/.exec(tok);
    if (x) {
      const c = CELL_OF.get(x[2]);
      if (c != null) m.spawns.push([c, x[1]]);
    }
  }
  return m;
}

function Arrow({ from, to, color, dash }: { from: number; to: number; color: string; dash?: boolean }) {
  const [x1, y1, x2, y2] = [px(from), py(from), px(to), py(to)];
  const d = Math.hypot(x2 - x1, y2 - y1) || 1;
  const k = (d - 9) / d;
  return (
    <line
      x1={x1} y1={y1} x2={x1 + (x2 - x1) * k} y2={y1 + (y2 - y1) * k}
      stroke={color} strokeWidth={3} strokeDasharray={dash ? "5 3" : undefined}
      markerEnd={`url(#ex-head-${color.slice(1)})`} opacity={0.9}
    />
  );
}

function ExBoard({ pos, label, coords, title }: { pos: string; label?: string; coords: boolean; title?: string }) {
  const d = decode(hexToBytes(pos));
  const m = marksOf(label);
  const heads = ["#f5f5f5", "#ff5a5a"];
  return (
    <div>
      {title && <div style={{ fontSize: 12, color: "#555", marginBottom: 2 }}>{title}</div>}
      <svg width={SVG_W} height={SVG_H} style={{ background: "#20242b", borderRadius: 8, flex: "0 0 auto" }}>
        <defs>
          {heads.map((c) => (
            <marker key={c} id={`ex-head-${c.slice(1)}`} viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
            </marker>
          ))}
        </defs>
        {Array.from({ length: d.cells.length }, (_, i) => (
          <g key={i} transform={`translate(${px(i)},${py(i)})`}>
            <polygon points={HEX_POINTS} fill={IS_GRAVEYARD[i] ? "#4a4438" : "#2c313a"} stroke="#454b55" strokeWidth={1} />
            {IS_GRAVEYARD[i] && <polygon points={HEX_POINTS} transform="scale(0.8)" fill="none" stroke="#c8b06a" strokeWidth={1.5} />}
            {IS_GRAVEYARD[i] && d.cells[i] === 0 && (
              <text textAnchor="middle" dy="0.35em" fontSize={15} fill="#9a8f74">†</text>
            )}
            {d.cells[i] !== 0 && <Piece cell={d.cells[i]} pstate={d.pstate[i]} grave={IS_GRAVEYARD[i]} />}
            {coords && (
              <text textAnchor="middle" y={-11} fontSize={8} fill="#8a909a">{cellName(i)}</text>
            )}
          </g>
        ))}
        {m.moves.map(([a, b], k) => <Arrow key={`m${k}`} from={a} to={b} color="#f5f5f5" />)}
        {m.attacks.map(([a, b], k) => <Arrow key={`a${k}`} from={a} to={b} color="#ff5a5a" dash />)}
        {m.spawns.map(([c, u], k) => (
          <g key={`s${k}`} transform={`translate(${px(c)},${py(c)})`}>
            <circle r={13} fill="none" stroke="#7CFC9A" strokeWidth={2.5} />
            <text textAnchor="middle" y={22} fontSize={9} fill="#7CFC9A">+{u}</text>
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 12, color: "#555", marginTop: 2 }}>
        turn {d.turn} · {d.side === 0 ? "yellow" : "blue"} to move · ${d.money[0]} / ${d.money[1]}
        {d.winner ? ` · ${["", "yellow wins", "blue wins", "draw"][d.winner]}` : ""}
      </div>
    </div>
  );
}

// ── one turn's details ───────────────────────────────────────────────────────

const th: React.CSSProperties = { textAlign: "left", padding: "2px 8px 2px 0", fontWeight: 600, borderBottom: "1px solid #ddd" };
const td: React.CSSProperties = { padding: "2px 8px 2px 0", verticalAlign: "top" };
const num: React.CSSProperties = { ...td, textAlign: "right", fontVariantNumeric: "tabular-nums" };

function CandTable({ cands, played, picked, onHover }: { cands: Cand[]; played?: string; picked?: string; onHover: (l?: string) => void }) {
  const total = cands.reduce((a, c) => a + c.n, 0);
  return (
    <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
      <thead>
        <tr>
          <th style={th}>turn</th>
          <th style={{ ...th, textAlign: "right" }}>visits</th>
          <th style={{ ...th, textAlign: "right" }}>Q</th>
          <th style={{ ...th, textAlign: "right" }}>prior</th>
        </tr>
      </thead>
      <tbody>
        {cands.map((c, k) => (
          <tr
            key={k}
            onMouseEnter={() => onHover(c.label)}
            onMouseLeave={() => onHover(undefined)}
            style={{ background: c.label === played ? "#e3f1e0" : c.label === picked ? "#fff4d6" : undefined, cursor: "default" }}
          >
            <td style={{ ...td, fontFamily: "monospace", fontSize: 11 }}>
              {c.label}
              {c.label === played && <strong> ← played</strong>}
              {c.label === picked && c.label !== played && <strong> ← picked</strong>}
            </td>
            <td style={num}>
              {c.n} <span style={{ color: "#888" }}>({pct(c.n / Math.max(1, total))})</span>
            </td>
            <td style={num}>{fmt(c.q, 3)}</td>
            <td style={num}>{c.prior != null ? pct(c.prior) : "–"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function T2View({ rec, hideIdle }: { rec: T2Rec; hideIdle: boolean }) {
  const idle = (l: string) => l.startsWith("spare ") || l.startsWith("no spawn");
  const commits = hideIdle ? rec.commits.filter((c) => !idle(c)) : rec.commits;
  const kids = [...rec.kids].sort((a, b) => b.n - a.n);
  return (
    <div style={{ fontSize: 12 }}>
      <div style={{ marginBottom: 4 }}>
        <strong>commitments ({rec.commits.length}{hideIdle ? `, ${rec.commits.length - commits.length} idle hidden` : ""}):</strong>{" "}
        <span style={{ fontFamily: "monospace", fontSize: 11 }}>{commits.join(" · ") || "none (turn start)"}</span>
      </div>
      <div style={{ marginBottom: 4 }}>
        node: {fmt(rec.n, 0)} visits, target {fmt(rec.q, 3)}
      </div>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={th}>next commitment</th>
            <th style={{ ...th, textAlign: "right" }}>visit share</th>
            <th style={{ ...th, textAlign: "right" }}>target</th>
            <th style={{ ...th, textAlign: "right" }}>mean</th>
            <th style={{ ...th, textAlign: "right" }}>returns</th>
          </tr>
        </thead>
        <tbody>
          {kids.slice(0, 30).map((k, i) => (
            <tr key={i}>
              <td style={{ ...td, fontFamily: "monospace", fontSize: 11 }}>{k.label}</td>
              <td style={num}>{pct(k.n)}</td>
              <td style={num}>{fmt(k.target, 3)}</td>
              <td style={num}>{fmt(k.mean, 3)}</td>
              <td style={num}>{fmt(k.returns, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {kids.length > 30 && <div style={{ color: "#888" }}>… {kids.length - 30} more</div>}
    </div>
  );
}

function TurnDetail({ t, next, coords, hideIdle }: { t: Turn; next?: string; coords: boolean; hideIdle: boolean }) {
  const [hover, setHover] = useState<string | undefined>(undefined);
  const shown = hover ?? t.played;
  return (
    <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <ExBoard pos={t.pos} label={shown} coords={coords} title={hover ? "hovered turn" : "turn start, played turn drawn"} />
        {next && <ExBoard pos={next} coords={coords} title="after the turn" />}
      </div>
      <div style={{ flex: "1 1 420px", minWidth: 360 }}>
        <div style={{ fontSize: 13, marginBottom: 6 }}>
          {t.mover && <><strong>seat {t.mover.toUpperCase()}</strong> · </>}
          {t.visits != null && <>{t.visits} visits over {t.kids} turns · root value {fmt(t.value, 3)}</>}
          {t.secs != null && <> · {fmt(t.secs, 1)} s</>}
          {t.necro_kill && <strong>necromancer kill (no search)</strong>}
        </div>
        <div style={{ fontSize: 12, marginBottom: 6 }}>
          <div><strong>played:</strong> <span style={{ fontFamily: "monospace", fontSize: 11 }}>{t.played ?? "–"}</span></div>
          {t.picked !== undefined && t.picked !== t.played && (
            <div><strong>picked from the root:</strong> <span style={{ fontFamily: "monospace", fontSize: 11 }}>{t.picked}</span> ({t.picked_n} visits)</div>
          )}
        </div>
        {t.cands && t.cands.length > 0 && (
          <>
            <h4 style={{ margin: "8px 0 4px" }}>tier-1 search: top turns</h4>
            <CandTable cands={t.cands} played={t.played} picked={t.picked} onHover={setHover} />
          </>
        )}
        {t.anchor && (
          <>
            <h4 style={{ margin: "12px 0 4px" }}>
              anchored tier-2 search{" "}
              <span style={{ fontWeight: 400, color: "#666" }}>
                ({t.anchor.from === "fallback" ? "reached nothing: the picked turn was played" : `${t.anchor.n_reached} complete turns reached`})
              </span>
            </h4>
            {t.anchor.reached.length > 0 && (
              <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%", marginBottom: 8 }}>
                <thead>
                  <tr>
                    <th style={th}>complete turn reached</th>
                    <th style={{ ...th, textAlign: "right" }}>times</th>
                    <th style={{ ...th, textAlign: "right" }}>value</th>
                  </tr>
                </thead>
                <tbody>
                  {t.anchor.reached.map((r, k) => (
                    <tr
                      key={k}
                      onMouseEnter={() => setHover(r.label)}
                      onMouseLeave={() => setHover(undefined)}
                      style={{ background: r.played ? "#e3f1e0" : undefined }}
                    >
                      <td style={{ ...td, fontFamily: "monospace", fontSize: 11 }}>
                        {r.label}
                        {r.played && <strong> ← played</strong>}
                      </td>
                      <td style={num}>{r.n}</td>
                      <td style={num}>{fmt(r.v, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {t.anchor.record && (
              <>
                <h4 style={{ margin: "8px 0 4px" }}>its tier-2 record (training target)</h4>
                <T2View rec={t.anchor.record} hideIdle={hideIdle} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── the view ─────────────────────────────────────────────────────────────────

type Tab = "games" | "t2";

export function Explorer({ onBack }: { onBack: () => void }) {
  const [sources, setSources] = useState<Source[] | null>(null);
  const [src, setSrc] = useState<string>("");
  const [cmp, setCmp] = useState<string>("");
  const [sum, setSum] = useState<Summary[] | null>(null);
  const [cmpSum, setCmpSum] = useState<Summary[] | null>(null);
  const [gameIdx, setGameIdx] = useState<number | null>(null);
  const [game, setGame] = useState<Game | null>(null);
  const [turnIdx, setTurnIdx] = useState(0);
  const [tab, setTab] = useState<Tab>("games");
  const [t2, setT2] = useState<T2Rec[] | null>(null);
  const [t2Idx, setT2Idx] = useState(0);
  const [coords, setCoords] = useState(true);
  const [hideIdle, setHideIdle] = useState(true);
  const [filter, setFilter] = useState<"all" | "finished" | "awins" | "alosses" | "early">("all");
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    getJson<Source[]>("/explorer-data/index")
      .then((s) => {
        setSources(s);
        setSrc((cur) => cur || s[0]?.src || "");
      })
      .catch((e) => setError(String(e)));
  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!src) return;
    setSum(null);
    setGame(null);
    setGameIdx(null);
    setT2(null);
    getJson<Summary[]>(`/explorer-data/${src}/summary`).then(setSum).catch((e) => setError(String(e)));
  }, [src]);

  useEffect(() => {
    if (!cmp) return setCmpSum(null);
    getJson<Summary[]>(`/explorer-data/${cmp}/summary`).then(setCmpSum).catch((e) => setError(String(e)));
  }, [cmp]);

  useEffect(() => {
    if (gameIdx == null || !src) return;
    setGame(null);
    setTurnIdx(0);
    getJson<Game>(`/explorer-data/${src}/game/${gameIdx}`).then(setGame).catch((e) => setError(String(e)));
  }, [src, gameIdx]);

  useEffect(() => {
    if (tab !== "t2" || !src.startsWith("sp") || t2) return;
    getJson<T2Rec[]>(`/explorer-data/${src}/t2`).then(setT2).catch((e) => setError(String(e)));
  }, [tab, src, t2]);

  const cur = sources?.find((s) => s.src === src);
  const isMatch = cur?.kind === "match";
  const rows = useMemo(() => {
    const s = sum ?? [];
    switch (filter) {
      case "finished": return s.filter((g) => !g.unfinished);
      case "awins": return s.filter(aWon);
      case "alosses": return s.filter(aLost);
      case "early": return s.filter((g) => !g.unfinished && (g.final_turn ?? 30) < 30);
      default: return s;
    }
  }, [sum, filter]);

  // Keyboard: left/right through the turns of the open game.
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (!game || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.key === "ArrowRight") setTurnIdx((i) => Math.min(game.turns.length - 1, i + 1));
      if (e.key === "ArrowLeft") setTurnIdx((i) => Math.max(0, i - 1));
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [game]);

  const t = game?.turns[turnIdx];
  const nextPos = game ? (game.turns[turnIdx + 1]?.pos ?? game.final) : undefined;
  const label = (s: Source) => s.label;

  return (
    <>
      <button
        onClick={onBack}
        style={{ marginBottom: "1rem", background: "none", border: "none", color: "#1d4ed8", cursor: "pointer", font: "inherit", padding: 0 }}
      >
        ← Menu
      </button>
      <h1 style={{ marginTop: 0, marginBottom: "0.5rem" }}>Minions training-data explorer</h1>
      <p style={{ marginTop: 0, color: "#555", maxWidth: 900 }}>
        Self-play games as the cycle generated them (tier-1 search, picked turn, anchored tier-2 search, played turn,
        tier-2 targets), the rooted phase's tier-2 records, and gate / self-check games with both seats' searches.
      </p>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap", marginBottom: "0.75rem", fontSize: 14 }}>
        <label>
          source{" "}
          <select value={src} onChange={(e) => setSrc(e.target.value)}>
            {sources?.map((s) => <option key={s.src} value={s.src}>{label(s)}</option>)}
          </select>
        </label>
        <label>
          compare with{" "}
          <select value={cmp} onChange={(e) => setCmp(e.target.value)}>
            <option value="">—</option>
            {sources?.filter((s) => s.src !== src).map((s) => <option key={s.src} value={s.src}>{label(s)}</option>)}
          </select>
        </label>
        <button onClick={load}>refresh list</button>
        <label><input type="checkbox" checked={coords} onChange={(e) => setCoords(e.target.checked)} /> cell names</label>
        <label><input type="checkbox" checked={hideIdle} onChange={(e) => setHideIdle(e.target.checked)} /> hide spare / no-spawn commitments</label>
      </div>
      {sources?.length === 0 && (
        <p style={{ color: "#666" }}>
          No logs yet: the cycle writes self-play games with <code>lattice-gen --game-log</code> and gate games with{" "}
          <code>steps-match --game-log</code>.
        </p>
      )}
      {sum && (
        <StatsTable
          cols={[
            { label: cur ? label(cur) : src, s: sum },
            ...(cmpSum && cmp ? [{ label: label(sources!.find((s) => s.src === cmp)!), s: cmpSum }] : []),
          ]}
        />
      )}
      {isMatch && sum?.[0] && (
        <p style={{ fontSize: 13, marginTop: 0 }}>
          A = <code>{shortNet(sum[0].a)}</code>, B = <code>{shortNet(sum[0].b)}</code>
        </p>
      )}
      {cur?.kind === "selfplay" && (
        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          {(["games", "t2"] as Tab[]).map((k) => (
            <button key={k} onClick={() => setTab(k)} style={{ fontWeight: tab === k ? 700 : 400 }}>
              {k === "games" ? "games" : "rooted tier-2 records"}
            </button>
          ))}
        </div>
      )}
      {(tab === "games" || isMatch) && sum && (
        <div style={{ display: "flex", gap: "1.25rem", alignItems: "flex-start" }}>
          <div style={{ flex: "0 0 auto", maxHeight: "80vh", overflowY: "auto", fontSize: 12, border: "1px solid #eee", borderRadius: 6 }}>
            <div style={{ padding: 6 }}>
              <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
                <option value="all">all games</option>
                <option value="finished">finished</option>
                <option value="early">ended before the cap</option>
                {isMatch && <option value="awins">A won</option>}
                {isMatch && <option value="alosses">A lost</option>}
              </select>{" "}
              {rows.length}
            </div>
            <table style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>#</th>
                  {isMatch && <th style={th}>A</th>}
                  <th style={th}>turns</th>
                  <th style={th}>result</th>
                  <th style={{ ...th, textAlign: "right" }}>kids</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((g) => (
                  <tr
                    key={g.i}
                    onClick={() => setGameIdx(g.i)}
                    style={{ cursor: "pointer", background: g.i === gameIdx ? "#e8f0f8" : undefined }}
                  >
                    <td style={td}>{isMatch ? `${g.pair}${g.a_first ? "a" : "b"}` : g.i}</td>
                    {isMatch && <td style={td}>{g.a_first ? "yellow" : "blue"}</td>}
                    <td style={td}>{g.start_turn}–{g.final_turn ?? "…"}</td>
                    <td style={td}>
                      {g.unfinished
                        ? "unfinished"
                        : isMatch
                          ? aWon(g) ? "A won" : aLost(g) ? "A lost" : "draw"
                          : ["…", "yellow", "blue", "draw"][g.winner]}
                    </td>
                    <td style={num}>{fmt(g.kids, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ flex: "1 1 auto", minWidth: 0 }}>
            {!game && <p style={{ color: "#666" }}>{gameIdx == null ? "Pick a game." : "loading…"}</p>}
            {game && t && (
              <>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                  {game.turns.map((x, k) => (
                    <button
                      key={k}
                      onClick={() => setTurnIdx(k)}
                      title={x.played}
                      style={{
                        minWidth: 34, padding: "2px 4px", fontSize: 12,
                        border: k === turnIdx ? "2px solid #2f5d8a" : "1px solid #ccc",
                        background: x.side === 0 ? "#fbefd0" : "#dbe6f8",
                        fontWeight: x.mover === "a" ? 700 : 400,
                      }}
                    >
                      {x.turn}
                      {x.anchor?.from === "fallback" ? "*" : ""}
                    </button>
                  ))}
                  <span style={{ fontSize: 12, color: "#666", alignSelf: "center" }}>
                    {isMatch ? "bold = seat A · " : ""}← / → keys · {game.unfinished ? "unfinished" : `ended turn ${game.final_turn}, ${["", "yellow wins", "blue wins", "draw"][game.winner]}`}
                  </span>
                </div>
                <TurnDetail t={t} next={nextPos} coords={coords} hideIdle={hideIdle} />
              </>
            )}
          </div>
        </div>
      )}
      {tab === "t2" && !isMatch && (
        <div>
          {!t2 && <p>loading…</p>}
          {t2 && t2.length === 0 && <p>No tier-2 records logged for this generation.</p>}
          {t2 && t2.length > 0 && (
            <>
              <div style={{ marginBottom: 8, fontSize: 13 }}>
                record{" "}
                <input type="number" min={0} max={t2.length - 1} value={t2Idx} onChange={(e) => setT2Idx(Number(e.target.value))} style={{ width: 70 }} />{" "}
                of {t2.length} (a sample of the rooted phase)
              </div>
              <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
                <ExBoard pos={t2[t2Idx].pos} coords={coords} title="turn start" />
                <div style={{ flex: "1 1 420px", minWidth: 360 }}>
                  <T2View rec={t2[t2Idx]} hideIdle={hideIdle} />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
