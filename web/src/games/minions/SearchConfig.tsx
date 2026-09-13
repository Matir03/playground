/**
 * The engine's search config: the settings analysis and the net seat run
 * with. It starts as self-play's recorded search (`search.json`), and every
 * setting can be changed here and restored.
 *
 * The engine sends the settings as JSON and this panel knows them only by
 * name, so a setting added on the engine side shows up (under "other") without
 * a change here.
 */
import { useEffect, useState } from "react";
import { create } from "@bufbuild/protobuf";
import type { Client } from "@connectrpc/connect";
import { ConnectError } from "@connectrpc/connect";
import {
  type EngineService,
  type SearchConfig,
  GetSearchConfigRequestSchema,
  SetSearchConfigRequestSchema,
} from "../../gen/playground/engine/engine_pb";

type Settings = Record<string, number | boolean>;
/** Inputs hold text while being edited, so a half-typed number is not
 *  refused mid-keystroke; it is parsed on apply. */
type Draft = Record<string, string | boolean>;

const GROUPS: { title: string; note?: string; fields: [string, string][] }[] = [
  {
    title: "search",
    fields: [
      ["sims", "Simulations per full search."],
      ["c_puct", "PUCT exploration constant."],
      [
        "forced_visits",
        "Visits guaranteed to each root candidate holding a forced quota. Excluded from PUCT and from the policy target; they buy a value estimate. Quotas total at most half the simulations.",
      ],
      ["forced_extra", "Non-explore root candidates also given a quota, drawn at random."],
      ["k_forced", "KataGo forced-playout constant. 0 = off."],
      ["min_explore_visits", "Visits guaranteed to every explore candidate at the root."],
      [
        "temp_moves",
        "Plies whose turn is picked in proportion to chosen visits rather than the most visited. The seat's move and the analysis 'chosen' row follow it. 0 = always the most visited.",
      ],
      ["kernel_sigma", "KR-PUCT kernel bandwidth. 0 = flat PUCT. Above 0 the influence matrix fills in."],
    ],
  },
  {
    title: "widening",
    fields: [
      ["pw_chunk", "Candidates generated per widen. 0 = the whole set at once when a node expands."],
      ["pw_alpha", "Interior nodes hold ceil(pw_beta · n^pw_alpha) candidates after n visits."],
      ["pw_beta", "See pw_alpha. ≤ 0 disables widening."],
      ["pw_resume", "Resume the descent after a widen instead of restarting it from the root."],
    ],
  },
  {
    title: "sampler",
    fields: [
      [
        "candidates",
        "Candidates per node (K). Once a node's proposals repeat, the rest are random playouts. A new K builds a TensorRT engine on first use (slow once).",
      ],
      ["t_steps", "Diffusion rounds per proposal."],
      ["random_fraction", "Share of proposals drawn as random playouts (explore candidates)."],
      ["temperature", "Sampling temperature of the diffusion head."],
      ["use_energy", "Score candidates with the energy head. Off = uniform priors."],
    ],
  },
  {
    title: "fast search",
    note: "used when search = fast: self-play's cheaper search, no forced visits, no random proposals",
    fields: [
      ["fast_sims", "Simulations per fast search."],
      ["candidates_fast", "Candidates per node in a fast search."],
      ["t_steps_fast", "Diffusion rounds per proposal in a fast search."],
    ],
  },
  {
    title: "self-play only",
    note: "not used by analysis or the seat",
    fields: [["full_search_prob", "Share of self-play turns searched with the full search."]],
  },
];

const INTEGER = new Set([
  "sims",
  "fast_sims",
  "forced_visits",
  "forced_extra",
  "temp_moves",
  "pw_chunk",
  "candidates",
  "candidates_fast",
  "t_steps",
  "t_steps_fast",
]);

const errorText = (e: unknown) => (e instanceof ConnectError ? e.message : String(e));

function toDraft(s: Settings): Draft {
  return Object.fromEntries(
    Object.entries(s).map(([k, v]) => [k, typeof v === "boolean" ? v : String(v)]),
  );
}

/** The draft as settings, or what is wrong with it. */
function parse(d: Draft, like: Settings): Settings | string {
  const out: Settings = {};
  for (const [k, v] of Object.entries(d)) {
    if (typeof like[k] === "boolean") {
      out[k] = Boolean(v);
      continue;
    }
    const n = Number(v);
    if (String(v).trim() === "" || !Number.isFinite(n)) return `${k} is not a number`;
    if (INTEGER.has(k) && !Number.isInteger(n)) return `${k} must be a whole number`;
    out[k] = n;
  }
  return out;
}

const same = (a: Settings, b: Settings) => Object.keys(a).every((k) => a[k] === b[k]);

export function SearchConfigPanel({ client }: { client: Client<typeof EngineService> }) {
  const [applied, setApplied] = useState<Settings | null>(null);
  const [initial, setInitial] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Draft>({});
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const take = (res: SearchConfig) => {
    const cur = JSON.parse(res.json) as Settings;
    setApplied(cur);
    setInitial(JSON.parse(res.defaultJson) as Settings);
    setDraft(toDraft(cur));
    setError(null);
  };

  useEffect(() => {
    client
      .getSearchConfig(create(GetSearchConfigRequestSchema, {}))
      .then(take, (e) => setError(errorText(e)));
  }, [client]);

  const send = async (reset: boolean) => {
    if (!applied) return;
    let json = "";
    if (!reset) {
      const s = parse(draft, applied);
      if (typeof s === "string") {
        setError(s);
        return;
      }
      json = JSON.stringify(s);
    }
    setBusy(true);
    try {
      take(await client.setSearchConfig(create(SetSearchConfigRequestSchema, { json, reset })));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!applied || !initial) {
    return error ? (
      <div style={{ color: "#c66", fontSize: "0.75rem", marginBottom: "0.4rem" }}>
        search config: {error}
      </div>
    ) : null;
  }

  const parsed = parse(draft, applied);
  const edited = typeof parsed === "string" || !same(parsed, applied);
  const changed = Object.keys(applied).filter((k) => applied[k] !== initial[k]);
  const known = new Set(GROUPS.flatMap((g) => g.fields.map(([k]) => k)));
  const other = Object.keys(applied).filter((k) => !known.has(k));
  const groups = other.length
    ? [...GROUPS, { title: "other", fields: other.map((k): [string, string] => [k, ""]) }]
    : GROUPS;

  return (
    <div style={{ margin: "0.1rem 0 0.5rem", fontSize: "0.78rem", color: "#999" }}>
      <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={() => setOpen(!open)} style={{ font: "inherit", padding: "0.1rem 0.5rem" }}>
          {open ? "▾" : "▸"} search config
        </button>
        <span>
          {applied.sims} sims · K {applied.candidates} · T {applied.t_steps} · random{" "}
          {applied.random_fraction} · forced {applied.forced_visits} · temp {applied.temp_moves}
        </span>
        <span style={{ color: changed.length ? "#d0a050" : "#6a6" }}>
          {changed.length ? `changed from self-play: ${changed.join(", ")}` : "= self-play's"}
        </span>
      </div>
      {open && (
        <div style={{ marginTop: "0.4rem" }}>
          {groups.map((g) => (
            <div key={g.title} style={{ marginBottom: "0.4rem" }}>
              <div style={{ color: "#bbb", marginBottom: "0.1rem" }}>
                {g.title}
                {"note" in g && g.note && <span style={{ color: "#777" }}> — {g.note}</span>}
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(12.5rem, 1fr))",
                  gap: "0.2rem 0.9rem",
                }}
              >
                {g.fields
                  .filter(([k]) => k in applied)
                  .map(([k, help]) => {
                    const v = draft[k];
                    const off = applied[k] !== initial[k];
                    return (
                      <label
                        key={k}
                        title={off ? `${help}\nself-play: ${String(initial[k])}` : help}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.35rem",
                          color: off ? "#d0a050" : "#999",
                        }}
                      >
                        <span style={{ flex: "1 1 auto", whiteSpace: "nowrap" }}>{k}</span>
                        {off && <span style={{ color: "#777" }}>({String(initial[k])})</span>}
                        {typeof v === "boolean" ? (
                          <input
                            type="checkbox"
                            checked={v}
                            onChange={(e) => setDraft({ ...draft, [k]: e.target.checked })}
                          />
                        ) : (
                          <input
                            type="text"
                            inputMode="decimal"
                            value={v ?? ""}
                            onChange={(e) => setDraft({ ...draft, [k]: e.target.value })}
                            style={{ width: "4.5rem", font: "inherit" }}
                          />
                        )}
                      </label>
                    );
                  })}
              </div>
            </div>
          ))}
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
            <button
              disabled={busy || !edited}
              onClick={() => void send(false)}
              style={{ font: "inherit", padding: "0.15rem 0.6rem" }}
            >
              apply
            </button>
            <button
              disabled={busy || !edited}
              onClick={() => {
                setDraft(toDraft(applied));
                setError(null);
              }}
              style={{ font: "inherit", padding: "0.15rem 0.6rem" }}
            >
              discard edits
            </button>
            <button
              disabled={busy || changed.length === 0}
              onClick={() => void send(true)}
              style={{ font: "inherit", padding: "0.15rem 0.6rem" }}
            >
              reset to self-play
            </button>
            <span style={{ color: "#777" }}>
              applies from the next analysis and the net seat's next turn
            </span>
          </div>
        </div>
      )}
      {error && <div style={{ color: "#c66", marginTop: "0.2rem" }}>{error}</div>}
    </div>
  );
}
