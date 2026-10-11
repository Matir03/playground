/** Timed matches: a fixed time per turn for each side (asymmetric allowed),
 *  each side's first turn untimed (the net's is a warm-up at a fixed visit
 *  count). A human whose time runs out has the turn ended on the spot; the
 *  net searches until its time and then wraps up. A net seat can play a fixed
 *  visit count per turn instead. */

export interface SideTC {
  /** The net only: "visits" plays a fixed visit count per turn, no clock. */
  mode: "clock" | "visits";
  /** Seconds per turn. */
  secs: number;
  visits: number;
  /** The net's tier-1 visits in flight. */
  inflight: number;
}

export const DEFAULT_TC: SideTC = { mode: "clock", secs: 60, visits: 800, inflight: 32 };

export function fmtClock(ms: number): string {
  const s = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

const num = { width: 52, fontSize: 13 };

function SideSettings({ name, isNet, tc, onChange }: { name: string; isNet: boolean; tc: SideTC; onChange: (t: SideTC) => void }) {
  const set = (p: Partial<SideTC>) => onChange({ ...tc, ...p });
  const clock = !isNet || tc.mode === "clock";
  return (
    <span style={{ display: "inline-flex", gap: 6, alignItems: "center", fontSize: 13 }}>
      <strong style={{ fontWeight: 600 }}>{name}</strong>
      {isNet && (
        <select value={tc.mode} onChange={(e) => set({ mode: e.target.value as SideTC["mode"] })} style={{ fontSize: 13 }}>
          <option value="clock">clock</option>
          <option value="visits">visits / turn</option>
        </select>
      )}
      {clock ? (
        <>
          <input type="number" min={1} step={5} value={tc.secs} style={num} onChange={(e) => set({ secs: Number(e.target.value) })} /> s / turn
        </>
      ) : (
        <input type="number" min={1} step={100} value={tc.visits} style={{ ...num, width: 70 }} onChange={(e) => set({ visits: Number(e.target.value) })} />
      )}
      {isNet && (
        <label title="tier-1 visits searched at once (1 = serial)">
          · in flight <input type="number" min={1} max={256} value={tc.inflight} style={num} onChange={(e) => set({ inflight: Number(e.target.value) })} />
        </label>
      )}
    </span>
  );
}

export interface ClockView {
  /** Time left in the side's current turn, or its full turn when not moving. */
  left: [number, number];
  running: number | null;
  /** Whether the side's current (or next) turn runs its clock. */
  timed: [boolean, boolean];
}

export function TimeControlBar(props: {
  on: boolean;
  setOn: (v: boolean) => void;
  tc: [SideTC, SideTC];
  setTc: (t: [SideTC, SideTC]) => void;
  warmup: number;
  setWarmup: (v: number) => void;
  netSide: [boolean, boolean];
  names: string[];
  view: ClockView;
}) {
  const { on, tc, view, netSide } = props;
  const face = (s: number) => {
    const visits = netSide[s] && tc[s].mode === "visits";
    const run = view.running === s;
    return (
      <span
        key={s}
        style={{
          display: "inline-flex", flexDirection: "column", alignItems: "center", minWidth: 92, padding: "3px 10px",
          borderRadius: 8, border: `2px solid ${run ? "#2f5d8a" : "#d8dbe0"}`,
          background: run && view.left[s] < 10000 && view.timed[s] && !visits ? "#fbe3d6" : run ? "#e8f0f8" : "#fff",
        }}
        title={view.timed[s] || visits ? undefined : "first turn: untimed"}
      >
        <span style={{ fontSize: 11, color: "#555" }}>{props.names[s]}</span>
        <span style={{ fontSize: 20, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
          {visits ? `${tc[s].visits} v` : fmtClock(view.left[s])}
        </span>
        {!view.timed[s] && !visits && <span style={{ fontSize: 10, color: "#777" }}>first turn untimed</span>}
      </span>
    );
  };
  return (
    <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap", margin: "0 0 0.75rem", fontSize: 13 }}>
      <label title="a fixed time per turn for each side; the first turn of each side is untimed">
        <input type="checkbox" checked={on} onChange={(e) => props.setOn(e.target.checked)} /> timed match
      </label>
      {on && (
        <>
          {face(0)}
          {face(1)}
          <span style={{ display: "inline-flex", flexDirection: "column", gap: 4 }}>
            <SideSettings name={props.names[0]} isNet={netSide[0]} tc={tc[0]} onChange={(t) => props.setTc([t, tc[1]])} />
            <SideSettings name={props.names[1]} isNet={netSide[1]} tc={tc[1]} onChange={(t) => props.setTc([tc[0], t])} />
          </span>
          {(netSide[0] || netSide[1]) && (
            <label title="the net's first turn: a fixed visit count, untimed">
              net warm-up turn <input type="number" min={1} step={100} value={props.warmup} style={{ ...num, width: 70 }} onChange={(e) => props.setWarmup(Number(e.target.value))} /> visits
            </label>
          )}
        </>
      )}
    </div>
  );
}
