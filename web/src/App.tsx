import { useEffect, useState } from "react";
import { engineClientFor } from "./lib/client";

const GAMES = [
  { id: "connect4", label: "Connect 4", engineUrl: "http://127.0.0.1:7100" },
  { id: "go9",      label: "Go 9×9",     engineUrl: "http://127.0.0.1:7101" },
  { id: "amazons",  label: "Amazons",    engineUrl: "http://127.0.0.1:7102" },
  { id: "arimaa",   label: "Arimaa",     engineUrl: "http://127.0.0.1:7103" },
] as const;

type EngineHealth =
  | { status: "unknown" }
  | { status: "ok"; version: string }
  | { status: "error"; message: string };

export function App() {
  const [healths, setHealths] = useState<Record<string, EngineHealth>>(
    Object.fromEntries(GAMES.map((g) => [g.id, { status: "unknown" }]))
  );

  useEffect(() => {
    for (const g of GAMES) {
      (async () => {
        try {
          const client = engineClientFor(g.engineUrl);
          const resp = await client.health({});
          setHealths((prev) => ({
            ...prev,
            [g.id]: { status: "ok", version: resp.version },
          }));
        } catch (e) {
          setHealths((prev) => ({
            ...prev,
            [g.id]: { status: "error", message: (e as Error).message },
          }));
        }
      })();
    }
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 640 }}>
      <h1>playground</h1>
      <p>MuZero-variant research playground. Engines:</p>
      <ul>
        {GAMES.map((g) => (
          <li key={g.id}>
            <strong>{g.label}</strong> @ {g.engineUrl} — <Health h={healths[g.id]} />
          </li>
        ))}
      </ul>
      <p style={{ color: "#888" }}>
        Scaffolding only — boards and gameplay UIs land starting at M1.
        See <code>docs/</code>.
      </p>
    </main>
  );
}

function Health({ h }: { h: EngineHealth }) {
  switch (h.status) {
    case "unknown": return <span>checking…</span>;
    case "ok":      return <span style={{ color: "green" }}>ok (v{h.version})</span>;
    case "error":   return <span style={{ color: "crimson" }}>down ({h.message})</span>;
  }
}
