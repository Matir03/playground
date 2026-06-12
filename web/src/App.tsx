import { useEffect, useState } from "react";
import { engineClientFor } from "./lib/client";
import { Connect4Board } from "./games/connect4/Board";
import { Go9Board } from "./games/go9/Board";

// Engines are reached under same-origin proxy paths (see vite.config.ts), so the
// whole app works behind one origin / one public tunnel without CORS.
const GAMES = [
  { id: "connect4", label: "Connect 4", engineUrl: "/c4",  playable: true },
  { id: "go9",      label: "Go 9×9",     engineUrl: "/go9", playable: true },
  { id: "amazons",  label: "Amazons",    engineUrl: "/a",   playable: false },
  { id: "arimaa",   label: "Arimaa",     engineUrl: "/ar",  playable: false },
] as const;

type EngineHealth =
  | { status: "unknown" }
  | { status: "ok"; version: string }
  | { status: "error"; message: string };

const connect4 = GAMES[0];
const connect4Client = engineClientFor(connect4.engineUrl);
const go9 = GAMES[1];
const go9Client = engineClientFor(go9.engineUrl);

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
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 720 }}>
      <h1>playground</h1>

      <h2 style={{ marginBottom: "0.5rem" }}>Connect 4</h2>
      <p style={{ marginTop: 0, color: "#555" }}>
        Set each seat to a human or the built-in perfect solver. The Rust engine
        enforces the rules and, for engine seats, plays optimally.{" "}
        <Health h={healths.connect4} />
      </p>
      {healths.connect4.status === "error" ? (
        <p style={{ color: "crimson" }}>
          Engine not reachable at {connect4.engineUrl}. Start it with{" "}
          <code>cargo run --bin connect4-engine -- serve</code>.
        </p>
      ) : (
        <Connect4Board client={connect4Client} />
      )}

      <h2 style={{ marginTop: "2rem", marginBottom: "0.5rem" }}>Go 9×9</h2>
      <p style={{ marginTop: 0, color: "#555" }}>
        Set each seat to a human or the self-play net. The Rust engine enforces
        the rules (ko, suicide, area scoring); the NN seat plays via the
        reification MCTS over the embedded net. <Health h={healths.go9} />
      </p>
      {healths.go9.status === "error" ? (
        <p style={{ color: "crimson" }}>
          Engine not reachable at {go9.engineUrl}. Start it with{" "}
          <code>go9-engine serve --model-local nn=&lt;checkpoint&gt;</code>.
        </p>
      ) : (
        <Go9Board client={go9Client} />
      )}

      <h2 style={{ marginTop: "2rem" }}>Engines</h2>
      <ul>
        {GAMES.map((g) => (
          <li key={g.id}>
            <strong>{g.label}</strong> @ {g.engineUrl} — <Health h={healths[g.id]} />
            {!g.playable && <span style={{ color: "#888" }}> (UI pending)</span>}
          </li>
        ))}
      </ul>
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
