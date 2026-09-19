import { useEffect, useState } from "react";
import { engineClientFor } from "./lib/client";
import { Connect4Board } from "./games/connect4/Board";
import { Go9Board } from "./games/go9/Board";
import { AmazonsBoard } from "./games/amazons/Board";
import { HexBoard } from "./games/hex/Board";
import { MinionsBoard } from "./games/minions/Board";
import { Recordings } from "./games/minions/Recordings";

// Engines are reached under same-origin proxy paths (see vite.config.ts), so the
// whole app works behind one origin without CORS.
const GAMES = [
  {
    id: "connect4", label: "Connect 4", engineUrl: "/c4", playable: true,
    blurb: "Play a human or the built-in perfect solver. The Rust engine enforces the rules and plays optimally for engine seats.",
    start: "cargo run --bin connect4-engine -- serve",
  },
  {
    id: "go9", label: "Go 9×9", engineUrl: "/go9", playable: true,
    blurb: "Play a human or the self-play net. The engine enforces ko / suicide / area scoring; the NN seat plays via the reification MCTS over the embedded net.",
    start: "go9-engine serve --model-local nn=<checkpoint>",
  },
  {
    id: "amazons", label: "Amazons", engineUrl: "/a", playable: true,
    blurb: "Play a human or the self-play net (gen71) on the standard 10×10 board. Pick a queen, its destination, then where to shoot the arrow; the NN seat plays via MCTS over the embedded net.",
    start: "amazons-engine serve --model-local nn=<checkpoint> --cuda",
  },
  {
    id: "hex", label: "Hex 13×13", engineUrl: "/hex", playable: true,
    blurb: "Black connects top↕bottom, White left↔right; first to link their two edges wins, and there are no draws. The engine seat plays the placeholder mover for now — there is no Hex net yet. Swap rule off.",
    start: "hex-engine serve --addr 127.0.0.1:7104",
  },
  {
    id: "minions", label: "Minions (simplified)", engineUrl: "/mi", playable: true,
    blurb: "One 7×7 hex board from Matir03/minions, with the first four techline units permanently unlocked and no generaling. Kill the enemy necromancer, hold 4 of the 6 graveyards at the end of their turn, or be ahead on net worth at the 20-turn cap. A turn is an attack phase then a spawn phase — the engine seat plays a greedy heuristic, not a net.",
    start: "minions-engine serve --addr 127.0.0.1:7105",
  },
  { id: "arimaa", label: "Arimaa", engineUrl: "/ar", playable: false, blurb: "", start: "" },
] as const;

type GameId = (typeof GAMES)[number]["id"];

type EngineHealth =
  | { status: "unknown" }
  | { status: "ok"; version: string }
  | { status: "error"; message: string };

// One client per playable engine, created once.
const clients: Partial<Record<GameId, ReturnType<typeof engineClientFor>>> = {
  connect4: engineClientFor("/c4"),
  go9: engineClientFor("/go9"),
  amazons: engineClientFor("/a"),
  hex: engineClientFor("/hex"),
  minions: engineClientFor("/mi"),
};

export function App() {
  const [selected, setSelected] = useState<GameId | null>(null);
  // Recorded matches are not an engine, so they are a sibling view rather than
  // a `GAMES` entry: everything in GAMES gets health-checked against an engine
  // URL and given a client, and a static recording needs neither.
  const [recordings, setRecordings] = useState(false);
  const [healths, setHealths] = useState<Record<string, EngineHealth>>(
    Object.fromEntries(GAMES.map((g) => [g.id, { status: "unknown" }])),
  );

  useEffect(() => {
    for (const g of GAMES) {
      (async () => {
        try {
          const resp = await engineClientFor(g.engineUrl).health({});
          setHealths((prev) => ({ ...prev, [g.id]: { status: "ok", version: resp.version } }));
        } catch (e) {
          setHealths((prev) => ({ ...prev, [g.id]: { status: "error", message: (e as Error).message } }));
        }
      })();
    }
  }, []);

  const game = GAMES.find((g) => g.id === selected) ?? null;

  return (
    // Wider while a game is open: a board with an analysis panel beside it
    // does not fit in 720px, and wrapping the panel underneath puts the board
    // off screen exactly when the panel is driving it.
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: game == null && !recordings ? 720 : 1400 }}>
      {recordings ? (
        <Recordings onBack={() => setRecordings(false)} />
      ) : game == null ? (
        <Menu healths={healths} onPick={setSelected} onRecordings={() => setRecordings(true)} />
      ) : (
        <GameView game={game} health={healths[game.id]} onBack={() => setSelected(null)} />
      )}
    </main>
  );
}

function Menu({
  healths,
  onPick,
  onRecordings,
}: {
  healths: Record<string, EngineHealth>;
  onPick: (id: GameId) => void;
  onRecordings: () => void;
}) {
  return (
    <>
      <h1 style={{ marginBottom: "0.25rem" }}>playground</h1>
      <p style={{ marginTop: 0, color: "#555" }}>Pick a game to play.</p>
      <div style={{ display: "grid", gap: "0.75rem", marginTop: "1.5rem" }}>
        {GAMES.map((g) => {
          const h = healths[g.id];
          const disabled = !g.playable;
          return (
            <button
              key={g.id}
              onClick={() => g.playable && onPick(g.id)}
              disabled={disabled}
              style={{
                textAlign: "left",
                padding: "1rem 1.25rem",
                borderRadius: 10,
                border: "1px solid #ddd",
                background: disabled ? "#f5f5f5" : "#fff",
                cursor: disabled ? "default" : "pointer",
                color: disabled ? "#999" : "inherit",
                font: "inherit",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <strong style={{ fontSize: "1.1rem" }}>{g.label}</strong>
                {g.playable ? <Health h={h} /> : <span style={{ color: "#888" }}>UI pending</span>}
              </div>
              {g.blurb && <div style={{ color: "#666", fontSize: "0.9rem", marginTop: 4 }}>{g.blurb}</div>}
            </button>
          );
        })}
      </div>
      <button
        onClick={onRecordings}
        style={{
          marginTop: "1.5rem", textAlign: "left", padding: "1rem 1.25rem", borderRadius: 10,
          border: "1px solid #ddd", background: "#fff", cursor: "pointer", font: "inherit", width: "100%",
        }}
      >
        <strong style={{ fontSize: "1.1rem" }}>Recorded matches</strong>
        <div style={{ color: "#666", fontSize: "0.9rem", marginTop: 4 }}>
          Replay games from an <code>eval-match</code> run, turn by turn. Served from static
          JSON, so no engine has to be running.
        </div>
      </button>
    </>
  );
}

function GameView({
  game,
  health,
  onBack,
}: {
  game: (typeof GAMES)[number];
  health: EngineHealth;
  onBack: () => void;
}) {
  const client = clients[game.id];
  return (
    <>
      <button
        onClick={onBack}
        style={{ marginBottom: "1rem", background: "none", border: "none", color: "#1d4ed8", cursor: "pointer", font: "inherit", padding: 0 }}
      >
        ← Menu
      </button>
      <h1 style={{ marginTop: 0, marginBottom: "0.5rem" }}>{game.label}</h1>
      <p style={{ marginTop: 0, color: "#555", maxWidth: 720 }}>
        {game.blurb} <Health h={health} />
      </p>
      {health.status === "error" || !client ? (
        <p style={{ color: "crimson" }}>
          Engine not reachable at {game.engineUrl}. Start it with <code>{game.start}</code>.
        </p>
      ) : game.id === "connect4" ? (
        <Connect4Board client={client} />
      ) : game.id === "amazons" ? (
        <AmazonsBoard client={client} />
      ) : game.id === "hex" ? (
        <HexBoard client={client} />
      ) : game.id === "minions" ? (
        <MinionsBoard client={client} />
      ) : (
        <Go9Board client={client} />
      )}
    </>
  );
}

function Health({ h }: { h: EngineHealth }) {
  switch (h.status) {
    case "unknown": return <span style={{ color: "#888" }}>checking…</span>;
    case "ok": return <span style={{ color: "green" }}>ok (v{h.version})</span>;
    case "error": return <span style={{ color: "crimson" }}>down</span>;
  }
}
