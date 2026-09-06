import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { engineClientFor } from "./lib/client";
import { Connect4Board } from "./games/connect4/Board";
import { Go9Board } from "./games/go9/Board";
import { AmazonsBoard } from "./games/amazons/Board";
import { HexBoard } from "./games/hex/Board";
import { MinionsBoard } from "./games/minions/Board";
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
];
// One client per playable engine, created once.
const clients = {
    connect4: engineClientFor("/c4"),
    go9: engineClientFor("/go9"),
    amazons: engineClientFor("/a"),
    hex: engineClientFor("/hex"),
    minions: engineClientFor("/mi"),
};
export function App() {
    const [selected, setSelected] = useState(null);
    const [healths, setHealths] = useState(Object.fromEntries(GAMES.map((g) => [g.id, { status: "unknown" }])));
    useEffect(() => {
        for (const g of GAMES) {
            (async () => {
                try {
                    const resp = await engineClientFor(g.engineUrl).health({});
                    setHealths((prev) => ({ ...prev, [g.id]: { status: "ok", version: resp.version } }));
                }
                catch (e) {
                    setHealths((prev) => ({ ...prev, [g.id]: { status: "error", message: e.message } }));
                }
            })();
        }
    }, []);
    const game = GAMES.find((g) => g.id === selected) ?? null;
    return (_jsx("main", { style: { fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 720 }, children: game == null ? (_jsx(Menu, { healths: healths, onPick: setSelected })) : (_jsx(GameView, { game: game, health: healths[game.id], onBack: () => setSelected(null) })) }));
}
function Menu({ healths, onPick, }) {
    return (_jsxs(_Fragment, { children: [_jsx("h1", { style: { marginBottom: "0.25rem" }, children: "playground" }), _jsx("p", { style: { marginTop: 0, color: "#555" }, children: "Pick a game to play." }), _jsx("div", { style: { display: "grid", gap: "0.75rem", marginTop: "1.5rem" }, children: GAMES.map((g) => {
                    const h = healths[g.id];
                    const disabled = !g.playable;
                    return (_jsxs("button", { onClick: () => g.playable && onPick(g.id), disabled: disabled, style: {
                            textAlign: "left",
                            padding: "1rem 1.25rem",
                            borderRadius: 10,
                            border: "1px solid #ddd",
                            background: disabled ? "#f5f5f5" : "#fff",
                            cursor: disabled ? "default" : "pointer",
                            color: disabled ? "#999" : "inherit",
                            font: "inherit",
                        }, children: [_jsxs("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "baseline" }, children: [_jsx("strong", { style: { fontSize: "1.1rem" }, children: g.label }), g.playable ? _jsx(Health, { h: h }) : _jsx("span", { style: { color: "#888" }, children: "UI pending" })] }), g.blurb && _jsx("div", { style: { color: "#666", fontSize: "0.9rem", marginTop: 4 }, children: g.blurb })] }, g.id));
                }) })] }));
}
function GameView({ game, health, onBack, }) {
    const client = clients[game.id];
    return (_jsxs(_Fragment, { children: [_jsx("button", { onClick: onBack, style: { marginBottom: "1rem", background: "none", border: "none", color: "#1d4ed8", cursor: "pointer", font: "inherit", padding: 0 }, children: "\u2190 Menu" }), _jsx("h1", { style: { marginTop: 0, marginBottom: "0.5rem" }, children: game.label }), _jsxs("p", { style: { marginTop: 0, color: "#555" }, children: [game.blurb, " ", _jsx(Health, { h: health })] }), health.status === "error" || !client ? (_jsxs("p", { style: { color: "crimson" }, children: ["Engine not reachable at ", game.engineUrl, ". Start it with ", _jsx("code", { children: game.start }), "."] })) : game.id === "connect4" ? (_jsx(Connect4Board, { client: client })) : game.id === "amazons" ? (_jsx(AmazonsBoard, { client: client })) : game.id === "hex" ? (_jsx(HexBoard, { client: client })) : game.id === "minions" ? (_jsx(MinionsBoard, { client: client })) : (_jsx(Go9Board, { client: client }))] }));
}
function Health({ h }) {
    switch (h.status) {
        case "unknown": return _jsx("span", { style: { color: "#888" }, children: "checking\u2026" });
        case "ok": return _jsxs("span", { style: { color: "green" }, children: ["ok (v", h.version, ")"] });
        case "error": return _jsx("span", { style: { color: "crimson" }, children: "down" });
    }
}
