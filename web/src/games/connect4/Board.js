import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Connect 4 play surface. Each seat is a human or the built-in perfect engine;
 * the Rust engine owns the rules (legality, win detection) and, for engine
 * seats, picks the move via the perfect solver (EngineService.RequestMove).
 *
 * Wire format (must match crates/games/connect4/src/game.rs):
 *   GameState.payload = 42 cell bytes (row-major, row 0 = bottom) + 2 ply bytes.
 *   cell value: 0 empty, 1 first (yellow), 2 second (red).
 *   Move.payload   = 1 byte: the column index.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { create } from "@bufbuild/protobuf";
import { ConnectError } from "@connectrpc/connect";
import { GameId, MoveSchema, Outcome, Player, } from "../../gen/playground/game/game_pb";
import { EnginePlayer, NewGameRequestSchema, ApplyMoveRequestSchema, RequestMoveRequestSchema, SearchBudgetSchema, } from "../../gen/playground/engine/engine_pb";
const ROWS = 6;
const COLS = 7;
const SEAT_LABELS = {
    human: "Human",
    solver: "Perfect solver",
    "nn-random": "NN — random",
    "nn-trained": "NN — trained (cloned solver)",
    "nn-selfplay": "NN — self-play",
};
function seatToPlayer(seat) {
    return seat.startsWith("nn") ? EnginePlayer.NN : EnginePlayer.SOLVER;
}
// Engine-side model name for NN seats (see connect4-engine --model NAME=URI).
function seatToModel(seat) {
    if (seat === "nn-random")
        return "random";
    if (seat === "nn-trained")
        return "trained";
    if (seat === "nn-selfplay")
        return "selfplay";
    return "";
}
/** Decode the engine's opaque payload into a [row][col] grid (row 0 = bottom). */
function decodeBoard(payload) {
    const grid = Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => 0));
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            grid[r][c] = (payload[r * COLS + c] ?? 0);
        }
    }
    return grid;
}
function outcomeLabel(o) {
    switch (o) {
        case Outcome.FIRST_WINS:
            return "Yellow wins";
        case Outcome.SECOND_WINS:
            return "Red wins";
        case Outcome.DRAW:
            return "Draw";
        default:
            return "";
    }
}
export function Connect4Board({ client, }) {
    const [handle, setHandle] = useState(null);
    const [state, setState] = useState(null);
    const [firstSeat, setFirstSeat] = useState("human");
    const [secondSeat, setSecondSeat] = useState("solver");
    const [thinking, setThinking] = useState(false);
    // NN search budget (ignored by the solver). Default: 1 second per move.
    const [limitMode, setLimitMode] = useState("seconds");
    const [seconds, setSeconds] = useState(1);
    const [visits, setVisits] = useState(800);
    const [lastStats, setLastStats] = useState(null);
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);
    // Guards the engine's auto-move so it fires exactly once per position, even
    // under React StrictMode's double-invoked effects (refs persist across them).
    const engineTurnRef = useRef(null);
    const newGame = useCallback(async () => {
        setError(null);
        setBusy(true);
        setLastStats(null);
        try {
            const h = await client.newGame(create(NewGameRequestSchema, { gameId: GameId.CONNECT4 }));
            setHandle(h);
            setState(await client.getState(h));
        }
        catch (e) {
            setError(e instanceof ConnectError ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }, [client]);
    useEffect(() => {
        void newGame();
    }, [newGame]);
    const seatToMove = (s) => s.toMove === Player.FIRST ? firstSeat : secondSeat;
    const drop = useCallback(async (col) => {
        if (!handle || !state || busy || thinking)
            return;
        if (state.outcome !== Outcome.IN_PROGRESS)
            return;
        if (seatToMove(state) !== "human")
            return;
        setError(null);
        setBusy(true);
        try {
            const next = await client.applyMove(create(ApplyMoveRequestSchema, {
                handle,
                move: create(MoveSchema, {
                    gameId: GameId.CONNECT4,
                    payload: new Uint8Array([col]),
                }),
            }));
            setState(next);
        }
        catch (e) {
            setError(e instanceof ConnectError ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }, 
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [client, handle, state, busy, thinking, firstSeat, secondSeat]);
    // Engine auto-move: when it's an engine seat's turn, ask the server to think.
    useEffect(() => {
        if (!handle || !state)
            return;
        if (state.outcome !== Outcome.IN_PROGRESS)
            return;
        const seat = seatToMove(state);
        if (seat === "human")
            return;
        const player = seatToPlayer(seat);
        const model = seatToModel(seat);
        const tag = `${handle.handleId}:${state.ply}`;
        if (engineTurnRef.current === tag)
            return;
        engineTurnRef.current = tag;
        const fromPly = state.ply;
        (async () => {
            setThinking(true);
            setError(null);
            // A brief pace before moving: makes engine-vs-engine watchable, gives
            // human-vs-engine a natural feel, and spaces out gRPC-Web calls so a
            // dropped-trailer hiccup can't cascade on a reused HTTP/1.1 connection.
            await new Promise((r) => setTimeout(r, 250));
            // Browser gRPC-Web over HTTP/1.1 occasionally drops a response trailer
            // ("missing trailer") on rapid calls; the underlying RPC still works, so
            // retry, reconciling against server state to avoid double-applying.
            for (let attempt = 0; attempt < 4; attempt++) {
                try {
                    let chosen;
                    try {
                        // Timeout guards against a browser gRPC-Web stream that hangs
                        // waiting for a dropped trailer (the solver itself is ~instant).
                        for await (const prog of client.requestMove(create(RequestMoveRequestSchema, {
                            handle,
                            player,
                            model,
                            budget: create(SearchBudgetSchema, limitMode === "seconds"
                                ? { maxMillis: Math.max(1, Math.round(seconds * 1000)) }
                                : { maxVisits: Math.max(1, Math.round(visits)) }),
                        }), { signal: AbortSignal.timeout(8000) })) {
                            if (prog.kind.case === "stats")
                                setLastStats(prog.kind.value);
                            else if (prog.kind.case === "chosen")
                                chosen = prog.kind.value;
                        }
                    }
                    catch (streamErr) {
                        if (!chosen)
                            throw streamErr; // only fatal if no move arrived
                    }
                    if (!chosen)
                        throw new Error("engine returned no move");
                    const next = await client.applyMove(create(ApplyMoveRequestSchema, { handle, move: chosen }), { signal: AbortSignal.timeout(8000) });
                    setState(next);
                    break;
                }
                catch (e) {
                    // Maybe the move applied server-side despite a client-side parse error.
                    const cur = await client
                        .getState(handle, { signal: AbortSignal.timeout(8000) })
                        .catch(() => null);
                    if (cur && cur.ply > fromPly) {
                        setState(cur);
                        break;
                    }
                    if (attempt === 3) {
                        setError(e instanceof ConnectError ? e.message : String(e));
                    }
                }
            }
            setThinking(false);
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [handle, state, firstSeat, secondSeat, client]);
    const board = state ? decodeBoard(state.payload) : null;
    const inProgress = state?.outcome === Outcome.IN_PROGRESS;
    const humanTurn = !!state && inProgress && seatToMove(state) === "human";
    const toMove = state?.toMove === Player.FIRST ? "Yellow" : "Red";
    const colFull = (c) => !!board && board[ROWS - 1][c] !== 0;
    return (_jsxs("section", { children: [_jsxs("div", { style: { display: "flex", gap: "1.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }, children: [_jsx(SeatPicker, { label: "Yellow (first)", seat: firstSeat, onChange: setFirstSeat }), _jsx(SeatPicker, { label: "Red (second)", seat: secondSeat, onChange: setSecondSeat }), _jsxs("label", { style: { display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }, children: ["NN search limit", _jsxs("span", { style: { display: "flex", gap: 4 }, children: [_jsx("input", { type: "number", min: limitMode === "seconds" ? 0.1 : 1, step: limitMode === "seconds" ? 0.1 : 50, value: limitMode === "seconds" ? seconds : visits, onChange: (e) => {
                                            const v = Number(e.target.value);
                                            if (limitMode === "seconds")
                                                setSeconds(v);
                                            else
                                                setVisits(v);
                                        }, style: { width: 70 } }), _jsxs("select", { value: limitMode, onChange: (e) => setLimitMode(e.target.value), children: [_jsx("option", { value: "seconds", children: "seconds" }), _jsx("option", { value: "visits", children: "visits" })] })] })] }), _jsx("button", { onClick: () => void newGame(), disabled: busy, children: "New game" })] }), _jsxs("div", { style: { marginBottom: "0.5rem", minHeight: "1.5em" }, children: [_jsx("strong", { children: state == null
                            ? "Loading…"
                            : inProgress
                                ? thinking
                                    ? `${toMove} (engine) thinking…`
                                    : `${toMove} to move`
                                : outcomeLabel(state.outcome) }), lastStats && (_jsxs("span", { style: { color: "#888", marginLeft: "0.75rem" }, children: ["last search: ", lastStats.visits.toLocaleString(), " visits", lastStats.reifications > 0 && `, ${lastStats.reifications} reified`, ", eval ", lastStats.bestValue.toFixed(2)] }))] }), _jsxs("div", { style: { display: "inline-block", background: "#1d4ed8", padding: 8, borderRadius: 8 }, children: [board &&
                        Array.from({ length: ROWS }, (_, i) => ROWS - 1 - i).map((r) => (_jsx("div", { style: { display: "flex" }, children: Array.from({ length: COLS }, (_, c) => (_jsx(Disc, { v: board[r][c] }, c))) }, r))), _jsx("div", { style: { display: "flex" }, children: Array.from({ length: COLS }, (_, c) => (_jsx("button", { onClick: () => void drop(c), disabled: !humanTurn || busy || colFull(c), title: `Drop in column ${c + 1}`, style: {
                                width: 44,
                                height: 28,
                                margin: 2,
                                cursor: humanTurn && !colFull(c) ? "pointer" : "default",
                            }, children: "\u2193" }, c))) })] }), error && _jsxs("p", { style: { color: "crimson", marginTop: "0.75rem" }, children: ["Error: ", error] })] }));
}
function SeatPicker({ label, seat, onChange, }) {
    return (_jsxs("label", { style: { display: "flex", flexDirection: "column", fontSize: "0.85rem", color: "#555" }, children: [label, _jsx("select", { value: seat, onChange: (e) => onChange(e.target.value), children: Object.keys(SEAT_LABELS).map((s) => (_jsx("option", { value: s, children: SEAT_LABELS[s] }, s))) })] }));
}
function Disc({ v }) {
    const color = v === 1 ? "#facc15" : v === 2 ? "#ef4444" : "#0b2a8a";
    return (_jsx("div", { style: {
            width: 44,
            height: 44,
            margin: 2,
            borderRadius: "50%",
            background: color,
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.35)",
        } }));
}
