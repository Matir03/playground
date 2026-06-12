"""Play our trained Go9 net against GNU Go to estimate its strength.

Our net moves via the engine's gRPC RequestMove (the reification MCTS over the
embedded candle net); GNU Go moves via GTP. Our engine is the authoritative
position tracker (it generates our moves, applies both sides', and detects the
two-pass end); GNU Go does the final scoring (it handles dead stones, which our
naive area scoring doesn't). Coordinates use a fixed bijection between our (r,c)
and GTP vertices — Go is symmetric, so any consistent mapping keeps both engines
in sync.

  python go9_vs_gnugo.py --games 6 --level 1 --seconds 1.0
"""

from __future__ import annotations

import argparse
import subprocess
import time

import grpc

from playground.gen.playground.engine import engine_pb2 as e
from playground.gen.playground.engine import engine_pb2_grpc as eg
from playground.gen.playground.game import game_pb2 as g

COLS = "ABCDEFGHJ"  # GTP skips 'I'
N = 9


def vertex_from_rc(r: int, c: int) -> str:
    return f"{COLS[c]}{r + 1}"


def rc_from_vertex(v: str) -> tuple[int, int]:
    c = COLS.index(v[0].upper())
    r = int(v[1:]) - 1
    return r, c


class GTP:
    """Minimal GTP client over a subprocess (single-line responses)."""

    def __init__(self, cmd: list[str]) -> None:
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def send(self, cmd: str) -> str:
        assert self.p.stdin and self.p.stdout
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        out: list[str] = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                break
            if line.strip() == "":
                if out:
                    break
                continue
            out.append(line.rstrip("\n"))
        text = "\n".join(out)
        # Strip the leading "=" / "?" and any id.
        return text[1:].strip() if text else ""

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        self.p.terminate()


def net_move(stub, handle, seconds: float):
    """Ask the engine's NN for a move; returns the Move proto."""
    chosen = None
    for prog in stub.RequestMove(
        e.RequestMoveRequest(
            handle=handle, player=e.ENGINE_PLAYER_NN, model="nn",
            budget=e.SearchBudget(max_millis=int(seconds * 1000)),
        )
    ):
        if prog.HasField("chosen"):
            chosen = prog.chosen
    if chosen is None:
        raise RuntimeError("engine returned no move")
    return chosen


def move_to_vertex(payload: bytes) -> str:
    if payload[0] == 0:
        return "pass"
    return vertex_from_rc(payload[1], payload[2])


def vertex_to_move(v: str):
    if v.lower() == "pass":
        return g.Move(game_id=g.GAME_ID_GO9, payload=bytes([0]))
    r, c = rc_from_vertex(v)
    return g.Move(game_id=g.GAME_ID_GO9, payload=bytes([1, r, c]))


def play_game(stub, gnugo: GTP, net_is_black: bool, seconds: float, cap: int = 200) -> str:
    """Play one game. Returns 'net' or 'gnugo' (winner)."""
    handle = stub.NewGame(e.NewGameRequest(game_id=g.GAME_ID_GO9))
    gnugo.send("clear_board")
    gnugo.send("boardsize 9")
    gnugo.send("komi 7.5")

    ply = 0
    while ply < cap:
        state = stub.GetState(handle)
        if state.outcome != g.OUTCOME_IN_PROGRESS:
            break
        black_to_move = state.to_move == g.PLAYER_FIRST
        color = "black" if black_to_move else "white"
        net_turn = black_to_move == net_is_black

        if net_turn:
            mv = net_move(stub, handle, seconds)
            stub.ApplyMove(e.ApplyMoveRequest(handle=handle, move=mv))
            gnugo.send(f"play {color} {move_to_vertex(mv.payload)}")
        else:
            v = gnugo.send(f"genmove {color}")
            if v.lower() == "resign":
                return "net", "gnugo-resign"
            stub.ApplyMove(e.ApplyMoveRequest(handle=handle, move=vertex_to_move(v)))
        ply += 1

    # Score with GNU Go (it removes dead stones; our area scoring doesn't).
    score = gnugo.send("final_score")  # e.g. "B+3.5", "W+12.5", "0" (jigo)
    if score.startswith("B"):
        black_won = True
    elif score.startswith("W"):
        black_won = False
    else:
        # fall back to our engine's outcome
        st = stub.GetState(handle)
        black_won = st.outcome == g.OUTCOME_FIRST_WINS
    net_won = black_won == net_is_black
    return "net" if net_won else "gnugo", score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="127.0.0.1:7101")
    ap.add_argument("--gnugo", default="gnugo")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--level", type=int, default=1, help="GNU Go strength 1..10")
    ap.add_argument("--seconds", type=float, default=1.0, help="net seconds/move")
    args = ap.parse_args()

    stub = eg.EngineServiceStub(grpc.insecure_channel(args.engine))
    print("engine:", stub.Health(e.HealthRequest()).version,
          "| gnugo level", args.level, "| net", args.seconds, "s/move", flush=True)

    net_wins = 0
    for gm in range(args.games):
        gnugo = GTP([args.gnugo, "--mode", "gtp", "--chinese-rules", "--level", str(args.level)])
        net_black = gm % 2 == 0
        t0 = time.time()
        winner, score = play_game(stub, gnugo, net_black, args.seconds)
        gnugo.close()
        net_wins += winner == "net"
        print(f"  game {gm + 1}: net={'B' if net_black else 'W'}  "
              f"winner={winner}  gnugo_score={score}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nnet vs GNU Go (level {args.level}): {net_wins}/{args.games} "
          f"({100 * net_wins / args.games:.0f}%)")


if __name__ == "__main__":
    main()
