"""Continuous Hex evaluator and gate, mirroring `c4_eval_loop.py`.

Scores each new learner snapshot against **MoHex** — an absolute yardstick, not
a chained self-play ELO that can drift upward against itself forever — and
promotes the best-scoring net to `live/best.safetensors` + `live/best.onnx`.

The gate is the point. Without one, every snapshot immediately becomes the
self-play generator, so a net that gets worse starts producing worse data and
the run walks downhill; that is exactly what the 2026-08-09 run did (generation
31 scored 0.31 against generation 1). AlphaGo Zero gated on a 55% win rate over
400 games for the same reason. Here the actor plays from `best.*`, so a snapshot
has to earn its way in.

To keep matches fast and comparable, the evaluator:
  * runs INFREQUENTLY — only the latest snapshot, once it is `--every`
    generations past the last one scored (intermediate ones are skipped);
  * PAUSES self-play by raising `live/PAUSE`, so the match isn't fighting the
    actor for the GPU (the actor drops `actor.paused` when parked);
  * records everything in `live/scores.json`.

MoHex must be built (scripts/fetch-mohex.sh) and `scripts/ort-env.sh` +
/tmp/cuda_env.sh sourced for the CUDA engine.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time


def snapshots(snaps: str) -> list[tuple[int, str]]:
    out = []
    for d in glob.glob(os.path.join(snaps, "gen*")):
        ck = os.path.join(d, "checkpoint.safetensors")
        name = os.path.basename(d)[3:]
        if os.path.exists(ck) and name.isdigit():
            out.append((int(name), ck))
    return sorted(out)


def request_pause(live: str, timeout: float) -> bool:
    """Ask the actor to park; True if it confirmed within `timeout`."""
    open(os.path.join(live, "PAUSE"), "w").close()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(os.path.join(live, "actor.paused")):
            return True
        time.sleep(1.0)
    return False


def release_pause(live: str) -> None:
    for f in ("PAUSE", "actor.paused"):
        try:
            os.remove(os.path.join(live, f))
        except FileNotFoundError:
            pass


def run_vs_mohex(engine: str, ckpt: str, games: int, sims: int, playouts: int,
                 mohex_bin: str, cuda: bool, seed: int) -> float:
    cmd = [engine, "vs-mohex", "--model", ckpt, "--games", str(games),
           "--sims", str(sims), "--mohex-playouts", str(playouts),
           "--mohex-bin", mohex_bin, "--seed", str(seed)]
    if cuda:
        cmd.append("--cuda")
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    m = re.search(r"net_vs_mohex\s+([\d.]+)", out.stdout)
    if not m:
        raise RuntimeError(f"no net_vs_mohex in output:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def export_onnx(ckpt: str, out_path: str, channels: int, blocks: int) -> None:
    """Re-export the promoted checkpoint so the ORT actor picks up the same net."""
    import torch

    from playground.model_server.hex_net import IN_PLANES, N, HexAZNet, load_checkpoint

    net = HexAZNet(channels=channels, blocks=blocks)
    load_checkpoint(net, ckpt)
    net.eval()
    tmp = out_path + ".tmp"
    torch.onnx.export(
        net, torch.zeros(1, IN_PLANES, N, N), tmp,
        input_names=["planes"], output_names=["value", "policy", "ownership"],
        dynamic_axes={k: {0: "b"} for k in ("planes", "value", "policy", "ownership")},
        opset_version=17, dynamo=False,
    )
    os.replace(tmp, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/models/hex_az/live")
    ap.add_argument("--engine", default="target/release/hex-engine")
    ap.add_argument("--mohex-bin", default="mohex")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--mohex-playouts", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--every", type=int, default=3, help="generations between evaluations")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--pause-timeout", type=float, default=90.0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--once", action="store_true", help="evaluate the latest snapshot and exit")
    args = ap.parse_args()

    live = args.live
    snaps = os.path.join(live, "snapshots")
    best_ckpt = os.path.join(live, "best.safetensors")
    best_onnx = os.path.join(live, "best.onnx")
    scores_path = os.path.join(live, "scores.json")
    os.makedirs(snaps, exist_ok=True)

    try:
        state = json.load(open(scores_path))
    except Exception:
        state = {"best_gen": -1, "best_score": -1.0, "processed_through": -1,
                 "scores": {}, "history": []}

    release_pause(live)  # never inherit a stale flag
    print(f"hex evaluator: best gen {state['best_gen']} ({state['best_score']:.3f}), "
          f"{args.games} games vs mohex@{args.mohex_playouts} every {args.every} gens",
          flush=True)

    while True:
        gens = snapshots(snaps)
        if gens and gens[-1][0] - state["processed_through"] >= args.every:
            gen, ck = gens[-1]
            parked = request_pause(live, args.pause_timeout)
            try:
                sc = run_vs_mohex(args.engine, ck, args.games, args.sims,
                                  args.mohex_playouts, args.mohex_bin, args.cuda, seed=gen)
            except Exception as e:
                release_pause(live)
                print(f"gen{gen}: eval failed: {e}", flush=True)
                if args.once:
                    return
                time.sleep(args.interval)
                continue
            release_pause(live)

            promoted = sc > state["best_score"]
            if promoted:
                shutil.copyfile(ck, best_ckpt)
                try:
                    export_onnx(ck, best_onnx, args.channels, args.blocks)
                except Exception as e:
                    print(f"gen{gen}: promoted but ONNX export failed: {e}", flush=True)
                state["best_gen"], state["best_score"] = gen, sc
            state["scores"][str(gen)] = sc
            state["history"].append({"gen": gen, "score": sc, "promoted": promoted})
            state["processed_through"] = gen
            tmp = scores_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, scores_path)

            tag = "PROMOTED" if promoted else f"best stays gen{state['best_gen']} ({state['best_score']:.3f})"
            pk = "" if parked else " [pause timeout: ran with contention]"
            print(f"gen{gen} vs mohex: {sc:.3f}  [{tag}]{pk}", flush=True)
            if args.once:
                return
        elif args.once:
            print("no new snapshot to evaluate", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
