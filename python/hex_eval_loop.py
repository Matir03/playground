"""Continuous Hex evaluator and gate, mirroring `c4_eval_loop.py`.

Promotes a new learner snapshot to `live/best.safetensors` + `live/best.onnx`
when it beats the **current best net** head to head (AlphaGo Zero's 55% gate),
and separately reports its score against **MoHex** as an absolute yardstick.

Those are deliberately two different things. Gating on MoHex directly sounds
tidier but has no resolution while the net is weak: a young net scores 0 of 8,
the incumbent scored 0 of 8, nothing ever promotes, and the actor freezes on
whatever it started with — worse than no gate. Beating the incumbent always
discriminates; MoHex tells you whether the ladder is climbing anything real.

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


def run_match(engine: str, cand: str, best: str, games: int, node_budget: int,
              threads: int, cuda: bool, seed: int, ort: bool = False) -> float:
    """Candidate vs incumbent; returns the candidate's score in [0,1].

    On the ORT/TensorRT backend a 40-game gate takes a couple of minutes rather
    than the ~40 candle needs, which matters when the gate is a phase of every
    third training cycle."""
    cmd = [engine, "eval-match", "--a", cand, "--b", best, "--games", str(games),
           "--node-budget", str(node_budget), "--threads", str(threads),
           "--max-batch", str(max(8, games)), "--seed", str(seed)]
    if ort:
        cmd += ["--backend", "ort", "--ep", "trt"]
    if cuda:
        cmd.append("--cuda")
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Log the match's own summary line (games played and elapsed). Without it the
    # only record of a gate is its score, so there is no way to tell a real
    # 40-game match from one that ended early -- the durations vary 150s..1250s
    # and reconstructing why meant arithmetic rather than reading a log.
    m = re.search(r"A_score\s+([\d.]+)\s*(\([^)]*\))?", out.stdout)
    if not m:
        raise RuntimeError(f"no A_score in output:\n{out.stdout}\n{out.stderr}")
    print(f"    gate: A_score {m.group(1)} {m.group(2) or ''}".rstrip(), flush=True)
    return float(m.group(1))


def run_vs_mohex(engine: str, ckpt: str, games: int, sims: int, playouts: int,
                 mohex_bin: str, cuda: bool, seed: int) -> tuple[float, int]:
    cmd = [engine, "vs-mohex", "--model", ckpt, "--games", str(games),
           "--sims", str(sims), "--mohex-playouts", str(playouts),
           "--mohex-bin", mohex_bin, "--seed", str(seed)]
    if cuda:
        cmd.append("--cuda")
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # MoHex segfaults on some positions; the engine retries and voids what it
    # cannot finish, so the score's denominator is the games that completed, not
    # the games requested. Keep both -- 0.000 over 2 games and over 8 are very
    # different readings. The per-game detail is on stderr, which is otherwise
    # discarded.
    for line in out.stderr.splitlines():
        if "voided" in line or "mohex died" in line:
            print(f"    {line.strip()}", flush=True)
    m = re.search(r"net_vs_mohex\s+([\d.]+)\s+\((\d+) games", out.stdout)
    if not m:
        raise RuntimeError(f"no net_vs_mohex in output:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1)), int(m.group(2))


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
    ap.add_argument("--games", type=int, default=24,
                    help="games in the candidate-vs-champion gate match")
    ap.add_argument("--promote-threshold", type=float, default=0.55,
                    help="AGZ used 55%% over 400 games; with fewer games this is noisier")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mohex-games", type=int, default=8,
                    help="games against MoHex, run only when a net promotes")
    ap.add_argument("--sims", type=int, default=400)
    # 100, not 1000: at 1000 playouts MoHex beats a young net every time and the
    # column reads 0.000 forever, which is no measurement at all. A weaker
    # yardstick gives a graded reading years earlier in the run. Scores are only
    # comparable within one setting — note it when you change it.
    ap.add_argument("--mohex-playouts", type=int, default=100)
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
    print(f"hex evaluator: champion gen {state['best_gen']} (vs mohex {state['best_score']:.3f}); "
          f"gate = {args.games} games vs champion at >={args.promote_threshold:.0%}, "
          f"every {args.every} gens", flush=True)

    while True:
        gens = snapshots(snaps)
        if gens and gens[-1][0] - state["processed_through"] >= args.every:
            gen, ck = gens[-1]
            parked = request_pause(live, args.pause_timeout)
            try:
                if os.path.exists(best_ckpt):
                    # Match on ORT when both sides have an ONNX: export the
                    # candidate (a second) to save ~35 minutes of match time.
                    cand_onnx = os.path.join(live, "candidate.onnx")
                    use_ort = os.path.exists(best_onnx)
                    if use_ort:
                        try:
                            export_onnx(ck, cand_onnx, args.channels, args.blocks)
                        except Exception as e:
                            print(f"gen{gen}: candidate ONNX export failed, using candle: {e}",
                                  flush=True)
                            use_ort = False
                    vs_best = run_match(
                        args.engine,
                        cand_onnx if use_ort else ck,
                        best_onnx if use_ort else best_ckpt,
                        args.games, args.sims, args.threads, args.cuda,
                        seed=gen, ort=use_ort,
                    )
                    promoted = vs_best >= args.promote_threshold
                else:
                    vs_best = 1.0  # nothing to beat yet
                    promoted = True
                # MoHex is the absolute yardstick, and it is expensive, so ask it
                # only about nets that just earned the seat. Its own failure must
                # not cost the promotion: a missing binary once discarded a net
                # that had already beaten the champion.
                mohex = None
                mohex_played = 0
                if promoted:
                    try:
                        mohex, mohex_played = run_vs_mohex(
                            args.engine, ck, args.mohex_games, args.sims,
                            args.mohex_playouts, args.mohex_bin, args.cuda, seed=gen)
                    except Exception as e:
                        print(f"gen{gen}: promoted; mohex reading unavailable ({e})", flush=True)
            except Exception as e:
                release_pause(live)
                print(f"gen{gen}: eval failed: {e}", flush=True)
                if args.once:
                    return
                time.sleep(args.interval)
                continue
            release_pause(live)

            if promoted:
                shutil.copyfile(ck, best_ckpt)
                try:
                    export_onnx(ck, best_onnx, args.channels, args.blocks)
                except Exception as e:
                    print(f"gen{gen}: promoted but ONNX export failed: {e}", flush=True)
                state["best_gen"] = gen
                if mohex is not None:
                    state["best_score"] = mohex
            # Carry the learner's losses for this generation into the record, so
            # strength and what the net was fitting sit side by side.
            try:
                metrics = json.load(open(os.path.join(os.path.dirname(ck), "metrics.json")))
            except Exception:
                metrics = {}
            entry = {
                "gen": gen, "vs_best": vs_best, "vs_mohex": mohex, "promoted": promoted,
                "mohex_playouts": args.mohex_playouts,
                "mohex_games": mohex_played,  # completed, not requested
                "loss_value": metrics.get("loss_value"),
                "loss_policy": metrics.get("loss_policy"),
                "loss_ownership": metrics.get("loss_ownership"),
            }
            state["scores"][str(gen)] = entry
            state["history"].append(entry)
            state["processed_through"] = gen
            tmp = scores_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, scores_path)

            tag = "PROMOTED" if promoted else f"champion stays gen{state['best_gen']}"
            mh = (f", vs mohex@{args.mohex_playouts} {mohex:.3f}"
                  f" ({mohex_played}/{args.mohex_games} games)") if mohex is not None else ""
            ls = ""
            if metrics:
                ls = (f"  [loss v {metrics['loss_value']:.4f}"
                      f" p {metrics['loss_policy']:.4f}"
                      f" o {metrics['loss_ownership']:.4f}]")
            print(f"gen{gen} vs champion: {vs_best:.3f}{mh}  [{tag}]{ls}", flush=True)
            if args.once:
                return
        elif args.once:
            print("no new snapshot to evaluate", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
