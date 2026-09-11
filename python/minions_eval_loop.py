"""Promotion gate for the Minions ladder.

Two measurements, deliberately separate (the Hex run's structure):

* **relative** — the newest snapshot against the reigning champion. Promote only
  above ``--promote-threshold``. The actor resolves its weights to ``best.*``
  when that file exists, so this gate is also what keeps a losing net's games
  out of the replay buffer.
* **absolute** — against the greedy heuristic in ``solver.rs``, the fixed
  yardstick that does not drift as the ladder climbs. Only run for nets that
  promoted, because it costs a full match and tells us nothing about a net that
  already lost; and its failure must never void a promotion.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import importlib.util as _ilu
import subprocess
import time

from playground.model_server.minions_net import build_model, export_onnx_atomic


def promote_onnx(ckpt: str, live: str) -> None:
    """Re-export the champion's graphs into ``<live>/best/``.

    The ORT actor resolves its weights to ``best/trunk.onnx`` when that exists,
    exactly as the candle actor prefers ``best.safetensors`` — so this is what
    stops self-play data coming from a net that lost its match.
    """
    cfg = json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
    net = build_model(cfg["arch"])
    from safetensors.torch import load_file
    net.load_state_dict(load_file(ckpt))
    best_dir = os.path.join(live, "best")
    export_onnx_atomic(net, best_dir)
    shutil.copyfile(os.path.join(os.path.dirname(ckpt), "config.json"),
                    os.path.join(best_dir, "config.json"))


_LAST_BIDS: tuple[float, float] | None = None
_LAST_TURNS: tuple[float, float] | None = None


def ensure_onnx(ckpt: str) -> None:
    """Export a snapshot's graphs next to its checkpoint, if they aren't there.

    The learner writes snapshots as ``checkpoint.safetensors`` + ``config.json``
    only, but an ORT seat loads ``trunk.onnx``/``diffusion.onnx``/``energy.onnx``
    from the checkpoint's directory. Without this, every eval against a snapshot
    dies on a missing trunk.onnx — which is how the overnight run lost its
    cycle-3 eval.
    """
    d = os.path.dirname(ckpt)
    if os.path.exists(os.path.join(d, "trunk.onnx")):
        return
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(d, "config.json")))
    net = build_model(cfg["arch"])
    net.load_state_dict(load_file(ckpt))
    export_onnx_atomic(net, d)


def run_match(engine, a, b, games, sims, candidates, t_steps, backend, ep, fp16, seed,
              render_file=None, render_games=5):
    if backend == "ort":
        ensure_onnx(a)
        if b:
            ensure_onnx(b)
    cmd = [engine, "eval-match", "--a", a, "--games", str(games), "--sims", str(sims),
           "--candidates", str(candidates), "--t-steps", str(t_steps), "--seed", str(seed),
           "--backend", backend, "--ep", ep]
    if b:
        cmd += ["--b", b]
    if fp16:
        cmd += ["--fp16"]
    # Rendered straight out of the scored match: both nets are already loaded,
    # so the sample costs nothing and is guaranteed to be the same games the
    # promotion number came from.
    if render_file:
        cmd += ["--render-file", render_file, "--render-games", str(render_games)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout
    # The bidding phase reports both seats' bids on stderr. Worth recording: a
    # bid pinned to the edge of the range means the value head cannot price a
    # handicap at all, and watching it move inward is how you see that change.
    bids = re.search(r"bids: A (-?[0-9.]+)\s+B (-?[0-9.]+)", proc.stderr or "")
    global _LAST_BIDS, _LAST_TURNS
    _LAST_BIDS = (float(bids.group(1)), float(bids.group(2))) if bids else None
    # Mean game length split by verdict, from A's seat. A net that is really
    # pulling ahead should win faster than it loses; lengths that stay equal say
    # the score is decided at the turn cap rather than over the board.
    t = re.search(r"turns: win (-?[0-9.]+)\s+loss (-?[0-9.]+)", proc.stderr or "")
    _LAST_TURNS = (float(t.group(1)), float(t.group(2))) if t else None
    m = re.search(r"A_score\s+([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"could not parse A_score from:\n{out}")
    mg = re.search(r"A_margin\s+(-?[0-9.]+)", out)
    margin = float(mg.group(1)) if mg else None
    # Mean terminal score: margin plus a speed bonus for finishing early.
    # Anchored to its own key rather than `A_score`, which is the win rate.
    gs = re.search(r"A_gamescore\s+(-?[0-9.]+)", out)
    game_score = float(gs.group(1)) if gs else None
    return float(m.group(1)), margin, game_score


_ELO = None


def elo_mod():
    """Load `minions_elo.py` beside this file, lazily.

    A sibling script rather than a package, so a plain import will not find it
    when the cycle runs this from the repo root.
    """
    global _ELO
    if _ELO is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minions_elo.py")
        spec = _ilu.spec_from_file_location("minions_elo", path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ELO = mod
    return _ELO


def past_champions(scores, exclude, n):
    """The `n` most recent promoted generations, newest first, minus `exclude`.

    `exclude` is the champion the candidate just played 40 games against -- that
    match is already logged and far better resolved than 6 games would be, so
    spending games on it again buys nothing.
    """
    promoted = [r["gen"] for r in scores.get("history", []) if r.get("promoted")]
    out = []
    for g in reversed(promoted):
        if g == exclude or g in out:
            continue
        out.append(g)
        if len(out) >= n:
            break
    return out


def run_ladder_matches(args, live, scores, gen, exclude):
    """Play the new champion against several earlier ones.

    The promotion gate only ever compares adjacent rungs, which measures each
    step but never the ladder: a run can promote thirty times on 0.55 margins
    and still be walking sideways if each net is merely different from the last
    rather than better. Matches against older champions tie distant rungs
    together, and BT turns that into one consistent rating scale.

    Six games each is deliberately coarse -- far too few to read on their own,
    which is the point of fitting them jointly rather than reporting them.
    """
    rows = []
    opponents = past_champions(scores, exclude, args.ladder_opponents)
    for i, opp in enumerate(opponents):
        ckpt = os.path.join(live, "snapshots", f"gen{opp}", "checkpoint.safetensors")
        if not os.path.exists(ckpt):
            # Snapshots are pruned eventually; a missing rung is normal.
            continue
        try:
            sc, _, _ = run_match(
                args.engine, os.path.join(live, "best.safetensors"), ckpt,
                args.ladder_games, args.sims, args.candidates, args.t_steps,
                args.backend, args.ep, args.fp16,
                # Distinct per opponent and per measuring generation, so repeat
                # pairings are independent samples rather than replays.
                seed=50000 + gen * 16 + i,
                render_file=None, render_games=0)
        except Exception as exc:  # noqa: BLE001
            print(f"gen {gen}: ladder match vs gen{opp} failed: {exc}", file=sys.stderr)
            continue
        rows.append({"a": f"gen{gen}", "b": f"gen{opp}", "games": args.ladder_games,
                     "score_a": sc, "kind": "ladder"})
        print(f"  ladder: gen{gen} vs gen{opp}: {sc:.3f} over {args.ladder_games}")
    return rows


def run_diagnostic(args, live, scores, scores_path) -> None:
    """Measure the newest net without gating anything on the result.

    With every generation auto-promoted there is no champion to defend, so a
    match against the *previous champion* would compare two nets one generation
    apart — well inside the noise of a 40-game match. Comparing against the net
    from `--lag` generations back gives the comparison enough separation to say
    something, and greedy stays as the fixed external yardstick.
    """
    snaps = sorted(glob.glob(os.path.join(live, "snapshots", "gen*")),
                   key=lambda p: int(re.search(r"gen(\d+)", p).group(1)))
    if not snaps:
        print("no snapshots to evaluate")
        return
    gen = int(re.search(r"gen(\d+)", snaps[-1]).group(1))
    ckpt = os.path.join(snaps[-1], "checkpoint.safetensors")
    if not os.path.exists(ckpt):
        print(f"gen {gen}: no checkpoint")
        return

    entry = {"gen": gen, "diagnostic": True}
    games_dir = os.path.join(live, "games")

    # ...against the net from `lag` generations back.
    #
    # Resolved by parsing each snapshot's own name rather than formatting the
    # number: the directories are `gen110`, and a `:04d` guess produced
    # `gen0110`, which silently skipped the comparison this eval exists for.
    want = gen - args.lag
    past = ""
    for cand in snaps:
        if int(re.search(r"gen(\d+)", cand).group(1)) == want:
            past = os.path.join(cand, "checkpoint.safetensors")
            break
    if past and os.path.exists(past):
        score, margin, gscore = run_match(
            args.engine, ckpt, past, args.games, args.sims, args.candidates,
            args.t_steps, args.backend, args.ep, args.fp16, seed=gen,
            render_file=os.path.join(games_dir, f"gen_{gen:04d}_vs_lag{args.lag}.txt"),
            render_games=args.render_games)
        entry[f"vs_gen_minus_{args.lag}"] = score
        if margin is not None:
            entry[f"vs_gen_minus_{args.lag}_margin"] = round(margin, 3)
        if gscore is not None:
            entry[f"vs_gen_minus_{args.lag}_score"] = round(gscore, 2)
        if _LAST_TURNS is not None:
            entry["lag_turns_win"], entry["lag_turns_loss"] = _LAST_TURNS
    else:
        print(f"gen {gen}: no snapshot at generation {want}, skipping the lag match")

    # ...and against greedy, the fixed external yardstick.
    try:
        g, gm, ggs = run_match(
            args.engine, ckpt, None, args.greedy_games, args.sims, args.candidates,
            args.t_steps, args.backend, args.ep, args.fp16, seed=1000 + gen,
            render_file=os.path.join(games_dir, f"gen_{gen:04d}_vs_greedy.txt"),
            render_games=args.render_games)
        entry["vs_greedy"] = g
        if gm is not None:
            entry["vs_greedy_margin"] = round(gm, 3)
        if ggs is not None:
            entry["vs_greedy_score"] = round(ggs, 2)
        if _LAST_TURNS is not None:
            entry["greedy_turns_win"], entry["greedy_turns_loss"] = _LAST_TURNS
    except Exception as exc:  # noqa: BLE001
        entry["vs_greedy_error"] = str(exc)[:200]

    metrics = os.path.join(snaps[-1], "metrics.json")
    if os.path.exists(metrics):
        entry.update({k: v for k, v in json.load(open(metrics)).items() if k != "generation"})
    scores["history"].append(entry)
    scores["processed_through"] = gen
    json.dump(scores, open(scores_path, "w"), indent=2)
    print(f"gen {gen}: {entry}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--engine", default="./target/release/minions-engine")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--t-steps", type=int, default=6)
    ap.add_argument("--promote-threshold", type=float, default=0.52)
    ap.add_argument("--greedy-games", type=int, default=12)
    ap.add_argument("--render-games", type=int, default=5,
                    help="games per match written turn-by-turn to <live>/games/. "
                         "0 disables. These are the scored games themselves.")
    ap.add_argument("--backend", default="candle", choices=["candle", "ort"])
    ap.add_argument("--ep", default="trt", choices=["trt", "cuda"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lag", type=int, default=0,
                    help="diagnostic mode: skip promotion entirely and measure the "
                         "newest net against the one from LAG generations back, "
                         "plus greedy. 0 keeps the promotion-gated ladder.")
    ap.add_argument("--ladder-games", type=int, default=6,
                    help="games against each earlier champion when a new one is "
                         "promoted. Coarse on purpose: they are fitted jointly "
                         "by Bradley-Terry, not read individually.")
    ap.add_argument("--ladder-opponents", type=int, default=4,
                    help="how many earlier champions to play, newest first, "
                         "skipping the one the gate just used.")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--period", type=int, default=120)
    args = ap.parse_args()

    live = args.live
    scores_path = os.path.join(live, "scores.json")
    scores = json.load(open(scores_path)) if os.path.exists(scores_path) else {
        "best_gen": 0, "best_score": 0.0, "processed_through": 0, "history": []
    }
    best = os.path.join(live, "best.safetensors")

    if args.lag > 0:
        run_diagnostic(args, live, scores, scores_path)
        return

    while True:
        snaps = sorted(glob.glob(os.path.join(live, "snapshots", "gen*")))
        pending = [s for s in snaps
                   if int(re.search(r"gen(\d+)", s).group(1)) > scores["processed_through"]]
        for snap in pending:
            gen = int(re.search(r"gen(\d+)", snap).group(1))
            ckpt = os.path.join(snap, "checkpoint.safetensors")
            if not os.path.exists(ckpt):
                continue
            champion = best if os.path.exists(best) else None
            # Who `champion` actually is, captured before a promotion overwrites
            # it -- the match log needs both players by name.
            champion_gen = scores.get("best_gen") if champion else None
            entry = {"gen": gen}

            if champion is None:
                # Seed the ladder. With no champion there is nothing to gate
                # against, so the first net becomes the champion outright rather
                # than being measured against the greedy heuristic — greedy is a
                # fixed external yardstick, not a rung on the ladder, and gating
                # on it would either block the run indefinitely (greedy is far
                # stronger than a generation-1 net) or, as it did while the
                # greedy seat was broken, rubber-stamp everything.
                score, promoted = None, True
                print(f"gen {gen}: no champion yet — seeding the ladder")
            else:
                games_dir = os.path.join(live, "games")
                render_file = os.path.join(games_dir, f"gen_{gen:04d}_vs_champion.txt")
                score, margin, gscore = run_match(args.engine, ckpt, champion, args.games, args.sims,
                                          args.candidates, args.t_steps, args.backend, args.ep,
                                          args.fp16, seed=gen,
                                          render_file=render_file,
                                          render_games=args.render_games)
                entry["vs_champion"] = score
                if _LAST_BIDS is not None:
                    entry["bid_candidate"], entry["bid_champion"] = _LAST_BIDS
                if _LAST_TURNS is not None:
                    entry["champ_turns_win"], entry["champ_turns_loss"] = _LAST_TURNS
                # Recorded, not yet gated on. Win/loss here can be a constant —
                # seat decides the game at the turn cap — and a pinned 0.5 means
                # nothing is ever promoted. Margin still separates two nets.
                if margin is not None:
                    entry["vs_champion_margin"] = round(margin, 3)
                if gscore is not None:
                    entry["vs_champion_score"] = round(gscore, 2)
                promoted = score >= args.promote_threshold

            if promoted:
                shutil.copyfile(ckpt, best + ".tmp")
                os.replace(best + ".tmp", best)
                promote_onnx(ckpt, live)
                scores["best_gen"] = gen
                if score is not None:
                    scores["best_score"] = score
                # Persist the promotion *before* the yardstick match. That match
                # is slow and purely informational; if the process dies during
                # it, we would otherwise be left with best.safetensors advanced
                # on disk but scores.json still naming the old champion.
                entry["promoted"] = True
                scores["history"].append(entry)
                scores["processed_through"] = gen
                json.dump(scores, open(scores_path, "w"), indent=2)
                scores["history"].pop()  # replaced below, after the yardstick

            # Every gate match is a rated result whether or not it promoted:
            # a rejected candidate losing 0.3 to the champion says as much
            # about both nets as a win does.
            match_rows = []
            if champion_gen is not None and score is not None:
                match_rows.append({"a": f"gen{gen}", "b": f"gen{champion_gen}",
                                   "games": args.games, "score_a": score,
                                   "kind": "gate"})
            if promoted and args.ladder_games > 0 and args.ladder_opponents > 0:
                match_rows += run_ladder_matches(args, live, scores, gen, champion_gen)

            # Absolute yardstick against the greedy heuristic. Tracking only —
            # it never gates promotion, so a weak early net still climbs its own
            # ladder while this records how far off the external bar it is.
            #
            # It measures the *champion*, not this candidate. The champion is the
            # net self-play actually runs on and the one the run's strength is
            # claimed for; a candidate the gate just rejected never plays a game
            # outside this eval, so scoring it against greedy would report a
            # number for a net that does not exist in the ladder. When the
            # candidate was promoted the two are the same net; when it was
            # rejected they are not, and the champion is the one that matters.
            #
            # Only on a promotion. The champion gate runs every generation to
            # decide promotion, but between promotions the champion is the same
            # net, and re-measuring it just draws another 30-game sample of a
            # number that is already known to the precision this run needs. One
            # reading per champion is the coarse signal we want; `--greedy-games
            # 0` skips it entirely.
            if promoted and args.greedy_games > 0 and os.path.exists(best):
              champ_gen = scores.get("best_gen", gen)
              entry["greedy_gen"] = champ_gen
              try:
                  g, gm, ggs = run_match(args.engine, best, None, args.greedy_games, args.sims,
                                    args.candidates, args.t_steps, args.backend, args.ep,
                                    # Seeded by the measuring generation, not the
                                    # champion's: a champion that survives several
                                    # gates gets re-measured, and reusing its own
                                    # seed would replay the identical 30 games and
                                    # add nothing. Fresh seeds make the repeats
                                    # independent samples that can be pooled.
                                    args.fp16, seed=1000 + gen,
                                    render_file=os.path.join(
                                        live, "games", f"gen_{gen:04d}_vs_greedy.txt"),
                                    render_games=args.render_games)
                  entry["vs_greedy"] = g
                  if gm is not None:
                      entry["vs_greedy_margin"] = round(gm, 3)
                  if ggs is not None:
                      entry["vs_greedy_score"] = round(ggs, 2)
                  if _LAST_TURNS is not None:
                      entry["greedy_turns_win"], entry["greedy_turns_loss"] = _LAST_TURNS
              except Exception as exc:  # noqa: BLE001
                  entry["vs_greedy_error"] = str(exc)[:200]

            if promoted and entry.get("vs_greedy") is not None:
                match_rows.append({"a": f"gen{entry.get('greedy_gen', gen)}",
                                   "b": "greedy", "games": args.greedy_games,
                                   "score_a": entry["vs_greedy"], "kind": "greedy"})

            # Refit over the whole record, not just the new rows: BT is a joint
            # fit, so today's games revise old ratings too.
            if match_rows:
                try:
                    E = elo_mod()
                    E.append(live, match_rows)
                    allm = E.load_matches(os.path.join(live, "matches.jsonl"))
                    ratings = E.fit(allm)
                    if ratings:
                        E.write_ratings(live, ratings, E.games_played(allm), len(allm))
                        mine = ratings.get(f"gen{gen}")
                        if mine is not None:
                            entry["elo"] = round(mine, 1)
                except Exception as exc:  # noqa: BLE001
                    # Ratings are a diagnostic. Never let them break the ladder.
                    print(f"gen {gen}: elo update failed: {exc}", file=sys.stderr)

            entry["promoted"] = promoted
            metrics = os.path.join(snap, "metrics.json")
            if os.path.exists(metrics):
                # Losses plus the derived diagnostics. `energy_kl`/`decl_kl`
                # subtract the target's own entropy, and `score_mae` is in
                # points -- these are the readable forms of the raw losses, so
                # a `loss_` prefix filter drops exactly the useful columns.
                keep = ("loss_", "energy_", "decl_", "score_")
                entry.update({k: v for k, v in json.load(open(metrics)).items()
                              if k.startswith(keep)})
            scores["history"].append(entry)
            scores["processed_through"] = gen
            json.dump(scores, open(scores_path, "w"), indent=2)
            print(f"gen {gen}: {entry}")
        if args.once:
            return
        time.sleep(args.period)


if __name__ == "__main__":
    main()
