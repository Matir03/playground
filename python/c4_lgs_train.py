"""Trainer for the Connect 4 reification-based latent-space net (docs/algorithm.md).

Reads v2 self-play dumps from `connect4-engine lgs-gen` and fits the LGS net:

  L = value MSE + lpi·MDN-policy-NLL + lrho·reifier-CE + lc·consistency + reg

All targets live in the *current* encoder's latent space: parents and reified
children are re-encoded each step, so the MDN policy target (the search's visit
distribution over next-LGS) and the reifier target track the moving encoder.
Writes checkpoint.safetensors + config.json (arch) that the candle self-play
net (crates/games/connect4/src/lgs_net.rs) loads directly.

Usage:
  # seed a random checkpoint to bootstrap self-play
  python c4_lgs_train.py --init data/models/connect4_lgs/live --channels 32 --blocks 3
  # train one or more dumps -> new checkpoint
  python c4_lgs_train.py --buffer run/buf --in live/checkpoint.safetensors \
      --out live/checkpoint.safetensors --epochs 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server import c4_lgs_net as M

PLANES = 2 * 6 * 7  # 84
MAX_CHILDREN = 7
REC_F32 = PLANES + 1 + MAX_CHILDREN * (2 + PLANES)  # 687, must match lgs_selfplay.rs


def read_dump(path: str):
    """Read a v2 dump into arrays.

    Returns (parent (N,2,6,7), z (N,), cols (N,7) int, visits (N,7),
             child (N,7,2,6,7))."""
    raw = np.fromfile(path, dtype=np.uint8)
    count = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    flat = np.frombuffer(raw[8 : 8 + count * REC_F32 * 4], dtype="<f4").reshape(count, REC_F32)
    parent = flat[:, :PLANES].reshape(count, 2, 6, 7)
    z = flat[:, PLANES].copy()
    block = flat[:, PLANES + 1 :].reshape(count, MAX_CHILDREN, 2 + PLANES)
    cols = block[:, :, 0].astype(np.int64)
    visits = block[:, :, 1].astype(np.float32)
    child = block[:, :, 2:].reshape(count, MAX_CHILDREN, 2, 6, 7)
    return parent, z, cols, visits, child


def load_buffer(path: str):
    files = (
        sorted(glob.glob(os.path.join(path, "*.bin")))
        if os.path.isdir(path)
        else [path]
    )
    if not files:
        raise SystemExit(f"no dump files under {path!r}")
    parts = [read_dump(f) for f in files]
    return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(5))


def lex_perm(ec, present):
    """Per-row permutation ordering children by *decreasing lexicographic* order of
    their encoding (present first, absent last). ec:(B,A,d) detached, present:(B,A).
    Returns (B,A) long indices. Gives the discrete move-policy a canonical, stable
    target assignment for set-prediction (pol_next[i] <- i-th lex child)."""
    e = ec.cpu().numpy()
    pres = (present > 0).cpu().numpy()
    d = e.shape[-1]
    # np.lexsort: last key is primary. Want primary = present-flag (absent last),
    # then dim0 desc, dim1 desc, ...  -> least-significant first: -dim_{d-1}..-dim0, flag.
    keys = [(-e[:, :, i]) for i in range(d - 1, -1, -1)]
    keys.append((~pres).astype(np.int8))
    perm = np.lexsort(keys, axis=1)  # (B,A) ascending => present-first, lex-desc
    return torch.from_numpy(perm).to(ec.device)


def min_loss_perm(pol_next, e_child, present, ptgt, prior_logits, ce_ratio):
    """Per-row assignment (slot j <- child) that MINIMIZES the move-policy loss —
    the optimal bipartite matching (DETR-style Hungarian), replacing the fixed lex
    order. The lex convention forced the policy to predict a target ordering it
    never uses at inference (search assigns priors by the *nearest* pol_next slot);
    matching to the loss-minimizing assignment removes that train/infer mismatch.

    Per-example cost of assigning slot j to child c:
        C[j,c] = present[c]*||pol_next[j]-e_child[c]||^2  -  ce_ratio*ptgt[c]*logp[j]
    (logp = log_softmax(prior_logits)) — i.e. the latent-MSE + prior-CE terms of
    the loss, with ce_ratio = npresent/nfull so the two match the loss's relative
    weighting (lpi = nmse/npresent + ce/nfull). Absent children give zero-cost
    columns and fill leftover slots. All inputs are detached (the matching is a
    discrete target choice, not differentiated). Returns (B,A) long indices with
    lp[b,j] = child assigned to slot j (gather-compatible with the lex path)."""
    from scipy.optimize import linear_sum_assignment
    B, A, _ = pol_next.shape
    mse = ((pol_next.unsqueeze(2) - e_child.unsqueeze(1)) ** 2).sum(-1)  # (B,slot,child)
    logp = torch.log_softmax(prior_logits, dim=1)                        # (B,slot)
    cost = present.unsqueeze(1) * mse - ce_ratio * (ptgt.unsqueeze(1) * logp.unsqueeze(2))
    C = cost.detach().cpu().numpy()
    perm = np.empty((B, A), dtype=np.int64)
    for b in range(B):
        _, col = linear_sum_assignment(C[b])  # rows 0..A-1 sorted -> col[j] = child for slot j
        perm[b] = col
    return torch.from_numpy(perm).to(pol_next.device)


N_ALIAS_SAMPLES = 4


def alias_decode_loss(net, w, mu, std, e_child, bch, bvis, nfull):
    """Decoder loss on intermediate ('aliased') latent points.

    The search queries the decoder on MDN *samples*, not exact encodings, so the
    decoder must map the latent neighborhood of real children to legal positions.
    Sample from the MDN (detached -> trains the decoder only, not the MDN/encoder),
    assign each sample to its nearest recorded child (latent NN over present
    children), and BCE-reconstruct that child's board. Normalized by nfull."""
    B, _, d = e_child.shape
    present = bvis > 0                                   # (B, MAX_CHILDREN)
    has_child = present.any(dim=1).float()               # (B,)
    comp = torch.multinomial(w.detach(), N_ALIAS_SAMPLES, replacement=True)  # (B, n)
    idx = comp.unsqueeze(-1).expand(-1, -1, d)
    mu_s = torch.gather(mu.detach(), 1, idx)
    std_s = torch.gather(std.detach(), 1, idx)
    samples = mu_s + std_s * torch.randn_like(std_s)     # (B, n, d)
    ec = e_child.detach()
    d2 = ((samples.unsqueeze(2) - ec.unsqueeze(1)) ** 2).sum(-1)  # (B, n, MAX_CHILDREN)
    d2 = d2.masked_fill(~present.unsqueeze(1), float("inf"))
    nn = d2.argmin(-1)                                   # (B, n) nearest present child
    tgt = torch.gather(
        bch.reshape(B, MAX_CHILDREN, -1), 1, nn.unsqueeze(-1).expand(-1, -1, PLANES)
    )                                                    # (B, n, PLANES)
    dec = net.decode(samples.reshape(B * N_ALIAS_SAMPLES, d)).reshape(B, N_ALIAS_SAMPLES, -1)
    bce = F.binary_cross_entropy_with_logits(dec, tgt, reduction="none").mean(-1)  # (B, n)
    return (bce.mean(-1) * has_child).sum() / nfull


def do_init(args) -> None:
    os.makedirs(args.init, exist_ok=True)
    net = M.build_lgs_model({"channels": args.channels, "blocks": args.blocks, "k": args.k,
                             "decoder": args.decoder, "move_policy": args.move_policy})
    M.save_checkpoint(net, os.path.join(args.init, "checkpoint.safetensors"))
    json.dump({"arch": M.arch_of(net)}, open(os.path.join(args.init, "config.json"), "w"))
    print(f"init: random LGS net -> {args.init} (arch {M.arch_of(net)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="write a random checkpoint to this dir and exit")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--k", type=int, default=M.K_COMPONENTS)
    ap.add_argument("--decoder", action="store_true", help="build a position-decoder edge head (vs reifier)")
    ap.add_argument("--move-policy", action="store_true", help="discrete per-move latent policy (vs MDN) — C4")
    ap.add_argument("--buffer", help="dump file or dir of *.bin")
    ap.add_argument("--in", dest="inp", help="checkpoint to resume from")
    ap.add_argument("--out", help="checkpoint to write")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lv", type=float, default=1.0, help="value (MSE) weight")
    ap.add_argument("--lpi", type=float, default=1.0, help="MDN policy weight")
    ap.add_argument("--lrho", type=float, default=1.0, help="reifier weight")
    # Consistency defaults OFF: the exact-match form (α·d_lat ≈ d_func) collapses
    # the latent while the value head is still ≈constant (d_func≈0 ⇒ d_lat→0),
    # which kills value/reifier learning (docs/algorithm.md "central tension").
    # Re-enable later as a ramped anti-collapse (Lipschitz) term.
    ap.add_argument("--lc", type=float, default=0.0, help="consistency weight (0 = off)")
    ap.add_argument("--ent", type=float, default=1e-3, help="mixture-entropy bonus")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.init:
        do_init(args)
        return
    if not args.buffer or not args.out:
        raise SystemExit("training needs --buffer and --out (or use --init)")

    dev = torch.device(args.device)
    if args.inp and os.path.exists(args.inp):
        cfg = json.load(open(os.path.join(os.path.dirname(args.inp), "config.json")))["arch"]
        net = M.build_lgs_model(cfg)
        M.load_checkpoint(net, args.inp)
    else:
        net = M.build_lgs_model({"channels": args.channels, "blocks": args.blocks, "k": args.k})
    net.to(dev).train()

    alpha = torch.nn.Parameter(torch.tensor(1.0, device=dev))  # consistency metric scale
    opt = torch.optim.Adam(list(net.parameters()) + [alpha], lr=args.lr, weight_decay=args.wd)

    parent, z, cols, visits, child = load_buffer(args.buffer)
    n = parent.shape[0]
    parent_t = torch.from_numpy(parent).to(dev)
    z_t = torch.from_numpy(z).to(dev)
    cols_t = torch.from_numpy(cols).to(dev)
    visits_t = torch.from_numpy(visits).to(dev)
    child_t = torch.from_numpy(child).to(dev)
    print(f"train: {n} examples, {args.epochs} epochs, batch {args.batch}, device {dev}")

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = {"v": 0.0, "pi": 0.0, "rho": 0.0, "c": 0.0, "n": 0}
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            bp, bz, bcol, bvis, bch = parent_t[b], z_t[b], cols_t[b], visits_t[b], child_t[b]
            B = bp.shape[0]

            e_parent = net.encode(bp)  # (B,d)
            # value
            lv = F.mse_loss(net.value(e_parent), bz)

            # children encodings + visit-weighted normalization
            e_child = net.encode(bch.reshape(B * MAX_CHILDREN, 2, 6, 7)).reshape(B, MAX_CHILDREN, -1)
            wsum = bvis.sum(dim=1, keepdim=True).clamp_min(1.0)
            pi = bvis / wsum  # (B,7), zero for absent children
            # PCR: value trains on all positions; policy/reifier only on full
            # (high-playout) ones, which carry visits. Normalize those losses by
            # the count of full records so the fast (value-only) majority doesn't
            # dilute the policy gradient.
            nfull = (bvis.sum(dim=1) > 0).sum().clamp_min(1).float()

            present = (bvis > 0).float()  # (B,7) real children
            npresent = present.sum().clamp_min(1.0)
            move_policy = getattr(net, "move_policy", False)
            sorted_planes = sorted_present = None
            if move_policy:
                # Discrete per-move policy (C4): the 7 outputs are a *set*; assign
                # each slot to a real child by the loss-MINIMIZING bipartite matching
                # (min_loss_perm), so the policy is trained against the same
                # slot<->child association search recovers (nearest pol_next slot)
                # rather than an arbitrary lex order. pol_next[i] -> matched child
                # encoding (MSE, the learned dynamics); pol_prior[i] -> that child's
                # visit-prob (CE).
                prior_logits, pol_nxt = net.move_policy_fwd(e_parent)  # (B,7),(B,7,d)
                child_ptgt = bvis / bvis.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,7) over children
                ce_ratio = float(npresent / nfull)
                lp = min_loss_perm(
                    pol_nxt.detach(), e_child.detach(), present, child_ptgt,
                    prior_logits.detach(), ce_ratio,
                )  # (B,7) optimal slot<-child assignment (not the batch perm!)
                pe = lp.unsqueeze(-1).expand(-1, -1, M.LGS_DIM)
                sorted_child = torch.gather(e_child, 1, pe)  # (B,7,d)
                sorted_vis = torch.gather(bvis, 1, lp)  # (B,7)
                sorted_present = torch.gather(present, 1, lp)  # (B,7)
                sorted_planes = torch.gather(
                    bch.reshape(B, MAX_CHILDREN, -1), 1, lp.unsqueeze(-1).expand(-1, -1, PLANES)
                )  # (B,7,PLANES) lex-ordered child boards (for the decoder loss)
                ptgt = sorted_vis / sorted_vis.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,7)
                ce = -(ptgt * F.log_softmax(prior_logits, dim=1)).sum(dim=1)  # (B,)
                lpi = (ce * (bvis.sum(dim=1) > 0)).sum() / nfull
                nmse = ((pol_nxt - sorted_child) ** 2).sum(-1)  # (B,7)
                lpi = lpi + (nmse * sorted_present).sum() / npresent
            else:
                w, mu, std = net.policy_mdn(e_parent)
                lpi = torch.zeros((), device=dev)
                for c in range(MAX_CHILDREN):
                    m = pi[:, c]  # (B,)
                    # Per-dimension NLL: a d-dim Gaussian log-density is ~O(d), which
                    # otherwise swamps the value/edge gradients on the shared
                    # encoder. Normalizing by d puts all loss terms on an O(1) scale.
                    logp = M.mdn_log_prob(e_child[:, c], w, mu, std) / M.LGS_DIM  # (B,)
                    lpi = lpi - (m * logp).sum()
                lpi = lpi / nfull

            # Edge head: reifier cross-entropy, or position-decoder reconstruction
            # (BCE of decode(E(S)) vs board planes — the decoder inverts the
            # encoder). Decoder trains on the parent (all records) plus the
            # visit-weighted full-move children (the positions search decodes).
            if getattr(net, "decoder", False):
                lrho = F.binary_cross_entropy_with_logits(net.decode(e_parent), bp.reshape(B, -1))
                crec = torch.zeros((), device=dev)
                for c in range(MAX_CHILDREN):
                    rec = F.binary_cross_entropy_with_logits(
                        net.decode(e_child[:, c]), bch[:, c].reshape(B, -1), reduction="none"
                    ).mean(dim=1)  # (B,)
                    crec = crec + (pi[:, c] * rec).sum()
                lrho = lrho + crec / nfull
                if move_policy:
                    # Decoder on the policy's predicted points: decode(pol_next[i])
                    # -> the i-th lex child's board (detached -> trains the decoder,
                    # the diagnostic the search's valid-decode rate measures).
                    dpx = net.decode(pol_nxt.detach().reshape(B * MAX_CHILDREN, M.LGS_DIM)).reshape(B, MAX_CHILDREN, -1)
                    pbce = F.binary_cross_entropy_with_logits(
                        dpx, sorted_planes, reduction="none"
                    ).mean(-1)  # (B,7)
                    lrho = lrho + (pbce * sorted_present).sum() / npresent
                else:
                    # Intermediate ("aliased") latent points: the search queries the
                    # decoder on MDN *samples*, not exact encodings. Teach it to map
                    # each sample to its nearest recorded child's board (detached).
                    lrho = lrho + alias_decode_loss(net, w, mu, std, e_child, bch, bvis, nfull)
            else:
                lrho = torch.zeros((), device=dev)
                for c in range(MAX_CHILDREN):
                    rlog = net.reify_logits(e_parent, e_child[:, c])  # (B,7)
                    ce = F.cross_entropy(rlog, bcol[:, c], reduction="none")
                    lrho = lrho + (pi[:, c] * ce).sum()
                lrho = lrho / nfull

            # consistency: random within-batch pairs, value-only d_func (start)
            j = torch.randperm(B, device=dev)
            d_lat = (e_parent - e_parent[j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            d_func = (net.value(e_parent) - net.value(e_parent)[j]).abs()
            lc = (alpha * d_lat - d_func).pow(2).mean()

            # mixture-entropy bonus (encourage spread, discourage collapse).
            # Only meaningful for the MDN; the discrete move-policy uses prior CE.
            ent = torch.zeros((), device=dev) if move_policy else \
                -(w.clamp_min(1e-9).log() * w).sum(-1).mean()

            loss = args.lv * lv + args.lpi * lpi + args.lrho * lrho + args.lc * lc - args.ent * ent
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot["v"] += float(lv.detach()) * B
            tot["pi"] += float(lpi.detach()) * B
            tot["rho"] += float(lrho.detach()) * B
            tot["c"] += float(lc.detach()) * B
            tot["n"] += B
        d = tot["n"]
        print(
            f"epoch {epoch}: v {tot['v']/d:.4f}  pi {tot['pi']/d:.4f}  "
            f"rho {tot['rho']/d:.4f}  c {tot['c']/d:.4f}  alpha {float(alpha):.3f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    net.cpu().eval()
    M.save_checkpoint(net, args.out)
    json.dump({"arch": M.arch_of(net)}, open(os.path.join(os.path.dirname(args.out) or ".", "config.json"), "w"))
    print(f"wrote {args.out} (arch {M.arch_of(net)})")


if __name__ == "__main__":
    main()
