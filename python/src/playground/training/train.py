"""Behavioral cloning: train a Connect 4 net to imitate the perfect solver.

Targets come from generate.py (perfect-bot positions only):
  - policy: cross-entropy on the column the solver played
  - value:  MSE on the eventual game result for the side to move

Trains the encoder + eval + policy-prior heads (the play-relevant ones; the
reifier is structural for Connect 4 — see net.py). Saves a checkpoint into the
model registry layout from docs/training.md.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from playground.model_server.net import (
    ConnectFourModel,
    connect4_planes,
    load_checkpoint,
    save_checkpoint,
)


def _planes(states: np.ndarray) -> torch.Tensor:
    return torch.stack([connect4_planes(bytes(s)) for s in states])


def train(
    data_path: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    init: str | None,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    d = np.load(data_path)
    states, cols, values = d["states"], d["cols"], d["values"]
    n = len(cols)
    if n == 0:
        raise SystemExit("dataset is empty — run gen-data first")
    print(f"dataset: {n} perfect-bot positions; value mean {values.mean():+.3f}")

    planes = _planes(states)
    cols_t = torch.from_numpy(cols)
    values_t = torch.from_numpy(values)

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = ConnectFourModel()
    if init:
        load_checkpoint(model, init)
        print(f"warm-started from {init}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate(idx):
        model.eval()
        with torch.no_grad():
            v_pred, logits = model(planes[idx])
            acc = (logits.argmax(1) == cols_t[idx]).float().mean().item()
            vmse = F.mse_loss(v_pred, values_t[idx]).item()
        return acc, vmse

    acc = vmse = 0.0
    for ep in range(epochs):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for i in range(0, len(order), batch_size):
            b = order[i : i + batch_size]
            v_pred, logits = model(planes[b])
            loss = F.mse_loss(v_pred, values_t[b]) + F.cross_entropy(logits, cols_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr_acc, _ = evaluate(tr_idx)
        acc, vmse = evaluate(val_idx)
        print(
            f"epoch {ep + 1:2}/{epochs}  train_move_acc {tr_acc:.3f}  "
            f"val_move_acc {acc:.3f}  val_value_mse {vmse:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.safetensors")
    save_checkpoint(model, ckpt)
    meta = {
        "trained_from": init,
        "data": data_path,
        "positions": int(n),
        "epochs": epochs,
        "val_move_acc": acc,
        "val_value_mse": vmse,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {ckpt}  (val move-match vs perfect bot: {acc:.1%})")
    return ckpt
