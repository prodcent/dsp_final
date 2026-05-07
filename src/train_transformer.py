"""
train_transformer.py
Standalone скрипт обучения encoder-only Transformer для прогноза временных
рядов (cycle time / case arrivals из event log).

Архитектура:
  - Linear projection scalar -> d_model
  - learnable positional embedding на context_len шагов
  - 2 слоя TransformerEncoder (multi-head self-attention + FFN, GELU, dropout)
  - mean-pool по времени
  - MLP head -> 1 скаляр

Тренируется на нормализованных delta-y, чтобы стабилизировать обучение
на коротком ряде. Сравнивает с naive_last и moving_average baselines.

Запуск:
  KMP_DUPLICATE_LIB_OK=TRUE python train_transformer.py \
      --dataset bpi2017 --kpi cycle_time --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_series(dataset: str, kpi: str) -> pd.Series:
    from datasets import load_dataset
    from forecasting import case_arrivals_series, cycle_time_series

    df, _ = load_dataset(dataset)
    if kpi == "cycle_time":
        s = cycle_time_series(df, freq="W", unit="hours")
    elif kpi == "arrivals":
        s = case_arrivals_series(df, freq="W")
    else:
        raise ValueError(f"unknown kpi {kpi}")
    return s.dropna().astype("float64")


def split_series(y: np.ndarray, test_horizon: int):
    n = len(y)
    if test_horizon >= n - 4:
        test_horizon = max(1, n // 4)
    return np.arange(0, n - test_horizon), np.arange(n - test_horizon, n)


def prepare_train_windows(y: np.ndarray, train_idx: np.ndarray, context_len: int):
    """Дельты + standardize (обучение по delta_y, прогноз - тоже delta_y)."""
    dy = np.diff(y, prepend=y[0]).astype("float32")
    train_end = train_idx[-1] + 1
    mu = float(dy[:train_end].mean())
    sd = float(dy[:train_end].std()) or 1.0
    dy_n = (dy - mu) / sd

    Xs, ys = [], []
    for t in range(context_len, train_end):
        Xs.append(dy_n[t - context_len : t])
        ys.append(dy_n[t])
    if not Xs:
        raise ValueError(f"too few train points (need >{context_len})")
    return np.stack(Xs).astype("float32"), np.array(ys, dtype="float32"), dy_n, mu, sd


class TimeSeriesTransformer(nn.Module):
    """Encoder-only Transformer для one-step-ahead forecasting."""

    def __init__(self, context_len: int = 24, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.context_len = context_len
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C]
        x = x.unsqueeze(-1)              # [B, C, 1]
        x = self.proj(x) + self.pos      # [B, C, d_model]
        x = self.encoder(x)              # [B, C, d_model]
        x = x.mean(dim=1)                # mean-pool: [B, d_model]
        return self.head(x).squeeze(-1)  # [B]


def train_one_run(
    y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    context_len: int = 24, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
    epochs: int = 80, lr: float = 1e-3, batch_size: int = 32,
    weight_decay: float = 1e-4, dropout: float = 0.1, seed: int = 42,
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)

    Xs, ys_norm, dy_n, mu, sd = prepare_train_windows(y, train_idx, context_len)
    if verbose:
        print(f"  train windows: {len(Xs)}, context_len={context_len}, batch={batch_size}, lr={lr}")

    model = TimeSeriesTransformer(
        context_len=context_len, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    Xt = torch.tensor(Xs); yt = torch.tensor(ys_norm)

    history = {"train_loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xs))
        ep_loss, n = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            out = model(Xt[b])
            loss = loss_fn(out, yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item()) * len(b); n += len(b)
        history["train_loss"].append(ep_loss / max(n, 1))
        if verbose and ep % 10 == 0:
            print(f"    ep {ep:>3}: train_loss = {history['train_loss'][-1]:.4f}")

    # one-step-ahead inference (autoregressive по доступному ряду)
    model.eval()
    preds = []
    with torch.no_grad():
        for t in test_idx:
            ctx = dy_n[t - context_len:t]
            x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            d_pred_n = float(model(x).item())
            d_pred = d_pred_n * sd + mu
            preds.append(y[t - 1] + d_pred)

    preds = np.asarray(preds, dtype="float64")
    y_test = y[test_idx]
    err = preds - y_test
    metrics = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "sMAPE": float(np.mean(2 * np.abs(err) / (np.abs(preds) + np.abs(y_test) + 1e-6))),
        "n_test": int(len(test_idx)),
        "n_train_windows": int(len(Xs)),
        "params": int(n_params),
        "history": history,
        "preds": preds.tolist(),
        "actual": y_test.tolist(),
    }
    return metrics


def baselines(y: np.ndarray, test_idx: np.ndarray, window: int = 4) -> Dict[str, np.ndarray]:
    naive = y[test_idx - 1]
    ma = []
    for i in test_idx:
        lo = max(0, i - window)
        ma.append(y[lo:i].mean() if i > lo else y[max(0, i - 1)])
    return {"naive_last": naive, "moving_avg": np.asarray(ma)}


def regression_metrics(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "sMAPE": float(np.mean(2 * np.abs(err) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bpi2017", choices=["142_БЗ", "bpi2017"])
    p.add_argument("--kpi", default="cycle_time", choices=["cycle_time", "arrivals"])
    p.add_argument("--test-horizon", type=int, default=8)
    p.add_argument("--context-len", type=int, default=24)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== {args.dataset} | {args.kpi} | test_h={args.test_horizon} ===")
    s = load_series(args.dataset, args.kpi)
    y = s.to_numpy(dtype="float64")
    train_idx, test_idx = split_series(y, args.test_horizon)
    print(f"  series: n={len(y)}  train={len(train_idx)}  test={len(test_idx)}")
    print(f"  mean(train)={y[train_idx].mean():.2f}  std={y[train_idx].std():.2f}")

    print("--- training Transformer ---")
    t = time.time()
    res = train_one_run(
        y, train_idx, test_idx,
        context_len=args.context_len, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )
    res["seconds"] = round(time.time() - t, 1)
    print(f"  Transformer test: {regression_metrics(y[test_idx], res['preds'])} ({res['seconds']}s)")

    print("--- baselines ---")
    bls = baselines(y, test_idx)
    base_metrics = {k: regression_metrics(y[test_idx], v) for k, v in bls.items()}
    for k, v in base_metrics.items():
        print(f"  {k}: {v}")

    out = {
        "dataset": args.dataset, "kpi": args.kpi,
        "params": vars(args),
        "transformer": res,
        "baselines": base_metrics,
    }
    out_path = Path(__file__).resolve().parent.parent / "results" / f"transformer_{args.dataset}_{args.kpi}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
