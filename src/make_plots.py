"""
make_plots.py
Генерация графиков для презентации курсового проекта.

Использует:
  - готовые JSON с метриками из results/
  - сырые event-logs из ../../data/ через src/datasets.py
  - logs/run_extended.log для loss curves

Все графики сохраняются в plots/ как PNG (300 DPI).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


# ============================================================
#                  DATA LOADING (для EDA)
# ============================================================


def load_142():
    from datasets import load_dataset
    df, _ = load_dataset("142_БЗ")
    return df


def load_bpi2017():
    from datasets import load_dataset
    df, _ = load_dataset("bpi2017")
    return df


# ============================================================
#                  EDA PLOTS
# ============================================================


def plot_case_duration_distribution(df: pd.DataFrame, name: str, unit: str = "hours"):
    factor = 3600.0 if unit == "hours" else 86400.0
    grp = df.groupby("case_id")["timestamp"]
    durs = (grp.max() - grp.min()).dt.total_seconds() / factor

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(durs, bins=60, color="#3b6dbf", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"{name}: распределение длительности кейсов")
    axes[0].set_xlabel(f"Длительность ({unit})")
    axes[0].set_ylabel("Количество кейсов")
    axes[0].axvline(durs.median(), color="red", linestyle="--", label=f"медиана = {durs.median():.0f}")
    axes[0].legend()

    axes[1].hist(np.log1p(durs), bins=60, color="#3b6dbf", edgecolor="white", alpha=0.85)
    axes[1].set_title(f"{name}: log1p длительности (heavy tail)")
    axes[1].set_xlabel(f"log(1 + duration_{unit})")
    axes[1].set_ylabel("Количество кейсов")

    fig.tight_layout()
    fig.savefig(PLOTS / f"eda_{name}_case_duration.png")
    plt.close(fig)
    print(f"  saved eda_{name}_case_duration.png")


def plot_weekly_kpi_series(df: pd.DataFrame, name: str):
    from forecasting import case_arrivals_series, cycle_time_series

    arrivals = case_arrivals_series(df, freq="W").dropna()
    cycle = cycle_time_series(df, freq="W", unit="hours").dropna()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    axes[0].plot(arrivals.index, arrivals.values, color="#1c8c4e", marker="o", markersize=3, lw=1)
    axes[0].set_title(f"{name}: количество новых кейсов в неделю")
    axes[0].set_ylabel("Кейсов / неделя")

    axes[1].plot(cycle.index, cycle.values, color="#bf3b3b", marker="o", markersize=3, lw=1)
    axes[1].set_title(f"{name}: средний cycle time кейсов, начатых на этой неделе")
    axes[1].set_ylabel("Часы")
    axes[1].set_xlabel("Дата (старт недели)")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / f"eda_{name}_weekly_kpi.png")
    plt.close(fig)
    print(f"  saved eda_{name}_weekly_kpi.png")


def plot_train_test_split(df: pd.DataFrame, name: str, test_horizon: int = 8):
    from forecasting import cycle_time_series, make_train_test

    s = cycle_time_series(df, freq="W", unit="hours").dropna().astype("float64")
    sp = make_train_test(s, test_horizon=test_horizon)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index[sp.train_idx], s.values[sp.train_idx], color="#3b6dbf", label=f"train ({len(sp.train_idx)} нед)", lw=1.5)
    ax.plot(s.index[sp.test_idx], s.values[sp.test_idx], color="#bf3b3b", label=f"test ({len(sp.test_idx)} нед)", lw=2, marker="o")
    ax.axvline(s.index[sp.test_idx[0]], color="black", linestyle="--", lw=1, alpha=0.5)
    ax.set_title(f"{name}: train/test split на ряду cycle_time (weekly)")
    ax.set_ylabel("Cycle time, часы")
    ax.set_xlabel("Дата")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / f"eda_{name}_train_test_split.png")
    plt.close(fig)
    print(f"  saved eda_{name}_train_test_split.png")


def plot_activity_distribution(df: pd.DataFrame, name: str, top_k: int = 15):
    counts = df["activity"].astype(str).value_counts().head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(np.arange(len(counts)), counts.values, color="#3b6dbf", alpha=0.85)
    ax.set_yticks(np.arange(len(counts)))
    short_labels = [c[:55] + "..." if len(c) > 55 else c for c in counts.index]
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Количество событий")
    ax.set_title(f"{name}: топ-{top_k} активностей по частоте")
    ax.invert_xaxis()
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS / f"eda_{name}_top_activities.png")
    plt.close(fig)
    print(f"  saved eda_{name}_top_activities.png")


# ============================================================
#                  MODEL COMPARISON PLOTS
# ============================================================


def _extract_models(results_dict: Dict, kpi_key: str) -> Dict[str, Dict]:
    fc = results_dict.get("B_forecasting", {})
    ent = fc.get(kpi_key, {})
    return ent.get("models", {})


def plot_models_comparison_bar(name: str, kpi_key: str, kpi_label: str):
    data = json.loads((RESULTS / f"{name}__extended.json").read_text(encoding="utf-8"))
    models = _extract_models(data, kpi_key)
    rows = []
    for k, v in models.items():
        if isinstance(v, dict) and "MAE" in v:
            rows.append({"model": k, "MAE": v["MAE"], "sMAPE": v["sMAPE"], "R2": v["R2"]})
    if not rows:
        print(f"  skip {name}/{kpi_key}: no model metrics")
        return
    df = pd.DataFrame(rows).sort_values("MAE")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#3b6dbf"] * len(df)
    if "timesfm_lite" in df["model"].values:
        idx = df.index[df["model"] == "timesfm_lite"][0]
        colors[df.index.get_loc(idx)] = "#bf3b3b"

    axes[0].barh(np.arange(len(df)), df["MAE"], color=colors, alpha=0.85)
    axes[0].set_yticks(np.arange(len(df)))
    axes[0].set_yticklabels(df["model"])
    axes[0].set_xlabel("MAE")
    axes[0].invert_yaxis()
    axes[0].set_title("MAE (меньше - лучше)")

    axes[1].barh(np.arange(len(df)), df["sMAPE"], color=colors, alpha=0.85)
    axes[1].set_yticks(np.arange(len(df)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("sMAPE")
    axes[1].invert_yaxis()
    axes[1].set_title("sMAPE (меньше - лучше)")

    axes[2].barh(np.arange(len(df)), df["R2"], color=colors, alpha=0.85)
    axes[2].axvline(0, color="black", lw=0.6)
    axes[2].set_yticks(np.arange(len(df)))
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("R²")
    axes[2].invert_yaxis()
    axes[2].set_title("R² (больше - лучше)")

    fig.suptitle(f"{name}: {kpi_label} - сравнение 6 моделей на test")
    fig.tight_layout()
    fig.savefig(PLOTS / f"results_{name}_{kpi_key}_bars.png")
    plt.close(fig)
    print(f"  saved results_{name}_{kpi_key}_bars.png")


def plot_pred_vs_actual(name: str, kpi_key: str, kpi_label: str, test_horizon: int = 8):
    """Перепрогон легких forecasters для построения predicted-vs-actual.
    Использует тот же seed/split что в основном прогоне."""
    from datasets import load_dataset
    from forecasting import (
        case_arrivals_series, cycle_time_series, make_train_test,
        forecast_naive_last, forecast_seasonal_naive, forecast_moving_average,
        forecast_ridge_lags, forecast_xgb_lags, forecast_timesfm_lite,
    )
    df, _ = load_dataset(name)
    if kpi_key == "case_arrivals_weekly":
        s = case_arrivals_series(df, freq="W").dropna().astype("float64")
    elif kpi_key == "cycle_time_weekly_hours":
        s = cycle_time_series(df, freq="W", unit="hours").dropna().astype("float64")
    else:
        return
    sp = make_train_test(s, test_horizon=test_horizon)
    seasonal_lag = 52 if len(s) > 60 else max(4, len(s) // 4)

    fcs = {
        "naive_last":        forecast_naive_last(s, sp),
        "seasonal_naive":    forecast_seasonal_naive(s, sp, seasonal_lag=seasonal_lag),
        "moving_avg":        forecast_moving_average(s, sp),
        "ridge_lags":        forecast_ridge_lags(s, sp),
        "xgb_lags":          forecast_xgb_lags(s, sp),
        "timesfm_lite":      forecast_timesfm_lite(s, sp),
    }
    y_true = s.to_numpy()[sp.test_idx]

    # 2x3 grid каждый предиктор отдельно
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    palette = {
        "naive_last": "#888",
        "seasonal_naive": "#5a9",
        "moving_avg": "#1c8c4e",
        "ridge_lags": "#7a3b9b",
        "xgb_lags": "#cc7711",
        "timesfm_lite": "#bf3b3b",
    }

    test_dates = s.index[sp.test_idx]
    train_tail = max(0, sp.test_idx[0] - 24)  # последние 24 train точки для контекста
    train_dates = s.index[train_tail:sp.test_idx[0]]
    train_vals = s.values[train_tail:sp.test_idx[0]]

    for ax, (model, pred) in zip(axes.flatten(), fcs.items()):
        ax.plot(train_dates, train_vals, color="#222", lw=0.8, alpha=0.55, label="train (контекст)")
        ax.plot(test_dates, y_true, color="black", lw=1.6, marker="o", markersize=3, label="actual")
        ax.plot(test_dates, pred, color=palette[model], lw=1.6, marker="x", markersize=4, label=f"pred {model}")
        mae = np.mean(np.abs(np.asarray(pred) - y_true))
        ax.set_title(f"{model} (MAE={mae:.1f})")
        ax.legend(loc="best", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Дата")
    for ax in axes[:, 0]:
        ax.set_ylabel(kpi_label)

    fig.suptitle(f"{name}: predicted vs actual - {kpi_label}", y=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / f"results_{name}_{kpi_key}_pred_vs_actual.png")
    plt.close(fig)
    print(f"  saved results_{name}_{kpi_key}_pred_vs_actual.png")
    return fcs, y_true, test_dates


def plot_residuals(fcs: Dict[str, np.ndarray], y_true: np.ndarray, name: str, kpi_key: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = {
        "naive_last": "#888",
        "seasonal_naive": "#5a9",
        "moving_avg": "#1c8c4e",
        "ridge_lags": "#7a3b9b",
        "xgb_lags": "#cc7711",
        "timesfm_lite": "#bf3b3b",
    }
    for model, pred in fcs.items():
        residuals = np.asarray(pred) - y_true
        ax.plot(np.arange(1, len(residuals) + 1), residuals, marker="o",
                lw=1.4, color=palette.get(model, "#000"), label=model, markersize=4)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Шаг прогноза (test horizon)")
    ax.set_ylabel("Residual = pred - actual")
    ax.set_title(f"{name}: ошибка по шагам ({kpi_key})")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / f"results_{name}_{kpi_key}_residuals.png")
    plt.close(fig)
    print(f"  saved results_{name}_{kpi_key}_residuals.png")


# ============================================================
#                  TRAINING CURVE (наш Transformer)
# ============================================================


def plot_training_curve_from_synthetic():
    """Перетренируем мини-Transformer и сохраним loss-curve.
    Используем malый train, чтобы показать честную динамику обучения."""
    from datasets import load_dataset
    from forecasting import cycle_time_series, make_train_test

    df, _ = load_dataset("bpi2017")
    s = cycle_time_series(df, freq="W", unit="hours").dropna().astype("float64")
    sp = make_train_test(s, test_horizon=8)
    y = s.to_numpy(dtype="float32")
    dy = np.diff(y, prepend=y[0]).astype("float32")
    train_end = sp.test_idx[0]
    mu = float(dy[:train_end].mean()); sd = float(dy[:train_end].std()) or 1.0
    dy_n = (dy - mu) / sd

    context_len = 24
    Xs, ys = [], []
    for t in range(context_len, train_end):
        Xs.append(dy_n[t - context_len: t])
        ys.append(dy_n[t])
    Xs = np.stack(Xs).astype("float32"); ys = np.array(ys, dtype="float32")

    import torch
    from torch import nn
    torch.manual_seed(42); np.random.seed(42)

    class TFM(nn.Module):
        def __init__(self, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.proj = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.zeros(1, context_len, d_model))
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                             dim_feedforward=d_model*2, dropout=0.1,
                                             batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        def forward(self, x):
            x = x.unsqueeze(-1); x = self.proj(x) + self.pos; x = self.enc(x); x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

    model = TFM().cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    Xt = torch.tensor(Xs); yt = torch.tensor(ys)

    losses = []
    for ep in range(80):
        model.train()
        idx = torch.randperm(len(Xs))
        ep_loss = 0.0
        bs = 32
        for i in range(0, len(Xs), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            out = model(Xt[b]); loss = loss_fn(out, yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item()) * len(b)
        losses.append(ep_loss / len(Xs))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#3b6dbf", lw=1.6)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Train loss (SmoothL1)")
    ax.set_title("Transformer-lite forecaster: динамика обучения (bpi2017 cycle_time)")
    fig.tight_layout()
    fig.savefig(PLOTS / "training_loss_curve.png")
    plt.close(fig)
    print(f"  saved training_loss_curve.png")


# ============================================================
#                  MASTER
# ============================================================


def main():
    print("=== Loading datasets ===")
    df_142 = load_142()
    df_bpi = load_bpi2017()

    print("=== EDA plots ===")
    plot_case_duration_distribution(df_142, "142_БЗ")
    plot_case_duration_distribution(df_bpi, "bpi2017")
    plot_weekly_kpi_series(df_142, "142_БЗ")
    plot_weekly_kpi_series(df_bpi, "bpi2017")
    plot_train_test_split(df_142, "142_БЗ")
    plot_train_test_split(df_bpi, "bpi2017")
    plot_activity_distribution(df_142, "142_БЗ")
    plot_activity_distribution(df_bpi, "bpi2017")

    print("=== Model comparison plots ===")
    for ds in ["142_БЗ", "bpi2017"]:
        for kpi_key, kpi_label in [
            ("case_arrivals_weekly", "новые кейсы / неделя"),
            ("cycle_time_weekly_hours", "cycle time, часы"),
        ]:
            plot_models_comparison_bar(ds, kpi_key, kpi_label)
            try:
                fcs, y_true, _ = plot_pred_vs_actual(ds, kpi_key, kpi_label)
                plot_residuals(fcs, y_true, ds, kpi_key)
            except Exception as e:
                print(f"  skip pred_vs_actual {ds}/{kpi_key}: {e}")

    print("=== Training curve ===")
    try:
        plot_training_curve_from_synthetic()
    except Exception as e:
        print(f"  skip training curve: {e}")

    print("\nDone. Plots in:", PLOTS)


if __name__ == "__main__":
    main()
