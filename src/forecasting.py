# core/forecasting.py
"""
Process Model & KPI Forecasting (De Smedt et al. 2023, "Process model
forecasting and change exploration using time series analysis of event
sequence data").

Идея: из event-log извлекаем регулярные временные ряды
  - case arrival rate per period (новые кейсы),
  - cycle time per period (среднее время полного цикла кейсов, начавшихся в окне),
  - DFG edge frequencies per period для топ-K рёбер,
и прогнозируем будущие значения.

Реализованные предикторы:
  - Naive last-value (persistence)
  - Seasonal-naive (значение из периода назад на seasonality_lag)
  - Moving-average baseline
  - Ridge на лагах + календарных фичах
  - XGBoost на лагах + календарных фичах
  - TimesFMLite: encoder-only Transformer над lag-окнами (foundation-style)

Метрики на test: MAE, RMSE, sMAPE, R^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
#                           METRICS
# ============================================================


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sm = smape(y_true, y_pred)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "sMAPE": sm, "R2": r2, "n": int(len(y_true))}


# ============================================================
#                           SERIES BUILDERS
# ============================================================


def case_arrivals_series(df: pd.DataFrame, freq: str = "W") -> pd.Series:
    """Число кейсов, начавшихся в каждом периоде (старт = первый event кейса)."""
    starts = df.groupby("case_id")["timestamp"].min()
    s = starts.dt.to_period(freq).value_counts().sort_index()
    s.index = s.index.to_timestamp()
    return s.astype("float64").rename("case_arrivals")


def case_completions_series(df: pd.DataFrame, freq: str = "W") -> pd.Series:
    ends = df.groupby("case_id")["timestamp"].max()
    s = ends.dt.to_period(freq).value_counts().sort_index()
    s.index = s.index.to_timestamp()
    return s.astype("float64").rename("case_completions")


def cycle_time_series(df: pd.DataFrame, freq: str = "W", unit: str = "hours") -> pd.Series:
    """
    Средний cycle time кейсов, начавшихся в данном окне.
    Считается только по завершённым в данных кейсам (как в De Smedt 2023:
    forecasting on completed traces).
    """
    factor = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[unit]
    grp = df.groupby("case_id")["timestamp"]
    starts = grp.min()
    durs = (grp.max() - starts).dt.total_seconds() / factor
    df_c = pd.DataFrame({"start": starts, "dur": durs})
    df_c["bucket"] = df_c["start"].dt.to_period(freq).dt.to_timestamp()
    s = df_c.groupby("bucket")["dur"].mean()
    return s.astype("float64").rename(f"cycle_time_{unit}")


def dfg_edge_series(df: pd.DataFrame, top_k: int = 5, freq: str = "W") -> pd.DataFrame:
    """
    Для top-K по частоте рёбер строит временные ряды частот в каждом периоде.
    Возвращает DataFrame [time x edge].
    """
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    df["next_activity"] = df.groupby("case_id")["activity"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    edges = df.dropna(subset=["next_activity"]).copy()
    edges["edge"] = edges["activity"].astype(str) + "→" + edges["next_activity"].astype(str)

    counts = edges["edge"].value_counts().head(top_k).index.tolist()
    edges = edges[edges["edge"].isin(counts)].copy()
    edges["bucket"] = edges["timestamp"].dt.to_period(freq).dt.to_timestamp()
    pivot = edges.groupby(["bucket", "edge"]).size().unstack(fill_value=0).sort_index()
    # выровнять по топ-К порядку
    pivot = pivot.reindex(columns=counts, fill_value=0)
    return pivot.astype("float64")


# ============================================================
#                           SUPERVISED FRAMING
# ============================================================


@dataclass
class SeriesSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    series: pd.Series
    test_horizon: int


def make_train_test(
    series: pd.Series, test_horizon: int = 8
) -> SeriesSplit:
    """Простое train-test разделение по времени: последние test_horizon точек в test."""
    n = len(series)
    if test_horizon >= n - 4:
        test_horizon = max(1, n // 4)
    train_idx = np.arange(0, n - test_horizon)
    test_idx = np.arange(n - test_horizon, n)
    return SeriesSplit(train_idx=train_idx, test_idx=test_idx, series=series, test_horizon=test_horizon)


def make_lag_features(
    series: pd.Series,
    lags: Tuple[int, ...] = (1, 2, 3, 4, 8, 12),
    add_calendar: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.astype("float64")}, index=series.index)
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
    df["roll_mean_4"] = df["y"].shift(1).rolling(4).mean()
    df["roll_std_4"] = df["y"].shift(1).rolling(4).std()
    if add_calendar:
        idx = df.index
        df["month"] = idx.month
        df["weekofyear"] = idx.isocalendar().week.astype("int32").to_numpy()
        df["quarter"] = idx.quarter
    df = df.dropna()
    return df


# ============================================================
#                           FORECASTERS
# ============================================================


def forecast_naive_last(series: pd.Series, split: SeriesSplit) -> np.ndarray:
    """Predict y_t = y_{t-1}."""
    y = series.to_numpy(dtype="float64")
    return y[split.test_idx - 1]


def forecast_seasonal_naive(series: pd.Series, split: SeriesSplit, seasonal_lag: int = 52) -> np.ndarray:
    y = series.to_numpy(dtype="float64")
    out = []
    for i in split.test_idx:
        j = i - seasonal_lag
        if j < 0:
            j = max(0, i - 1)
        out.append(y[j])
    return np.asarray(out, dtype="float64")


def forecast_moving_average(series: pd.Series, split: SeriesSplit, window: int = 4) -> np.ndarray:
    y = series.to_numpy(dtype="float64")
    out = []
    for i in split.test_idx:
        lo = max(0, i - window)
        out.append(y[lo:i].mean() if i > lo else y[max(0, i - 1)])
    return np.asarray(out, dtype="float64")


def forecast_ridge_lags(series: pd.Series, split: SeriesSplit) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feat = make_lag_features(series)
    train_X, train_y, test_X = _split_supervised(feat, split)
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X)
    test_X_s = scaler.transform(test_X)
    model = Ridge(alpha=1.0)
    model.fit(train_X_s, train_y)
    return model.predict(test_X_s)


def forecast_xgb_lags(series: pd.Series, split: SeriesSplit) -> np.ndarray:
    import xgboost as xgb

    feat = make_lag_features(series)
    train_X, train_y, test_X = _split_supervised(feat, split)
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_X.to_numpy(dtype="float32"), train_y)
    return model.predict(test_X.to_numpy(dtype="float32"))


def _split_supervised(
    feat: pd.DataFrame, split: SeriesSplit
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Из feat выбираем train/test по индексам split.series."""
    test_index = split.series.index[split.test_idx]
    train_mask = ~feat.index.isin(test_index)
    test_mask = feat.index.isin(test_index)
    train_X = feat.loc[train_mask].drop(columns=["y"])
    train_y = feat.loc[train_mask, "y"].to_numpy(dtype="float64")
    test_X = feat.loc[test_mask].drop(columns=["y"])
    return train_X, train_y, test_X


# ============================================================
#                           TIMESFM-LITE TRANSFORMER
# ============================================================


def forecast_chronos_zero_shot(
    series: pd.Series,
    split: SeriesSplit,
    model_id: str = "amazon/chronos-bolt-tiny",
    num_samples: int = 20,
) -> np.ndarray:
    """
    Zero-shot foundation-model прогноз через Amazon Chronos (Bolt). Один из
    нескольких real foundation models, доступных на CPU. Не требует никакого
    обучения - только pretrained веса.

    На каждой test-точке делается one-step-ahead forecast по всему
    предшествующему контексту. Это медленнее, чем lag-based xgb, но даёт
    честный foundation-model baseline.
    """
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
    y = series.to_numpy(dtype="float32")
    preds = []
    for t in split.test_idx:
        ctx = torch.tensor(y[:t], dtype=torch.float32)
        forecast = pipe.predict(context=ctx, prediction_length=1)
        # forecast shape: [1, prediction_length] for bolt models (точечный квантиль 0.5),
        # либо [1, num_samples, prediction_length] для samples-based.
        arr = forecast.detach().cpu().numpy()
        if arr.ndim == 3:
            point = float(np.median(arr[0, :, 0]))
        else:
            point = float(arr[0, 0])
        preds.append(point)
    return np.asarray(preds, dtype="float64")


def forecast_timesfm_lite(
    series: pd.Series,
    split: SeriesSplit,
    context_len: int = 24,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 80,
    lr: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    Лёгкий аналог TimesFM (Lim et al., 2024, ICML): encoder-only Transformer,
    обучаемый на (context_len, 1) → 1.
    Принимает stationary deltas (y_t - y_{t-1}) для устойчивости.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    y = series.to_numpy(dtype="float64")
    if len(y) < context_len + 4:
        # fallback: naive
        return forecast_naive_last(series, split)

    # Дельты
    dy = np.diff(y, prepend=y[0]).astype("float32")
    mu = float(dy[: split.test_idx[0]].mean())
    sd = float(dy[: split.test_idx[0]].std()) or 1.0
    dy_norm = (dy - mu) / sd

    # Сборка train-обучающих окон
    train_end = split.test_idx[0]
    Xs, ys = [], []
    for t in range(context_len, train_end):
        Xs.append(dy_norm[t - context_len : t])
        ys.append(dy_norm[t])
    if not Xs:
        return forecast_naive_last(series, split)
    Xs = np.stack(Xs).astype("float32")  # [N, C]
    ys = np.array(ys, dtype="float32")

    # Модель
    class _TFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.zeros(1, context_len, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=0.1, batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        def forward(self, x):  # x: [B, C]
            x = x.unsqueeze(-1)  # [B, C, 1]
            x = self.proj(x) + self.pos
            x = self.enc(x)
            x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _TFM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
    yt = torch.tensor(ys, dtype=torch.float32, device=device)

    bs = min(64, len(Xs))
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xs), device=device)
        for i in range(0, len(Xs), bs):
            b = idx[i : i + bs]
            opt.zero_grad()
            out = model(Xt[b])
            loss = loss_fn(out, yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    # Прогноз: рекуррентно для каждой test-точки используем последние известные
    # значения (для honest one-step ahead). Тут чистый one-step-ahead.
    model.eval()
    preds = []
    with torch.no_grad():
        for t in split.test_idx:
            ctx = dy_norm[t - context_len : t]
            xt = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
            d_pred_norm = float(model(xt).item())
            d_pred = d_pred_norm * sd + mu
            preds.append(y[t - 1] + d_pred)
    return np.asarray(preds, dtype="float64")


# ============================================================
#                           HIGH-LEVEL RUNNER
# ============================================================


def run_forecast_suite(
    series: pd.Series, test_horizon: int = 8, seasonal_lag: int = 52,
) -> Dict[str, Dict]:
    """Прогон стандартного набора моделей. Возвращает {model: metrics}."""
    split = make_train_test(series, test_horizon=test_horizon)
    y_true = series.to_numpy(dtype="float64")[split.test_idx]
    out: Dict[str, Dict] = {}

    out["naive_last"] = ts_metrics(y_true, forecast_naive_last(series, split))
    out["seasonal_naive"] = ts_metrics(y_true, forecast_seasonal_naive(series, split, seasonal_lag=seasonal_lag))
    out["moving_avg"] = ts_metrics(y_true, forecast_moving_average(series, split, window=4))
    try:
        out["ridge_lags"] = ts_metrics(y_true, forecast_ridge_lags(series, split))
    except Exception as e:
        out["ridge_lags"] = {"error": str(e)}
    try:
        out["xgb_lags"] = ts_metrics(y_true, forecast_xgb_lags(series, split))
    except Exception as e:
        out["xgb_lags"] = {"error": str(e)}
    try:
        out["timesfm_lite"] = ts_metrics(y_true, forecast_timesfm_lite(series, split))
    except Exception as e:
        out["timesfm_lite"] = {"error": str(e)}
    try:
        out["chronos_zero_shot"] = ts_metrics(y_true, forecast_chronos_zero_shot(series, split))
    except Exception as e:
        out["chronos_zero_shot"] = {"error": str(e)}
    return out
