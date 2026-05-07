# core/features.py
"""
Построение фичей для задачи Remaining Time Prediction (RTP).

Никаких утечек: при предсказании из префикса длины L используются только
данные событий 1..L и неизменные case-level атрибуты.

Два формата:
  1) prefix-таблица: строка на префикс с числовыми и категориальными фичами,
     удобна для tabular ML (RF, XGBoost, наивный baseline).
  2) sequence arrays: токены активности + параллельный тензор числовых
     per-step фичей; для LSTM/Transformer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SECONDS_IN = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}


# ============================================================
#                         БАЗОВЫЕ ВРЕМЕННЫЕ ВЕЛИЧИНЫ
# ============================================================


def add_inter_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет inter-case фичи, известные на момент события:
      - active_cases: число кейсов, открытых на момент события
        (cases с min_ts <= event_ts <= max_ts).
    Реализация O(N log N) через сортировку: считаем кумулятивный поток
    стартов и завершений по timestamp.

    Это аналог workload-фичи из PGTNet (Elyasi 2024) и из
    Verenich 2018 для inter-case patterns.
    """
    df = df.copy().sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    case_min = df.groupby("case_id")["timestamp"].transform("min")
    case_max = df.groupby("case_id")["timestamp"].transform("max")

    # event-based active count
    starts = df.groupby("case_id")["timestamp"].min().reset_index().rename(columns={"timestamp": "ts"})
    starts["delta"] = 1
    ends = df.groupby("case_id")["timestamp"].max().reset_index().rename(columns={"timestamp": "ts"})
    ends["delta"] = -1
    flow = pd.concat([starts[["ts", "delta"]], ends[["ts", "delta"]]]).sort_values("ts")
    # для одинаковых ts кейсы открываются раньше, чем закрываются (стабильность)
    flow["delta"] = flow["delta"].astype("int32")
    flow["active"] = flow["delta"].cumsum()
    flow_unique = flow.groupby("ts", as_index=False)["active"].max().sort_values("ts")

    # для каждого события - active_cases в момент его timestamp
    ts_sorted = flow_unique["ts"].to_numpy()
    active_sorted = flow_unique["active"].to_numpy()
    df_ts = df["timestamp"].to_numpy()
    # binary search: searchsorted right - 1
    idx = np.searchsorted(ts_sorted, df_ts, side="right") - 1
    idx = np.clip(idx, 0, len(active_sorted) - 1)
    df["active_cases"] = active_sorted[idx].astype("int32")
    return df


def add_event_temporal(df: pd.DataFrame, target_unit: str = "hours") -> pd.DataFrame:
    """Добавляет per-event величины: remaining_time, elapsed_time, dt_since_prev,
    prefix_len, hour, weekday, month."""
    if target_unit not in SECONDS_IN:
        raise ValueError(f"target_unit must be in {list(SECONDS_IN)}")
    factor = SECONDS_IN[target_unit]

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("case_id", sort=False)
    case_start = grp["timestamp"].transform("min")
    case_end = grp["timestamp"].transform("max")

    df["elapsed_time"] = (df["timestamp"] - case_start).dt.total_seconds() / factor
    df["remaining_time"] = (case_end - df["timestamp"]).dt.total_seconds() / factor
    df["prefix_len"] = grp.cumcount() + 1

    dt_prev = grp["timestamp"].diff().dt.total_seconds() / factor
    df["dt_since_prev"] = dt_prev.fillna(0.0)

    df["hour"] = df["timestamp"].dt.hour.astype("int16")
    df["weekday"] = df["timestamp"].dt.weekday.astype("int16")
    df["month"] = df["timestamp"].dt.month.astype("int16")

    # cyclic кодирование
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0).astype("float32")

    return df


# ============================================================
#                         CASE-BASED SPLIT
# ============================================================


@dataclass
class CaseSplit:
    train_cases: np.ndarray
    val_cases: np.ndarray
    test_cases: np.ndarray


def time_based_split(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
) -> CaseSplit:
    """
    Time-based split (Weytjens & De Weerdt 2022):
      - сортируем кейсы по дате старта,
      - первые train%  это train,
      - следующие val% это val,
      - остаток это test.
    Это жёстче case-based и выявляет drift.
    """
    starts = df.groupby("case_id")["timestamp"].min().sort_values()
    cases = starts.index.to_numpy()
    n = len(cases)
    n_tr = int(n * train)
    n_va = int(n * val)
    return CaseSplit(
        train_cases=cases[:n_tr],
        val_cases=cases[n_tr : n_tr + n_va],
        test_cases=cases[n_tr + n_va :],
    )


def case_based_split(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> CaseSplit:
    assert abs(train + val + test - 1.0) < 1e-6
    cases = df["case_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    n = len(cases)
    n_tr = int(n * train)
    n_va = int(n * val)
    return CaseSplit(
        train_cases=cases[:n_tr],
        val_cases=cases[n_tr : n_tr + n_va],
        test_cases=cases[n_tr + n_va :],
    )


# ============================================================
#                         PREFIX TABLE (для tabular ML)
# ============================================================


def build_prefix_table(
    df: pd.DataFrame,
    max_prefix_len: int = 40,
    target_unit: str = "hours",
    case_attr_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Возвращает DataFrame, где каждая строка - один префикс.

    Состав фичей (всё доступно к моменту префикса, никакой утечки):
      - prefix_len, elapsed_time, dt_since_prev (последнего события)
      - mean_dt, max_dt в префиксе
      - n_unique_acts, max_act_repeats
      - last_activity, prev_activity, prev2_activity, last_resource
      - hour, weekday последнего события (как числовые)
      - hour_sin/cos, dow_sin/cos
      - case-level атрибуты (если переданы)
    Таргет: remaining_time (last event префикса).
    """
    case_attr_cols = case_attr_cols or []
    df = add_event_temporal(df, target_unit=target_unit)
    df = df[df["prefix_len"] <= max_prefix_len].copy()
    df["activity"] = df["activity"].astype(str)
    if "resource" in df.columns:
        df["resource"] = df["resource"].astype(str)
    else:
        df["resource"] = "<NA>"

    rows: List[Dict] = []
    for cid, g in df.groupby("case_id", sort=False):
        acts = g["activity"].to_numpy()
        rsrc = g["resource"].to_numpy()
        rt = g["remaining_time"].to_numpy()
        et = g["elapsed_time"].to_numpy()
        dtp = g["dt_since_prev"].to_numpy()
        hour = g["hour"].to_numpy()
        weekday = g["weekday"].to_numpy()
        hour_sin = g["hour_sin"].to_numpy()
        hour_cos = g["hour_cos"].to_numpy()
        dow_sin = g["dow_sin"].to_numpy()
        dow_cos = g["dow_cos"].to_numpy()

        # case-level берем из первой строки (все равны для всего кейса)
        first = g.iloc[0]
        case_extras = {c: first[c] for c in case_attr_cols if c in g.columns}

        L = len(acts)
        for k in range(1, L + 1):
            row = {
                "case_id": cid,
                "prefix_len": k,
                "elapsed_time": float(et[k - 1]),
                "dt_since_prev": float(dtp[k - 1]),
                "mean_dt": float(np.mean(dtp[:k])),
                "max_dt": float(np.max(dtp[:k])),
                "n_unique_acts": int(len(np.unique(acts[:k]))),
                "max_act_repeats": int(
                    np.max(np.unique(acts[:k], return_counts=True)[1])
                ),
                "last_activity": str(acts[k - 1]),
                "prev_activity": str(acts[k - 2]) if k >= 2 else "<START>",
                "prev2_activity": str(acts[k - 3]) if k >= 3 else "<START>",
                "last_resource": str(rsrc[k - 1]),
                "hour": int(hour[k - 1]),
                "weekday": int(weekday[k - 1]),
                "hour_sin": float(hour_sin[k - 1]),
                "hour_cos": float(hour_cos[k - 1]),
                "dow_sin": float(dow_sin[k - 1]),
                "dow_cos": float(dow_cos[k - 1]),
                "target": float(rt[k - 1]),
            }
            row.update(case_extras)
            rows.append(row)

    table = pd.DataFrame(rows)
    return table


# ============================================================
#                         ENCODE PREFIX TABLE → ARRAYS
# ============================================================


@dataclass
class TabularEncoder:
    """Замораживает категории, среднее/std по числовым на train, применяет на val/test."""

    cat_cols: List[str]
    num_cols: List[str]
    top_k_per_cat: int = 30
    cat_vocabs: Dict[str, List[str]] = None
    num_means: Dict[str, float] = None
    num_stds: Dict[str, float] = None

    def fit(self, df: pd.DataFrame) -> None:
        self.cat_vocabs = {}
        for c in self.cat_cols:
            top = df[c].astype(str).value_counts().head(self.top_k_per_cat).index.tolist()
            self.cat_vocabs[c] = top

        self.num_means = {}
        self.num_stds = {}
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            mu = float(np.nanmean(v)) if v.notna().any() else 0.0
            sd = float(np.nanstd(v)) if v.notna().any() else 1.0
            self.num_means[c] = mu
            self.num_stds[c] = sd if sd > 1e-8 else 1.0

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        feature_blocks: List[np.ndarray] = []
        feature_names: List[str] = []

        # числовые: стандартизация + бинарный NaN-флаг
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
            isnan = np.isnan(v)
            v_filled = np.where(isnan, self.num_means[c], v)
            v_std = (v_filled - self.num_means[c]) / self.num_stds[c]
            feature_blocks.append(v_std.astype("float32").reshape(-1, 1))
            feature_names.append(f"num__{c}")
            feature_blocks.append(isnan.astype("float32").reshape(-1, 1))
            feature_names.append(f"num__{c}__isnan")

        # категориальные: one-hot top-K + "<OTHER>"
        for c in self.cat_cols:
            vocab = self.cat_vocabs[c]
            v = df[c].astype(str).to_numpy()
            for cat in vocab:
                feature_blocks.append((v == cat).astype("float32").reshape(-1, 1))
                feature_names.append(f"cat__{c}__{cat}")
            other_mask = ~np.isin(v, vocab)
            feature_blocks.append(other_mask.astype("float32").reshape(-1, 1))
            feature_names.append(f"cat__{c}__<OTHER>")

        X = np.hstack(feature_blocks).astype("float32")
        return X, feature_names


# ============================================================
#                         SEQUENCE ARRAYS (для LSTM/Transformer)
# ============================================================


@dataclass
class SequenceMeta:
    act2id: Dict[str, int]
    id2act: Dict[int, str]
    max_len: int
    target_unit: str
    numeric_features: List[str]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    target_mean: float
    target_std: float


def build_nap_arrays(
    df: pd.DataFrame,
    max_prefix_len: int = 40,
    target_unit: str = "hours",
    train_cases: Optional[np.ndarray] = None,
    fit_scaler: bool = True,
    meta: Optional["SequenceMeta"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "SequenceMeta"]:
    """
    Аналог build_sequence_arrays, но таргет - id следующей активности.

    Кодирование:
      - 0 = padding (никогда не таргет)
      - 1..V = activity IDs
      - V+1 = END (нет следующего события - префикс закончил кейс)
    """
    df = add_event_temporal(df, target_unit=target_unit)
    df = df[df["prefix_len"] <= max_prefix_len].copy()
    df["activity"] = df["activity"].astype(str)

    numeric_features = ["elapsed_time", "dt_since_prev", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    if fit_scaler:
        if train_cases is not None:
            train_df = df[df["case_id"].isin(train_cases)]
        else:
            train_df = df
        unique_acts = sorted(train_df["activity"].unique().tolist())
        act2id = {a: i + 1 for i, a in enumerate(unique_acts)}
        id2act = {i: a for a, i in act2id.items()}

        num_arr = train_df[numeric_features].to_numpy(dtype="float32")
        num_mean = num_arr.mean(axis=0)
        num_std = num_arr.std(axis=0)
        num_std = np.where(num_std < 1e-6, 1.0, num_std)

        meta = SequenceMeta(
            act2id=act2id, id2act=id2act, max_len=max_prefix_len,
            target_unit=target_unit, numeric_features=numeric_features,
            numeric_mean=num_mean, numeric_std=num_std,
            target_mean=0.0, target_std=1.0,  # для NAP не используются
        )
    elif meta is None:
        raise ValueError("при fit_scaler=False передайте meta")

    act2id = meta.act2id
    max_len = meta.max_len
    num_mean = meta.numeric_mean
    num_std = meta.numeric_std
    V = len(act2id)
    END_TOKEN = V + 1

    X_tok_list, X_num_list, y_list, lens_list = [], [], [], []
    for _, g in df.groupby("case_id", sort=False):
        acts = g["activity"].to_numpy()
        ids_full = np.array([act2id.get(a, 0) for a in acts], dtype="int64")
        nums_full = g[numeric_features].to_numpy(dtype="float32")
        nums_full = (nums_full - num_mean) / num_std

        L = len(acts)
        for k in range(1, L + 1):
            tok_padded = np.zeros(max_len, dtype="int64")
            num_padded = np.zeros((max_len, len(numeric_features)), dtype="float32")
            tok_padded[:k] = ids_full[:k]
            num_padded[:k] = nums_full[:k]
            X_tok_list.append(tok_padded)
            X_num_list.append(num_padded)
            if k < L:
                y_list.append(int(ids_full[k]))  # next activity
            else:
                y_list.append(END_TOKEN)
            lens_list.append(k)

    X_tok = np.stack(X_tok_list, axis=0)
    X_num = np.stack(X_num_list, axis=0)
    y = np.array(y_list, dtype="int64")
    lens = np.array(lens_list, dtype="int64")
    return X_tok, X_num, y, lens, meta


def build_sequence_arrays(
    df: pd.DataFrame,
    max_prefix_len: int = 40,
    target_unit: str = "hours",
    train_cases: Optional[np.ndarray] = None,
    fit_scaler: bool = True,
    meta: Optional[SequenceMeta] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SequenceMeta]:
    """
    Возвращает X_tok (int64 [N,T]), X_num (float32 [N,T,F]), y (float32 [N]),
    lengths (int64 [N]), meta.

    Если fit_scaler=True - словарь активности и нормализация по train_cases (или
    по всем данным, если train_cases не задан). Иначе используются переданные meta.
    """
    df = add_event_temporal(df, target_unit=target_unit)
    df = df[df["prefix_len"] <= max_prefix_len].copy()
    df["activity"] = df["activity"].astype(str)

    numeric_features = ["elapsed_time", "dt_since_prev", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    if fit_scaler:
        if train_cases is not None:
            train_df = df[df["case_id"].isin(train_cases)]
        else:
            train_df = df
        unique_acts = sorted(train_df["activity"].unique().tolist())
        act2id = {a: i + 1 for i, a in enumerate(unique_acts)}  # 0 = pad
        id2act = {i: a for a, i in act2id.items()}

        num_arr = train_df[numeric_features].to_numpy(dtype="float32")
        num_mean = num_arr.mean(axis=0)
        num_std = num_arr.std(axis=0)
        num_std = np.where(num_std < 1e-6, 1.0, num_std)

        t_train = train_df["remaining_time"].to_numpy(dtype="float32")
        t_mean = float(t_train.mean())
        t_std = float(t_train.std())
        t_std = t_std if t_std > 1e-6 else 1.0

        meta = SequenceMeta(
            act2id=act2id,
            id2act=id2act,
            max_len=max_prefix_len,
            target_unit=target_unit,
            numeric_features=numeric_features,
            numeric_mean=num_mean,
            numeric_std=num_std,
            target_mean=t_mean,
            target_std=t_std,
        )
    else:
        if meta is None:
            raise ValueError("При fit_scaler=False нужно передать meta")

    act2id = meta.act2id
    max_len = meta.max_len
    num_mean = meta.numeric_mean
    num_std = meta.numeric_std

    # сборка по кейсам
    X_tok_list: List[np.ndarray] = []
    X_num_list: List[np.ndarray] = []
    y_list: List[float] = []
    lengths: List[int] = []

    for _, g in df.groupby("case_id", sort=False):
        acts = g["activity"].to_numpy()
        ids_full = np.array([act2id.get(a, 0) for a in acts], dtype="int64")
        nums_full = g[numeric_features].to_numpy(dtype="float32")
        nums_full = (nums_full - num_mean) / num_std
        rem = g["remaining_time"].to_numpy(dtype="float32")

        L = len(acts)
        for k in range(1, L + 1):
            tok_pref = ids_full[:k]
            num_pref = nums_full[:k]
            tok_padded = np.zeros(max_len, dtype="int64")
            num_padded = np.zeros((max_len, len(numeric_features)), dtype="float32")
            tok_padded[:k] = tok_pref
            num_padded[:k] = num_pref
            X_tok_list.append(tok_padded)
            X_num_list.append(num_padded)
            y_list.append(rem[k - 1])
            lengths.append(k)

    X_tok = np.stack(X_tok_list, axis=0)
    X_num = np.stack(X_num_list, axis=0)
    y = np.array(y_list, dtype="float32")
    lengths_arr = np.array(lengths, dtype="int64")

    return X_tok, X_num, y, lengths_arr, meta
