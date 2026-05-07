# core/datasets.py
"""
Реестр датасетов: единый интерфейс загрузки CSV/XES в нормализованный
DataFrame с колонками case_id, activity, timestamp, [resource], [case_attrs].

Идея простая: каждый датасет описывается DatasetSpec, по нему функция
load_dataset(name) возвращает готовый DataFrame и список case-level
числовых фичей (если их можно безопасно использовать в момент предсказания).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# корни данных
NIR_ROOT = Path(__file__).resolve().parents[1]
DIPLOMA_ROOT = NIR_ROOT.parent
DATA_ROOT_DIPLOMA = DIPLOMA_ROOT / "data"
DATA_ROOT_NIR = NIR_ROOT / "data"


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[], Tuple[pd.DataFrame, List[str]]]
    description: str = ""


# ----------------- Универсальная нормализация -----------------


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Не хватает колонок: {missing}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["case_id", "activity", "timestamp"])
    df = df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


# ----------------- Конкретные лоадеры -----------------


def _load_bpi2012() -> Tuple[pd.DataFrame, List[str]]:
    import pm4py

    path = DATA_ROOT_NIR / "BPI_Challenge_2012.xes.gz"
    log = pm4py.read_xes(str(path))
    if not isinstance(log, pd.DataFrame):
        log = pm4py.convert_to_dataframe(log)

    df = log.rename(
        columns={
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "time:timestamp": "timestamp",
            "org:resource": "resource",
        }
    )
    keep = ["case_id", "activity", "timestamp"]
    if "resource" in df.columns:
        keep.append("resource")
    if "case:AMOUNT_REQ" in df.columns:
        df["case_amount_req"] = pd.to_numeric(df["case:AMOUNT_REQ"], errors="coerce")
        keep.append("case_amount_req")
    df = df[keep]
    df = _normalize(df)
    case_attr_cols = ["case_amount_req"] if "case_amount_req" in df.columns else []
    return df, case_attr_cols


def _load_142_bz() -> Tuple[pd.DataFrame, List[str]]:
    raw = DATA_ROOT_DIPLOMA / "142_БЗ" / "raw" / "event_log_142_179.csv"
    case_table_path = DATA_ROOT_DIPLOMA / "142_БЗ" / "raw" / "case_table_142_179.csv"

    needed = [
        "ps.Идентификатор случая",
        "ps.Название события",
        "ps.Время события",
        "sj.УН инспектора",
        "sj.Регион",
        "sj.Сумма исковых требований",
        "sj.Контрольный срок завершения ПЗ",
    ]
    df = pd.read_csv(raw, sep=";", low_memory=False, usecols=needed)
    df = df.rename(
        columns={
            "ps.Идентификатор случая": "case_id",
            "ps.Название события": "activity",
            "ps.Время события": "timestamp",
            "sj.УН инспектора": "resource",
            "sj.Регион": "case_region",
            "sj.Сумма исковых требований": "case_claim_amount",
            "sj.Контрольный срок завершения ПЗ": "case_sla_deadline",
        }
    )

    df["case_region"] = pd.to_numeric(df["case_region"], errors="coerce")
    df["case_claim_amount"] = pd.to_numeric(df["case_claim_amount"], errors="coerce")
    df["case_sla_deadline"] = pd.to_datetime(df["case_sla_deadline"], errors="coerce")

    df = _normalize(df)

    # case_sla_remaining_days_at_event - сколько дней до контрольного срока на момент события.
    # Это валидная фича: на момент префикса дата SLA известна.
    sla = df["case_sla_deadline"]
    dt = (sla - df["timestamp"]).dt.total_seconds() / 86400.0
    df["case_sla_days_to_deadline"] = dt

    # Берем case-level числовые фичи: они известны на момент любого события кейса
    # (фактически они приходят из case_table и неизменны).
    df = df.drop(columns=["case_sla_deadline"])
    case_attr_cols = ["case_region", "case_claim_amount", "case_sla_days_to_deadline"]
    return df, case_attr_cols


def _load_xes_simple(name: str, fname: str) -> Tuple[pd.DataFrame, List[str]]:
    import pm4py

    path = DATA_ROOT_DIPLOMA / name / "raw" / fname
    log = pm4py.read_xes(str(path))
    if not isinstance(log, pd.DataFrame):
        log = pm4py.convert_to_dataframe(log)
    df = log.rename(
        columns={
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "time:timestamp": "timestamp",
            "org:resource": "resource",
        }
    )
    keep = ["case_id", "activity", "timestamp"]
    if "resource" in df.columns:
        keep.append("resource")
    df = df[keep]
    df = _normalize(df)
    return df, []


def _load_sepsis() -> Tuple[pd.DataFrame, List[str]]:
    return _load_xes_simple("sepsis", "sepsis.xes.gz")


def _load_bpi2017() -> Tuple[pd.DataFrame, List[str]]:
    return _load_xes_simple("bpi2017", "bpi2017.xes.gz")


REGISTRY: Dict[str, DatasetSpec] = {
    "bpi2012": DatasetSpec("bpi2012", _load_bpi2012, "BPI Challenge 2012 (loan)"),
    "142_БЗ": DatasetSpec("142_БЗ", _load_142_bz, "Внутренняя БЗ ФНС: привлечение к СО"),
    "sepsis": DatasetSpec("sepsis", _load_sepsis, "Sepsis Cases"),
    "bpi2017": DatasetSpec("bpi2017", _load_bpi2017, "BPI Challenge 2017 (loan)"),
}


def load_dataset(name: str) -> Tuple[pd.DataFrame, List[str]]:
    if name not in REGISTRY:
        raise KeyError(f"Неизвестный датасет: {name}. Доступны: {list(REGISTRY)}")
    return REGISTRY[name].loader()
