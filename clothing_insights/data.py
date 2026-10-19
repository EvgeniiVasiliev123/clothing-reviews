"""Загрузка, очистка и проверка качества данных."""
from __future__ import annotations

import numpy as np
import pandas as pd

RENAME = {"Clothing ID": "item_id", "Age": "age", "Title": "title", "Review Text": "text", "Rating": "rating",
          "Recommended IND": "recommended", "Positive Feedback Count": "helpful",
          "Division Name": "division", "Department Name": "department", "Class Name": "cls"}
AGE_BINS = [0, 29, 39, 49, 59, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def load(path: str) -> tuple[pd.DataFrame, dict]:
    """Читает CSV, переименовывает колонки, удаляет полные дубли. Возвращает (данные, отчёт о чистке)."""
    raw = pd.read_csv(path)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    missing = set(RENAME) - set(raw.columns)
    if missing:
        raise ValueError(f"В файле нет колонок: {sorted(missing)}")
    df = raw.rename(columns=RENAME)[list(RENAME.values())]
    df = _clean(df)
    return df.df, {**df.report, "rows_raw": int(len(raw))}


class _Cleaned:
    def __init__(self, df: pd.DataFrame, report: dict):
        self.df, self.report = df, report


def _clean(df: pd.DataFrame) -> _Cleaned:
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dups = n0 - len(df)
    df["recommended"] = df["recommended"].astype(int)
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    df["has_text"] = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
    report = {
        "rows": int(len(df)), "duplicates_removed": int(n_dups),
        "missing_text": int((~df["has_text"]).sum()), "missing_class": int(df["cls"].isna().sum()),
        "invalid_rating": int((~df["rating"].between(1, 5)).sum()),
        "recommended_rate": float(df["recommended"].mean()),
        "items": int(df["item_id"].nunique()),
        # логическая проверка: рейтинг 5, но «не рекомендую» (и наоборот) — шум разметки
        "rating5_not_recommended": int(((df["rating"] == 5) & (df["recommended"] == 0)).sum()),
        "rating1_recommended": int(((df["rating"] == 1) & (df["recommended"] == 1)).sum()),
    }
    return _Cleaned(df, report)


def make_demo(n: int = 6000, seed: int = 11) -> pd.DataFrame:
    """Синтетические отзывы в формате датасета — для запуска и CI без скачивания файла."""
    rng = np.random.default_rng(seed)
    classes = ["Dresses", "Knits", "Blouses", "Pants", "Jeans", "Skirts"]
    cls = rng.choice(classes, n, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    item = rng.integers(1, 120, n)
    bad_items = set(range(1, 8))
    age = np.clip(rng.normal(43, 12, n).round(), 18, 90).astype(int)
    p_rec = np.where(np.isin(item, list(bad_items)), 0.55, 0.85)
    rec = (rng.random(n) < p_rec).astype(int)
    pos = ["love this, fits perfectly", "so soft and flattering", "great quality, will buy again", "runs small, size up, still love it"]
    neg = ["the fabric is thin and cheap", "looks boxy and unflattering", "too short on me", "returned it, not worth the price",
           "fabric is itchy", "not as pictured"]
    text = [(rng.choice(neg if r == 0 and rng.random() < 0.8 else pos)) for r in rec]
    rating = np.where(rec == 1, rng.choice([4, 5], n, p=[0.3, 0.7]), rng.choice([1, 2, 3], n))
    return pd.DataFrame({"item_id": item, "age": age, "title": "t", "text": text, "rating": rating, "recommended": rec,
                         "helpful": 0, "division": "General", "department": "Tops", "cls": cls})
