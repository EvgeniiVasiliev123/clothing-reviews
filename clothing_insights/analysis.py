"""Пять вопросов владельца небольшого производства -> пять таблиц с результатами."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats
from .themes import LABELS, PRODUCT_THEMES


def class_summary(df: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """Q1. Где клиенты менее довольны? Доля «рекомендую» по классам изделий против остальных."""
    d = df[df["cls"].notna()]
    total_x, total_n = int(d["recommended"].sum()), len(d)
    rows = []
    for name, g in d.groupby("cls"):
        n, x = len(g), int(g["recommended"].sum())
        if n < min_n:
            continue
        lo, hi = stats.wilson_ci(x, n)
        rest_rate = (total_x - x) / (total_n - n)
        rows.append({"class": name, "n": n, "recommend_rate": x / n, "ci_low": lo, "ci_high": hi,
                     "diff_vs_rest": x / n - rest_rate, "low_rating_share": float((g["rating"] <= 2).mean()),
                     "p_value": stats.two_prop_z(x, n, total_x - x, total_n - n)})
    out = pd.DataFrame(rows).sort_values("recommend_rate").reset_index(drop=True)
    out["p_holm"] = stats.holm_adjust(out["p_value"])
    out["significant"] = out["p_holm"] < 0.05
    return out


def theme_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Q2. Почему клиенты не рекомендуют? Относительный риск «не рекомендую» при упоминании темы."""
    t = df[df["has_text"]]
    notrec, rec = t[t["recommended"] == 0], t[t["recommended"] == 1]
    rows = []
    for theme in list(LABELS) + ["any_product_theme"]:
        has = t[t[theme]]
        non = t[~t[theme]]
        rr = stats.relative_risk(int((has["recommended"] == 0).sum()), len(has),
                                 int((non["recommended"] == 0).sum()), len(non))
        rows.append({"theme": theme, "label": LABELS.get(theme, "Любая из «продуктовых» тем"),
                     "reviews": len(has), "share_all": len(has) / len(t),
                     "share_not_recommended": float(notrec[theme].mean()), "share_recommended": float(rec[theme].mean()),
                     "rr": rr["rr"], "rr_low": rr["ci"][0], "rr_high": rr["ci"][1], "p_value": rr["p_value"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = stats.holm_adjust(out["p_value"].fillna(1.0))
    return out.sort_values("rr", ascending=False).reset_index(drop=True)


def sizing_by_class(df: pd.DataFrame, min_n: int = 300) -> pd.DataFrame:
    """Q3. В какую сторону «уезжает» размер: упоминания маломерит / большемерит по классам."""
    t = df[df["has_text"] & df["cls"].notna()]
    rows = []
    for name, g in t.groupby("cls"):
        if len(g) < min_n:
            continue
        rows.append({"class": name, "reviews": len(g), "undersized_share": float(g["undersized"].mean()),
                     "oversized_share": float(g["oversized"].mean())})
    out = pd.DataFrame(rows)
    out["net_runs_small"] = out["undersized_share"] - out["oversized_share"]
    return out.sort_values("net_runs_small", ascending=False).reset_index(drop=True)


def age_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Q4. Для кого изделие: удовлетворённость по возрастным группам."""
    rows = []
    for band, g in df.groupby("age_band", observed=True):
        n, x = len(g), int(g["recommended"].sum())
        lo, hi = stats.wilson_ci(x, n)
        tg = g[g["has_text"]]
        rows.append({"age_band": str(band), "n": n, "recommend_rate": x / n, "ci_low": lo, "ci_high": hi,
                     "fabric_quality_share": float(tg["fabric_quality"].mean()),
                     "shape_unflattering_share": float(tg["shape_unflattering"].mean())})
    out = pd.DataFrame(rows)
    table = pd.crosstab(df["age_band"], df["recommended"]).to_numpy()
    chi2, p = sps.chi2_contingency(table, correction=False)[:2]
    return out, {"chi2": float(chi2), "p_value": float(p), "cramers_v": stats.cramers_v(table)}


def problem_items(df: pd.DataFrame, min_n: int = 15, threshold: float = 0.75) -> tuple[pd.DataFrame, dict]:
    """Q5. Какие модели «подводят»: байесовское сжатие, чтобы не ловить шум на малых выборках."""
    g = df.groupby("item_id").agg(n=("recommended", "size"), recommended=("recommended", "sum"),
                                  cls=("cls", lambda s: s.mode().iloc[0] if s.notna().any() else "—")).reset_index()
    g = g[g["n"] >= min_n].copy()
    a, b = stats.fit_beta_prior(g["recommended"].to_numpy(), g["n"].to_numpy())
    g["raw_rate"] = g["recommended"] / g["n"]
    g["shrunk_rate"] = (a + g["recommended"]) / (a + b + g["n"])
    g["p_below_threshold"] = sps.beta.cdf(threshold, a + g["recommended"], b + g["n"] - g["recommended"])
    # частые темы жалоб у «не рекомендую»-отзывов модели
    neg = df[(df["recommended"] == 0) & df["has_text"]]
    tops = {}
    for item_id, grp in neg.groupby("item_id"):
        counts = {th: int(grp[th].sum()) for th in PRODUCT_THEMES + ["undersized", "oversized"]}
        best = [LABELS[k] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:2] if v > 0]
        tops[item_id] = ", ".join(best)
    g["top_complaints"] = g["item_id"].map(tops).fillna("")
    g = g.sort_values(["p_below_threshold", "shrunk_rate"], ascending=[False, True]).reset_index(drop=True)
    meta = {"prior_a": a, "prior_b": b, "items_analyzed": len(g), "threshold": threshold,
            "overdispersion": stats.overdispersion_test(g["recommended"].to_numpy(), g["n"].to_numpy()),
            "flagged_p_gt_0_5": int((g["p_below_threshold"] > 0.5).sum())}
    return g, meta
