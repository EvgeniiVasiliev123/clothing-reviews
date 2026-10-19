from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def class_satisfaction(cls_df, overall: float, out: Path) -> str:
    d = cls_df.sort_values("recommend_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(d) + 1.4))
    y = np.arange(len(d))
    colors = ["#C44E52" if s else "#4C72B0" for s in d["significant"]]
    ax.hlines(y, d["ci_low"] * 100, d["ci_high"] * 100, color="gray")
    ax.scatter(d["recommend_rate"] * 100, y, c=colors, zorder=3)
    ax.axvline(overall * 100, color="black", ls="--", lw=1)
    ax.set_yticks(y, [f"{c} (n={n})" for c, n in zip(d["class"], d["n"])])
    ax.invert_yaxis()
    ax.set_xlabel("Доля «рекомендую», % (95% ДИ). Красный — значимо отличается от остальных")
    ax.set_title("Удовлетворённость по классам изделий")
    return _save(fig, out, "class_satisfaction.png")


def theme_lift_plot(lift_df, out: Path) -> str:
    d = lift_df[lift_df["theme"] != "any_product_theme"].dropna(subset=["rr"]).sort_values("rr")
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.hlines(y, d["rr_low"], d["rr_high"], color="gray")
    ax.scatter(d["rr"], y, color="#DD8452", zorder=3)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{l} ({s:.1%} отзывов)" for l, s in zip(d["label"], d["share_all"])])
    ax.set_xlabel("Во сколько раз чаще «не рекомендую», если тема упомянута (RR, 95% ДИ)")
    ax.set_title("Какие темы связаны с неудовлетворённостью")
    return _save(fig, out, "theme_lift.png")


def sizing_plot(sz, out: Path) -> str:
    d = sz.sort_values("net_runs_small")
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["undersized_share"] * 100, height=0.4, label="Маломерит", color="#4C72B0")
    ax.barh(y + 0.2, d["oversized_share"] * 100, height=0.4, label="Большемерит", color="#DD8452")
    ax.set_yticks(y, d["class"])
    ax.set_xlabel("% отзывов с упоминанием")
    ax.set_title("Куда «уезжает» размер по классам")
    ax.legend()
    return _save(fig, out, "sizing.png")


def age_plot(age_df, overall: float, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(age_df))
    err = [(age_df["recommend_rate"] - age_df["ci_low"]) * 100, (age_df["ci_high"] - age_df["recommend_rate"]) * 100]
    ax.errorbar(x, age_df["recommend_rate"] * 100, yerr=err, fmt="o", color="#55A868", capsize=4)
    ax.axhline(overall * 100, color="black", ls="--", lw=1)
    ax.set_xticks(x, age_df["age_band"])
    ax.set_ylabel("Доля «рекомендую», %")
    ax.set_xlabel("Возраст автора отзыва")
    ax.set_title("Удовлетворённость по возрасту")
    return _save(fig, out, "age.png")


def items_plot(items_df, threshold: float, out: Path, top: int = 12) -> str:
    d = items_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.scatter(d["raw_rate"] * 100, y, facecolors="none", edgecolors="gray", label="сырая доля")
    ax.scatter(d["shrunk_rate"] * 100, y, color="#C44E52", label="с поправкой на малую выборку")
    ax.axvline(threshold * 100, color="black", ls="--", lw=1)
    ax.set_yticks(y, [f"ID {i} (n={n})" for i, n in zip(d["item_id"], d["n"])])
    ax.set_xlabel("Доля «рекомендую», %")
    ax.set_title(f"Модели с наибольшей вероятностью оказаться ниже {threshold:.0%}")
    ax.legend()
    return _save(fig, out, "problem_items.png")
