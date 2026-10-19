"""Запуск всех шагов: данные -> темы -> 5 вопросов -> таблицы, графики, report.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, plots
from .data import _clean, load, make_demo
from .themes import tag_themes

THRESHOLD = 0.75


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def run(df: pd.DataFrame, cleaning: dict) -> dict:
    df = tag_themes(df)
    overall = float(df["recommended"].mean())
    cls = analysis.class_summary(df)
    lift = analysis.theme_lift(df)
    sizing = analysis.sizing_by_class(df)
    age, age_test = analysis.age_summary(df)
    items, items_meta = analysis.problem_items(df, threshold=THRESHOLD)
    return {"df": df, "cleaning": cleaning, "overall": overall, "class": cls, "lift": lift, "sizing": sizing,
            "age": age, "age_test": age_test, "items": items, "items_meta": items_meta}


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(res: dict, figures: list[str], out: Path) -> None:
    c, lift, cls, age, items, im = res["cleaning"], res["lift"], res["class"], res["age"], res["items"], res["items_meta"]
    od = im["overdispersion"]
    cls_t = cls.assign(**{"Доля «рекомендую»": (cls["recommend_rate"] * 100).round(1).astype(str) + "%",
                          "Разница с остальными": (cls["diff_vs_rest"] * 100).round(1).map("{:+.1f} п.п.".format),
                          "p (Холм)": cls["p_holm"].map(_fmt_p)}).rename(columns={"class": "Класс", "n": "Отзывов"})
    cls_t = cls_t[["Класс", "Отзывов", "Доля «рекомендую»", "Разница с остальными", "p (Холм)"]]
    lift_t = pd.DataFrame({
        "Тема": lift["label"], "Доля отзывов": (lift["share_all"] * 100).round(1).astype(str) + "%",
        "Среди «не рекомендую»": (lift["share_not_recommended"] * 100).round(1).astype(str) + "%",
        "Среди «рекомендую»": (lift["share_recommended"] * 100).round(1).astype(str) + "%",
        "RR [95% ДИ]": [f"{r:.2f} [{lo:.2f}; {hi:.2f}]" for r, lo, hi in zip(lift["rr"], lift["rr_low"], lift["rr_high"])],
        "p (Холм)": lift["p_holm"].map(_fmt_p)})
    age_t = pd.DataFrame({"Возраст": age["age_band"], "Отзывов": age["n"],
                          "Доля «рекомендую»": (age["recommend_rate"] * 100).round(1).astype(str) + "%"})
    top = items.head(10)
    items_t = pd.DataFrame({"ID модели": top["item_id"], "Класс": top["cls"], "Отзывов": top["n"],
                            "Сырая доля": (top["raw_rate"] * 100).round(0).astype(int).astype(str) + "%",
                            "С поправкой": (top["shrunk_rate"] * 100).round(0).astype(int).astype(str) + "%",
                            f"P(< {THRESHOLD:.0%})": top["p_below_threshold"].round(2),
                            "Частые жалобы": top["top_complaints"]})
    lines = [
        "# Отчёт: что в отзывах мешает продуктам", "",
        "## Данные", "",
        f"- Отзывов после очистки: {c['rows']:,} (удалено дублей: {c['duplicates_removed']}); моделей: {c['items']:,}",
        f"- Без текста: {c['missing_text']:,}; без класса: {c['missing_class']}; "
        f"рейтинг 5, но «не рекомендую»: {c['rating5_not_recommended']}; рейтинг 1, но «рекомендую»: {c['rating1_recommended']}",
        f"- Общая доля «рекомендую»: {res['overall'] * 100:.1f}%", "",
        "## 1. Где клиенты менее довольны (классы изделий, n ≥ 100)", "", _md_table(cls_t), "",
        "## 2. Какие темы связаны с «не рекомендую»", "",
        "RR — во сколько раз чаще встречается «не рекомендую» среди отзывов, где тема упомянута, чем среди остальных. "
        "Это связь, а не причинно-следственный эффект.", "", _md_table(lift_t), "",
        "## 3. Размер", "", "См. график `sizing.png`: доли упоминаний «маломерит» и «большемерит» по классам.", "",
        "## 4. Возраст", "", _md_table(age_t), "",
        f"Хи-квадрат = {res['age_test']['chi2']:.1f}, p = {res['age_test']['p_value']:.2g}, "
        f"V Крамера = {res['age_test']['cramers_v']:.3f} (эффект малый).", "",
        f"## 5. Модели-кандидаты на разбор (n ≥ 15 отзывов, порог {THRESHOLD:.0%})", "",
        f"Проверено моделей: {im['items_analyzed']}. Различия между моделями "
        f"{'превышают' if od['p_value'] < 0.05 else 'не превышают'} случайный шум (хи-квадрат {od['chi2']:.0f} при df = {od['df']}, "
        f"p = {od['p_value']:.2g}). Моделей с вероятностью > 50% оказаться ниже порога: {im['flagged_p_gt_0_5']}.", "",
        _md_table(items_t), "",
        "## Графики", "", *[f"![{f}](figures/{f})" for f in figures], "",
        "## Ограничения", "",
        "- Данные одного магазина, женская одежда, анонимизированы: общие выводы о рынке делать нельзя.",
        "- Темы определяются словарём ключевых слов с учётом отрицаний; это приближение, возможны ошибки в обе стороны.",
        "- «Рекомендую / не рекомендую» — самооценка клиента; причины в тексте указаны не всегда.",
        "- Нет данных о продажах, возвратах, цене и себестоимости, поэтому экономический эффект оценить нельзя.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clothing_insights")
    ap.add_argument("--data", default="data/reviews.csv")
    ap.add_argument("--demo", action="store_true", help="синтетические данные вместо файла")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    if args.demo:
        cleaned = _clean(make_demo().assign(age_band=None, has_text=True)[
            ["item_id", "age", "title", "text", "rating", "recommended", "helpful", "division", "department", "cls"]])
        df, cleaning = cleaned.df, {**cleaned.report, "rows_raw": len(cleaned.df)}
    else:
        df, cleaning = load(args.data)
    res = run(df, cleaning)

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    names = [plots.class_satisfaction(res["class"], res["overall"], figs), plots.theme_lift_plot(res["lift"], figs),
             plots.sizing_plot(res["sizing"], figs), plots.age_plot(res["age"], res["overall"], figs),
             plots.items_plot(res["items"], THRESHOLD, figs)]
    res["class"].to_csv(out / "tables" / "class_summary.csv", index=False)
    res["lift"].to_csv(out / "tables" / "theme_lift.csv", index=False)
    res["sizing"].to_csv(out / "tables" / "sizing_by_class.csv", index=False)
    res["age"].to_csv(out / "tables" / "age_summary.csv", index=False)
    res["items"].to_csv(out / "tables" / "problem_items.csv", index=False)
    summary = {k: res[k] for k in ("cleaning", "overall", "age_test", "items_meta")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_report(res, names, out)
    print(f"Готово: {out / 'report.md'}")
    return 0
