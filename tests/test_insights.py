import json

import numpy as np
import pandas as pd
import pytest

from clothing_insights import analysis, stats
from clothing_insights.data import _clean, make_demo
from clothing_insights.report import _json_default, run
from clothing_insights.themes import tag_text, tag_themes


def test_negation_handling():
    assert tag_text("the fabric is thin and cheap")["fabric_quality"]
    assert not tag_text("so soft and not at all itchy")["fabric_quality"]
    assert not tag_text("not too short, perfect length")["too_short"]
    assert tag_text("not flattering on me")["shape_unflattering"]
    assert tag_text("this top is not worth the price")["price"]
    assert not tag_text("worth every penny")["price"]
    # отрицание действует внутри предложения/клаузы и не «просачивается» дальше
    assert tag_text("not flattering at all, fabric is itchy")["fabric_quality"]


def test_sizing_direction():
    t = tag_text("runs small so size up")
    assert t["undersized"] and not t["oversized"]
    t = tag_text("runs big, go down a size")
    assert t["oversized"] and not t["undersized"]


def test_wilson_and_rr():
    lo, hi = stats.wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo and hi < 0.60
    rr = stats.relative_risk(60, 100, 20, 100)
    assert rr["rr"] == pytest.approx(3.0) and rr["ci"][0] < 3.0 < rr["ci"][1] and rr["p_value"] < 0.001
    assert np.isnan(stats.relative_risk(0, 100, 20, 100)["rr"])


def test_holm_and_cramers_v():
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.cramers_v([[50, 50], [50, 50]]) == pytest.approx(0.0)
    assert stats.cramers_v([[100, 0], [0, 100]]) == pytest.approx(1.0)


def test_beta_shrinkage_pulls_small_samples_to_mean():
    s = np.array([1, 90, 80]); n = np.array([2, 100, 100])
    a, b = stats.fit_beta_prior(s, n)
    shrunk_small = (a + 1) / (a + b + 2)
    overall = s.sum() / n.sum()
    # 1 успех из 2 (сырые 50%) после сжатия оказывается ближе к общему среднему, чем сырая оценка
    assert abs(shrunk_small - overall) < abs(0.5 - overall)
    od = stats.overdispersion_test(np.array([50, 50, 50]), np.array([100, 100, 100]))
    assert od["chi2"] == pytest.approx(0.0) and od["p_value"] == pytest.approx(1.0)


def _demo():
    cleaned = _clean(make_demo(n=5000)[["item_id", "age", "title", "text", "rating", "recommended", "helpful",
                                         "division", "department", "cls"]])
    return cleaned.df, {**cleaned.report, "rows_raw": len(cleaned.df)}


def test_pipeline_finds_planted_effects():
    df, cleaning = _demo()
    res = run(df, cleaning)
    lift = res["lift"].set_index("theme")
    assert lift.loc["fabric_quality", "rr"] > 2 and lift.loc["fabric_quality", "p_holm"] < 0.01
    assert lift.loc["undersized", "rr"] < lift.loc["fabric_quality", "rr"]
    items = res["items"]
    # в демо-данных модели 1..7 «плохие» — они должны оказаться в верхушке списка
    assert len(set(items.head(7)["item_id"]) & set(range(1, 8))) >= 5
    assert items["p_below_threshold"].between(0, 1).all()
    assert res["items_meta"]["overdispersion"]["p_value"] < 0.05
    json.dumps({k: res[k] for k in ("cleaning", "overall", "age_test", "items_meta")}, default=_json_default)


def test_class_summary_has_holm_and_wilson():
    df, cleaning = _demo()
    cls = analysis.class_summary(tag_themes(df), min_n=100)
    assert (cls["ci_low"] <= cls["recommend_rate"]).all() and (cls["recommend_rate"] <= cls["ci_high"]).all()
    assert (cls["p_holm"] >= cls["p_value"]).all()
