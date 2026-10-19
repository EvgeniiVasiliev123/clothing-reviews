"""Статистические помощники: доверительные интервалы, относительный риск, Холм, байесовское сжатие."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = sps.norm.ppf(1 - alpha / 2), x / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """p-value двустороннего z-теста для двух долей (объединённая дисперсия)."""
    if min(n1, n2) == 0:
        return float("nan")
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return float(2 * sps.norm.sf(abs((x1 / n1 - x2 / n2) / se))) if se > 0 else 1.0


def relative_risk(x_exp: int, n_exp: int, x_non: int, n_non: int, alpha: float = 0.05) -> dict:
    """RR = P(исход | фактор) / P(исход | нет фактора), ДИ по log-RR; p-value по хи-квадрат."""
    if min(x_exp, x_non) == 0 or min(n_exp, n_non) == 0:
        return {"rr": float("nan"), "ci": (float("nan"), float("nan")), "p_value": float("nan")}
    rr = (x_exp / n_exp) / (x_non / n_non)
    se = np.sqrt(1 / x_exp - 1 / n_exp + 1 / x_non - 1 / n_non)
    z = sps.norm.ppf(1 - alpha / 2)
    table = [[x_exp, n_exp - x_exp], [x_non, n_non - x_non]]
    p = sps.chi2_contingency(table, correction=False)[1]
    return {"rr": float(rr), "ci": (float(rr * np.exp(-z * se)), float(rr * np.exp(z * se))), "p_value": float(p)}


def holm_adjust(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def cramers_v(table) -> float:
    t = np.asarray(table)
    chi2 = sps.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def fit_beta_prior(successes: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """Априорное Beta(a, b) методом моментов по набору объектов (отнимаем биномиальный шум)."""
    s, n = np.asarray(successes, float), np.asarray(trials, float)
    m = s.sum() / n.sum()
    p = s / n
    between = np.sum(n * (p - m) ** 2) / n.sum()
    noise = m * (1 - m) * len(n) / n.sum()
    var = max(between - noise, 1e-4)
    total = np.clip(m * (1 - m) / var - 1, 2.0, 1000.0)
    return float(m * total), float((1 - m) * total)


def overdispersion_test(successes: np.ndarray, trials: np.ndarray) -> dict:
    """Различаются ли объекты сильнее, чем допускает чистый биномиальный шум (хи-квадрат Пирсона)."""
    s, n = np.asarray(successes, float), np.asarray(trials, float)
    m = s.sum() / n.sum()
    chi2 = float(np.sum(n * (s / n - m) ** 2) / (m * (1 - m)))
    df = len(n) - 1
    return {"chi2": chi2, "df": df, "ratio": chi2 / df, "p_value": float(sps.chi2.sf(chi2, df))}
