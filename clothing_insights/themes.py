"""Словарная разметка тем в тексте отзывов с учётом отрицаний ("not at all itchy" — не жалоба)."""
from __future__ import annotations

import re

import pandas as pd

THEME_PATTERNS = {
    "fabric_quality": r"\b(?:cheap(?:ly)?|flimsy|itch(?:y|es)|scratchy|pill(?:s|ing|ed)|fell apart|falling apart|ripped|tore|poorly made|poor(?:ly)? quality|low quality|shrunk|shrank|shrinks|see[- ]through|too sheer|unravel\w*|loose threads?|wrinkles? (?:easily|so easily|terribly|badly)|stretched out|faded|too thin|very thin|so thin|paper[- ]thin|thin and|rough fabric|stiff fabric)\b",
    "shape_unflattering": r"\b(?:unflattering|boxy|frumpy|shapeless|matronly|sack|tent[- ]like|like a tent|bulky)\b",
    "too_short": r"\b(?:too short|bit short|very short|shorter than|so short)\b",
    "expectation_gap": r"\b(?:not as (?:pictured|shown|advertised)|different (?:than|from) (?:the )?(?:picture|photo|pic|online|website|model)s?|misleading|looks? nothing like)\b",
    "price": r"\b(?:overpriced|too expensive|waste of money|pricey|too much money)\b",
    "undersized": r"\b(?:run|runs|running|ran) (?:a bit |a little |very |pretty |really |quite |way |much |somewhat )?(?:small|tight|narrow)|\btoo (?:small|tight|snug|narrow)\b|\bsize up\b|\bsized up\b|\bsizing up\b|go up a size|\bsmaller than\b",
    "oversized": r"\b(?:run|runs|running|ran) (?:a bit |a little |very |pretty |really |quite |way |much |somewhat )?(?:large|big|huge|loose)|\btoo (?:big|large|loose|baggy)\b|\bsize down\b|\bsized down\b|\bsizing down\b|go down a size|\bbigger than\b",
    "returned": r"\b(?:returned|returning|return it|sent (?:it )?back|send (?:it )?back|exchanged)\b",
}
LABELS = {
    "fabric_quality": "Ткань и качество пошива", "shape_unflattering": "Крой / «не идёт по фигуре»",
    "too_short": "Слишком короткое", "expectation_gap": "Не как на фото", "price": "Цена / «не стоит денег»",
    "undersized": "Маломерит", "oversized": "Большемерит", "returned": "Упоминание возврата",
}
# Темы, которые производитель может менять сам (не считая цены и возврата):
PRODUCT_THEMES = ["fabric_quality", "shape_unflattering", "too_short", "expectation_gap"]

# Жалобы, выраженные через отрицание положительного слова ("not flattering", "not worth"):
NEGATORS = re.compile(r"\b(?:not|no|never|without|hardly|nor|neither|barely)\b|n't|\bnot at all\b")
_NEG_PATTERNS = {
    "shape_unflattering": r"\b(?:not|n't|never|no)\s+(?:\w+\s+){0,2}(?:flattering|flatter)\b|\bdoesn'?t flatter|\bdidn'?t flatter",
    "price": r"\b(?:not|n't|never)\s+(?:\w+\s+){0,2}worth\b",
}
_CLAUSE_SPLIT = re.compile(r"[.!?;,:]|\b(?:and|but|though|however|although)\b")
_COMPILED = {k: re.compile(v) for k, v in THEME_PATTERNS.items()}
_NEG_COMPILED = {k: re.compile(v) for k, v in _NEG_PATTERNS.items()}


def _has_unnegated(text: str, pattern: re.Pattern, window: int = 35) -> bool:
    for m in pattern.finditer(text):
        prefix = _CLAUSE_SPLIT.split(text[max(0, m.start() - window): m.start()])[-1]
        if not NEGATORS.search(prefix):
            return True
    return False


def tag_text(text: str) -> dict[str, bool]:
    low = text.lower()
    out = {k: _has_unnegated(low, p) for k, p in _COMPILED.items()}
    for k, p in _NEG_COMPILED.items():  # "not flattering", "not worth" — как раз жалобы
        out[k] = out[k] or bool(p.search(low))
    return out


def tag_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет булевы колонки тем. Отзывы без текста получают NaN-эквивалент (False + has_text=False)."""
    df = df.copy()
    tags = df["text"].where(df["has_text"], "").astype(str).map(tag_text)
    for k in THEME_PATTERNS:
        df[k] = tags.map(lambda d, key=k: d[key]) & df["has_text"]
    df["any_product_theme"] = df[PRODUCT_THEMES].any(axis=1)
    return df
