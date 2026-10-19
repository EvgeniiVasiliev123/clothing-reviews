"""Печатает случайные совпадения темы для ручной проверки точности словарной разметки.
Пример: python scripts/sample_matches.py --theme fabric_quality --n 25"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clothing_insights.data import load  # noqa: E402
from clothing_insights.themes import THEME_PATTERNS, tag_themes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/reviews.csv")
ap.add_argument("--theme", default="fabric_quality", choices=list(THEME_PATTERNS))
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()

df, _ = load(args.data)
df = tag_themes(df)
sample = df[df[args.theme]].sample(args.n, random_state=args.seed)
for i, row in enumerate(sample.itertuples(), 1):
    low = row.text.lower()
    m = re.search(THEME_PATTERNS[args.theme], low)
    a = m.start() if m else 0
    print(f"{i:>2} | рек={row.recommended} | ...{low[max(0, a - 60): a + 70]}...".replace("\n", " "))
