"""Скачивает «Womens Clothing E-Commerce Reviews.csv» в data/reviews.csv. Файл не хранится в репозитории.

Первоисточник: Kaggle, «Women's E-Commerce Clothing Reviews» (Nick Brooks, публичная лицензия CC0 по описанию набора).
Если автоматическая загрузка не удалась, скачайте файл с Kaggle вручную и сохраните как data/reviews.csv.
"""
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Womens%20Clothing%20E-Commerce%20Reviews.csv",
]
EXPECTED_ROWS = 23_486
target = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"


def main() -> int:
    target.parent.mkdir(exist_ok=True)
    for url in MIRRORS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read()
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось скачать {url}: {exc}")
            continue
        # в тексте отзывов есть переводы строк, поэтому строки считаем через pandas
        import io

        import pandas as pd

        rows = len(pd.read_csv(io.BytesIO(content)))
        if rows != EXPECTED_ROWS:
            print(f"Неожиданное число строк: {rows} (ожидалось {EXPECTED_ROWS}), файл не сохранён")
            continue
        target.write_bytes(content)
        print(f"Сохранено: {target} ({rows:,} строк)")
        return 0
    print("Скачайте файл с Kaggle вручную и сохраните как data/reviews.csv")
    return 1


if __name__ == "__main__":
    sys.exit(main())
