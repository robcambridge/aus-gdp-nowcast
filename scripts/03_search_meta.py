"""
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

DISPLAY = [
    "Series ID",
    "Data Item Description",
    "Series Type",
    "Freq.",
    "Unit",
    "Table",
    "Series Start",
    "Series End",
]


def meta_files() -> dict[str, Path]:
    """Map catalogue number -> saved metadata csv."""
    out = {}
    for p in sorted(RAW.glob("meta_*.csv")):
        cat = p.stem.replace("meta_", "").replace("_", ".")
        out[cat] = p
    return out


def load(cat: str) -> pd.DataFrame:
    files = meta_files()
    if cat not in files:
        have = ", ".join(files) or "(none)"
        sys.exit(
            f"No metadata for {cat}. You have: {have}\n"
            "Run scripts/01_discover.py first to download it."
        )
    return pd.read_csv(files[cat], low_memory=False)


def summarise(df: pd.DataFrame, cat: str) -> None:
    print(f"\nCatalogue {cat}: {len(df)} series")

    if "Freq." in df:
        print("\nFrequencies:")
        print(df["Freq."].value_counts().to_string())
    if "Series Type" in df:
        print("\nSeries types:")
        print(df["Series Type"].value_counts().to_string())
    if "Table" in df:
        print(f"\nTables ({df['Table'].nunique()}):")
        print(df["Table"].value_counts().head(15).to_string())

    print("\nTip: search for a single plain word, e.g.")
    print(f"    uv run python scripts\\03_search_meta.py {cat} exports")


def search(df: pd.DataFrame, words: list[str], sa: bool, freq: str | None) -> pd.DataFrame:
    col = "Data Item Description"
    hits = df.copy()

    for w in words:
        hits = hits[hits[col].str.contains(w, case=False, na=False, regex=False)]

    if sa and "Series Type" in hits:
        hits = hits[hits["Series Type"].str.contains("Seasonally", case=False, na=False)]

    if freq and "Freq." in hits:
        hits = hits[hits["Freq."].str.startswith(freq, na=False)]

    return hits


def main(argv: list[str]) -> None:
    if not RAW.exists():
        sys.exit("data/raw does not exist. Are you in C:\\projects\\macro-model ?")

    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}

    if not args:
        files = meta_files()
        if not files:
            sys.exit("No metadata files yet. Run scripts/01_discover.py first.")
        print("Metadata you have downloaded:\n")
        for cat, p in files.items():
            n = sum(1 for _ in p.open(encoding="utf-8", errors="ignore")) - 1
            print(f"  {cat:<12} {n:>6} series   {p}")
        print("\nNext:  uv run python scripts\\03_search_meta.py <catalogue> <word>")
        return

    cat, words = args[0], args[1:]
    df = load(cat)

    if not words:
        summarise(df, cat)
        return

    freq = "M" if "--monthly" in flags else "Q" if "--quarterly" in flags else None
    hits = search(df, words, sa="--sa" in flags, freq=freq)

    filters = " ".join(words)
    if "--sa" in flags:
        filters += " [seasonally adjusted]"
    if freq:
        filters += f" [{'monthly' if freq == 'M' else 'quarterly'}]"

    print(f"\nCatalogue {cat} -- searching for: {filters}")
    print(f"{len(hits)} matches\n")

    if hits.empty:
        print("Nothing found. Try a shorter or more common word.")
        print("ABS wording is fussy: 'exports' may appear as 'Credits'.")
        return

    cols = [c for c in DISPLAY if c in hits.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 65)
    pd.set_option("display.max_rows", 60)
    print(hits[cols].to_string(index=False))

    if len(hits) > 60:
        print(f"\n(showing first 60 of {len(hits)} -- add another word to narrow)")

    print("\nCopy the Series ID you want into src/ausgdp/config.py, e.g.")
    print(f'    series_id="{hits.iloc[0]["Series ID"]}",')


if __name__ == "__main__":
    main(sys.argv[1:])
