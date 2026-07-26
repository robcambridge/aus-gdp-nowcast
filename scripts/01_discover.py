"""
Usage
-----
    uv run python scripts/01_discover.py                 # all collections
    uv run python scripts/01_discover.py 6202.0          # just labour force

Output
------
Prints candidate matches, and writes the full metadata for each collection to
data/raw/meta_<cat>.csv so you can search it in a spreadsheet.

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.config import ALL_SPECS  # noqa: E402

OUT_DIR = Path("data/raw")


def main(wanted_cats: list[str] | None = None) -> None:
    try:
        import readabs as ra
        from readabs import metacol as mc
    except ImportError:
        sys.exit("readabs not installed. Run:  uv add readabs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    abs_specs = [s for s in ALL_SPECS if s.source == "abs"]
    cats = sorted({s.collection for s in abs_specs})
    if wanted_cats:
        cats = [c for c in cats if c in wanted_cats]
        if not cats:
            sys.exit(f"None of {wanted_cats} are in the registry. Known: {cats}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    for cat in cats:
        print(f"\n{'=' * 78}\nCatalogue {cat}\n{'=' * 78}")
        try:
            data, meta = ra.read_abs_cat(cat, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  COULD NOT DOWNLOAD: {type(exc).__name__}: {exc}")
            print("  The ABS may have moved or renamed this collection.")
            print("  Check https://www.abs.gov.au/statistics and update config.py")
            continue

        meta_path = OUT_DIR / f"meta_{cat.replace('.', '_')}.csv"
        meta.to_csv(meta_path, index=False)
        print(f"  {len(meta)} series in this collection")
        print(f"  Full metadata written to {meta_path}")

        for spec in [s for s in abs_specs if s.collection == cat]:
            print(f"\n  --- {spec.name} ---")
            print(f"      searching: {spec.search}")
            try:
                matches = ra.search_abs_meta(meta, spec.search)
            except Exception as exc:  # noqa: BLE001
                print(f"      search failed: {exc}")
                continue

            if matches.empty:
                print("      NO MATCHES. The description has probably changed.")
                print(f"      Open {meta_path} and search manually.")
                continue

            cols = [mc.id, mc.did, mc.stype, mc.freq, mc.unit, mc.table, mc.start, mc.end]
            cols = [c for c in cols if c in matches.columns]
            print(matches[cols].to_string(index=False, max_rows=12))

            if len(matches) == 1:
                sid = matches.iloc[0][mc.id]
                print(f"      UNIQUE MATCH -> put  series_id=\"{sid}\"  in config.py")
            else:
                print(f"      {len(matches)} matches -- narrow your search terms, then")
                print("      pin one series_id in config.py")

    print(f"\n{'=' * 78}")
    print("Next: paste the confirmed series IDs into src/ausgdp/config.py,")
    print("then verify each publication lag against the ABS release calendar:")
    print("https://www.abs.gov.au/release-calendar")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
