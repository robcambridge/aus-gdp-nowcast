
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.config import ALL_SPECS, unverified_lags  # noqa: E402
from ausgdp.fetch import fetch_one, save_raw  # noqa: E402


def describe(name: str, series: pd.Series, unit_hint: str = "") -> None:
    print(f"\n{'-' * 70}")
    print(f"{name}")
    print(f"{'-' * 70}")
    print(f"  observations : {len(series)}")
    print(f"  first period : {series.index[0]}")
    print(f"  last period  : {series.index[-1]}")
    print(f"  frequency    : {series.index.freqstr}")
    print(f"  mean         : {series.mean():,.1f}")
    print(f"  min / max    : {series.min():,.1f} / {series.max():,.1f}")
    if unit_hint:
        print(f"  expected     : {unit_hint}")
    print("\n  last 6 values:")
    for period, value in series.tail(6).items():
        print(f"    {period}   {value:>15,.2f}")


def main() -> None:
    pinned = [s for s in ALL_SPECS if s.series_id]
    unpinned = [s for s in ALL_SPECS if not s.series_id and s.source == "abs"]

    if not pinned:
        sys.exit(
            "No series pinned yet.\n"
            "Put series_id=\"...\" into the SeriesSpecs in src/ausgdp/config.py first."
        )

    print(f"Checking {len(pinned)} pinned series. Downloading...\n")

    fetched: dict[str, pd.Series] = {}
    failed: dict[str, str] = {}

    for spec in pinned:
        try:
            fetched[spec.name] = fetch_one(spec, verbose=False)
        except Exception as exc:  # noqa: BLE001
            failed[spec.name] = f"{type(exc).__name__}: {exc}"

    for spec in pinned:
        if spec.name in fetched:
            describe(f"{spec.name}   [{spec.series_id}]", fetched[spec.name])

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  fetched OK : {len(fetched)}")
    print(f"  failed     : {len(failed)}")
    print(f"  not pinned : {len(unpinned)}")

    for name, msg in failed.items():
        print(f"\n  FAILED {name}\n    {msg}")

    if unpinned:
        print("\n  Still need a series_id in config.py:")
        for spec in unpinned:
            print(f"    - {spec.name}  (catalogue {spec.collection})")

    missing_lags = unverified_lags()
    if missing_lags:
        print(f"\n  Publication lags still UNVERIFIED ({len(missing_lags)}):")
        print(f"    {', '.join(missing_lags)}")
        print("    Check https://www.abs.gov.au/release-calendar")

    if fetched:
        save_raw(fetched)
        print("\n  Raw series saved to data/raw/ -- you can now work offline.")


if __name__ == "__main__":
    main()
