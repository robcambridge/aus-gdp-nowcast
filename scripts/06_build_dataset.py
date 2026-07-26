

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.config import BALANCED_PANEL_START, SHORT_HISTORY, SPECS_BY_NAME  # noqa: E402
from ausgdp.dataset import as_of, ragged_edge_report  # noqa: E402
from ausgdp.fetch import load_raw  # noqa: E402
from ausgdp.transforms import (  # noqa: E402
    adf_table,
    make_panel,
    quarterly_regressor,
    transform_series,
)

PROCESSED = Path("data/processed")


def main() -> None:
    raw = load_raw()
    if not raw:
        sys.exit(
            "No raw series in data/raw/.\n"
            "Run:  uv run python scripts\\05_check_series.py"
        )

    known = {k: v for k, v in raw.items() if k in SPECS_BY_NAME}
    skipped = sorted(set(raw) - set(known))
    if skipped:
        print(f"Ignoring unregistered files: {', '.join(skipped)}\n")

    print("=" * 74)
    print("RAW SERIES AS DOWNLOADED")
    print("=" * 74)
    for name, s in sorted(known.items()):
        print(f"  {name:<20} {len(s):>4} obs  {s.index[0]} to {s.index[-1]}")

    # --- build the actual model regressors ---------------------------------
    # Monthly series stay as LEVELS in the panel. What the models actually see
    # is the quarterly regressor: average the quarter's levels, then take the
    # change against the previous quarter's average. We build it here at full
    # (3-month) aggregation purely so we can test and report on it.
    regressors = {}
    for name, s in known.items():
        spec = SPECS_BY_NAME[name]
        if spec.stored_as_level:
            regressors[name] = quarterly_regressor(s, spec.aggregation, 3)
        else:
            regressors[name] = transform_series(s, spec.transform)

    print("\n" + "=" * 74)
    print("QUARTERLY REGRESSORS  (what the models actually see)")
    print("=" * 74)
    print(f"  {'series':<20} {'construction':<16} {'n':>5} {'mean':>8} {'sd':>8}")
    for name, s in sorted(regressors.items()):
        spec = SPECS_BY_NAME[name]
        how = spec.aggregation if spec.stored_as_level else spec.transform
        print(f"  {name:<20} {how:<16} {len(s):>5} {s.mean():>8.3f} {s.std():>8.3f}")

    # --- stationarity ------------------------------------------------------
    print("\n" + "=" * 74)
    print("STATIONARITY (Augmented Dickey-Fuller)")
    print("=" * 74)
    print("  Null hypothesis: the series has a unit root (is NOT stationary).")
    print("  p < 0.05 means we reject that, which is what we want.\n")
    adf = adf_table(regressors)
    print(adf.to_string(index=False))

    failures = adf.loc[adf["stationary_5pct"] == False]  # noqa: E712
    if len(failures):
        print(f"\n  WARNING: {len(failures)} series failed. Do not ignore this --")
        print("  either transform differently or discuss it in your report.")

    # --- panel -------------------------------------------------------------
    panel = make_panel(known, SPECS_BY_NAME)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "panel.csv", index=False)
    adf.to_csv(PROCESSED / "adf_tests.csv", index=False)

    print("\n" + "=" * 74)
    print("POINT-IN-TIME PANEL")
    print("=" * 74)
    print(f"  {len(panel):,} stamped observations, {panel['series'].nunique()} series")
    print(f"  earliest publication : {panel['available_from'].min().date()}")
    print(f"  latest publication   : {panel['available_from'].max().date()}")

    # --- what do we know today? -------------------------------------------
    today = pd.Timestamp.today().normalize()
    print("\n" + "=" * 74)
    print(f"WHAT IS PUBLICLY KNOWN AS OF {today.date()}")
    print("=" * 74)
    print(ragged_edge_report(panel, today).to_string(index=False))

    snap = as_of(panel, today)
    if not snap.quarterly.empty and "gdp_growth" in snap.quarterly.columns:
        last_gdp = snap.quarterly["gdp_growth"].last_valid_index()
        print(f"\n  Latest published GDP growth : {last_gdp}")
        print(f"  Quarter to be nowcast       : {last_gdp + 1}  (not yet published)")

    print("\n" + "=" * 74)
    print("SAMPLE CONSTRAINTS")
    print("=" * 74)
    print(f"  Balanced panel starts    : {BALANCED_PANEL_START}")
    print(f"  Excluded from balanced   : {', '.join(sorted(SHORT_HISTORY))}")
    print("\n  Written:")
    print(f"    {PROCESSED / 'panel.csv'}")
    print(f"    {PROCESSED / 'adf_tests.csv'}")
    print("\n  Next:  uv run python scripts\\07_benchmark.py")


if __name__ == "__main__":
    main()
