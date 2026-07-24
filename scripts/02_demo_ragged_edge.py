"""Step 2: see the ragged edge with your own eyes.

This runs on SYNTHETIC data and needs no network, so you can run it right now
before any ABS download works. Its only purpose is to make the point-in-time
machinery concrete: what did I know, on what date?

Run it, read the output, and make sure you can explain every line of the tables
it prints. Once the real data lands, the same functions do the same job.

Usage
-----
    uv run python scripts/02_demo_ragged_edge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.config import SPECS_BY_NAME  # noqa: E402
from ausgdp.dataset import (  # noqa: E402
    as_of,
    build_panel,
    ragged_edge_report,
    target_release_dates,
)

RNG = np.random.default_rng(20240903)


def fake_monthly(name: str, start="2018-01", n=84) -> pd.Series:
    idx = pd.period_range(start=start, periods=n, freq="M")
    return pd.Series(RNG.normal(0.2, 0.6, n).round(3), index=idx, name=name)


def fake_quarterly(name: str, start="2018Q1", n=28) -> pd.Series:
    idx = pd.period_range(start=start, periods=n, freq="Q")
    return pd.Series(RNG.normal(0.5, 0.7, n).round(3), index=idx, name=name)


def main() -> None:
    # Use the REAL specs from config.py -- only the numbers are fake.
    names_m = ["employment", "unemployment_rate", "retail_turnover", "building_approvals"]
    series = {n: fake_monthly(n) for n in names_m}
    series["gdp_growth"] = fake_quarterly("gdp_growth")

    panel = build_panel(series, SPECS_BY_NAME)

    print("=" * 78)
    print("THE LONG PANEL -- every observation carries its publication date")
    print("=" * 78)
    print(panel.tail(8).to_string(index=False))

    print("\n" + "=" * 78)
    print("PUBLICATION LAGS IN USE")
    print("=" * 78)
    for n in [*names_m, "gdp_growth"]:
        s = SPECS_BY_NAME[n]
        flag = "" if s.lag_verified else "   <-- UNVERIFIED"
        print(f"  {n:<20} {s.freq}  {s.lag_days:>3} days{flag}")

    # The natural backtest dates: one per GDP release.
    releases = target_release_dates(panel)
    print("\n" + "=" * 78)
    print("GDP RELEASE DATES (these are your backtest vintages)")
    print("=" * 78)
    print(releases.tail(5).to_string())

    # Pick one vintage and look at it closely.
    vintage = releases.iloc[-3]
    ref_just_published = releases.index[-3]

    print("\n" + "=" * 78)
    print(f"VINTAGE {vintage.date()} -- the morning {ref_just_published} GDP was published")
    print("=" * 78)

    snap = as_of(panel, vintage)
    print(f"\n{snap}\n")

    print("What each series had reached on that date:")
    print(ragged_edge_report(panel, vintage).to_string(index=False))

    print("\nThe monthly block (last 6 rows) -- note the staircase of NaNs:")
    print(snap.monthly.tail(6).to_string())

    print("\nThe quarterly block (last 4 rows):")
    print(snap.quarterly.tail(4).to_string())

    # --- The check that matters -------------------------------------------
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK")
    print("=" * 78)

    next_q = ref_just_published_next(ref_just_published)
    leaked = next_q in [str(p) for p in snap.quarterly.index]
    print(f"  Is {next_q} GDP (the quarter we want to nowcast) visible? {leaked}")
    assert not leaked, "LEAKAGE: the target quarter is already in the snapshot"

    visible_rows = panel.loc[panel["available_from"] <= vintage]
    assert snap.n_values == len(visible_rows)
    print(f"  Snapshot holds {snap.n_values} values; "
          f"{len(visible_rows)} were published. Match.")
    print("\n  No leakage. This is the property every backtest step must preserve.")

    # --- Why the edge is ragged, and why timing is your edge ---------------
    print("\n" + "=" * 78)
    print("THE SAME QUARTER, SEEN FROM FOUR DIFFERENT DAYS")
    print("=" * 78)
    print("Nowcasting 2024Q3 GDP. Watch the information set grow.\n")

    rows = []
    for day in ["2024-09-03", "2024-09-20", "2024-10-20", "2024-11-20"]:
        s = as_of(panel, day)
        last = {c: str(s.monthly[c].last_valid_index()) for c in sorted(s.monthly.columns)}
        rows.append({"vintage": day, "n_obs": s.n_values, **last})

    evolution = pd.DataFrame(rows).set_index("vintage")
    print(evolution.to_string())

    print("\n  Read down each column: series arrive at different times, so the")
    print("  bottom edge of the data is a staircase, not a straight line.")
    print("  Read across each row: by 20 Nov you hold all three months of the")
    print("  September quarter for the fast series -- but ABS will not publish")
    print("  2024Q3 GDP until 2024-12-04. That gap is the whole opportunity.")


def ref_just_published_next(ref: str) -> str:
    return str(pd.Period(ref, freq="Q") + 1)


if __name__ == "__main__":
    main()
