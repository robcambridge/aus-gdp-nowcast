"""Step 10: the live nowcast. What is GDP growth doing right now?

Everything up to here evaluated the models on history. This produces the actual
current forecast: standing today, with the data published so far, what is the
model's nowcast for the quarter that has not yet been released?

This is the headline number for the dashboard and the README. Unlike the
backtest, there is no "actual" to compare against yet -- that is the point. When
the ABS publishes the quarter (in a few weeks), you find out whether you were
right, which is the honest test of any forecasting system.

USAGE
-----
    uv run python scripts\\10_live_nowcast.py

OUTPUT
------
    data/processed/live_nowcast.json
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="A date index has been provided")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.benchmarks import (  # noqa: E402
    BENCHMARKS,
    Context,
)
from ausgdp.bridge import make_bridge, make_bridge_average, make_ridge  # noqa: E402
from ausgdp.config import SHORT_HISTORY  # noqa: E402
from ausgdp.dataset import as_of, ragged_edge_report  # noqa: E402
from ausgdp.factor import make_dfm  # noqa: E402

PROCESSED = Path("data/processed")
WINDOW = 40


def main() -> None:
    path = PROCESSED / "panel.csv"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts\\06_build_dataset.py first.")

    panel = pd.read_csv(path, parse_dates=["ref_end", "available_from"])
    today = pd.Timestamp.today().normalize()

    indicators = sorted(set(panel.loc[panel["freq"] == "M", "series"]) - SHORT_HISTORY)
    all_monthly = sorted(set(panel.loc[panel["freq"] == "M", "series"]))

    # What quarter are we nowcasting? The one after the latest published GDP.
    snap = as_of(panel, today)
    if snap.quarterly.empty or "gdp_growth" not in snap.quarterly.columns:
        sys.exit("No GDP data visible. Check the panel.")

    y_visible = snap.quarterly["gdp_growth"].dropna()
    last_gdp = y_visible.index[-1]
    target_q = last_gdp + 1

    print("=" * 74)
    print(f"LIVE NOWCAST as of {today.date()}")
    print("=" * 74)
    print(f"  Latest published GDP : {last_gdp} ({y_visible.iloc[-1]:+.2f}%)")
    print(f"  Nowcasting           : {target_q}  (not yet published by the ABS)\n")

    print("  Data available for the target quarter:")
    edge = ragged_edge_report(panel, today)
    for _, row in edge.iterrows():
        marker = ""
        last_q = pd.Period(row["last_ref_period"], freq="Q" if row["freq"] == "Q" else "M")
        if row["freq"] == "M" and last_q.asfreq("Q") == target_q:
            marker = "  <- has target-quarter data"
        print(f"    {row['series']:<20} through {row['last_ref_period']}{marker}")
    print()

    ctx = Context(y=y_visible, snapshot=snap, target=target_q, vintage=today)

    models = dict(BENCHMARKS)
    models["bridge_average"] = make_bridge_average(indicators, add_ar=False, window=WINDOW)
    for name in indicators:
        models[f"bridge_{name}"] = make_bridge(name, add_ar=False, window=WINDOW)
    models["ridge"] = make_ridge(indicators, window=WINDOW, add_ar=False)
    models["dfm_1f"] = make_dfm(all_monthly, factors=1)

    print("  Fitting models (the DFM takes a moment)...\n")
    forecasts = {}
    for name, fn in models.items():
        try:
            forecasts[name] = round(float(fn(ctx)), 3)
        except Exception:  # noqa: BLE001
            forecasts[name] = None

    # Headline = the model that won the backtest.
    headline_model = "bridge_average"
    headline = forecasts.get(headline_model)

    print("=" * 74)
    print("MODEL FORECASTS")
    print("=" * 74)
    ordered = sorted(
        (k for k in forecasts if forecasts[k] is not None),
        key=lambda k: (k != headline_model, k),
    )
    for name in ordered:
        star = "  *" if name == headline_model else ""
        print(f"    {name:<24} {forecasts[name]:+.2f}%{star}")
    print("\n  * headline model (best in backtest: rolling-window bridge average)")

    print("\n" + "=" * 74)
    print(f"  HEADLINE NOWCAST for {target_q}: {headline:+.2f}%")
    print("=" * 74)
    print("  No actual exists yet. The ABS will publish this quarter in a few")
    print("  weeks; that release is the real test.")

    payload = {
        "as_of": str(today.date()),
        "generated": str(date.today()),
        "target_quarter": str(target_q),
        "latest_published_quarter": str(last_gdp),
        "latest_published_value": round(float(y_visible.iloc[-1]), 3),
        "headline_model": headline_model,
        "headline_nowcast": headline,
        "forecasts": forecasts,
        "indicators": indicators,
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "live_nowcast.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  Written: {PROCESSED / 'live_nowcast.json'}")


if __name__ == "__main__":
    main()
