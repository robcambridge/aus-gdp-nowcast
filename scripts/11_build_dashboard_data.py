"""Step 11: assemble everything into one JSON the dashboard reads.

Collects the live nowcast, the horse-race scores, the vintage curve, and the
news decomposition into a single data file so the dashboard is a static page
with no server and no computation of its own.

USAGE
-----
    uv run python scripts\\11_build_dashboard_data.py

OUTPUT
------
    dashboard/data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROCESSED = Path("data/processed")
DASHBOARD = Path("docs")


def _read_json(name: str) -> dict:
    path = PROCESSED / name
    return json.loads(path.read_text()) if path.exists() else {}


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    live = _read_json("live_nowcast.json")
    if not live:
        sys.exit("Run scripts\\10_live_nowcast.py first.")

    scores = _read_csv("horserace_scores.csv")
    forecasts = _read_csv("horserace_by_vintage.csv")
    news = _read_csv("news_decomposition.csv")

    # Vintage curve: RMSE by offset for each model on the modern sample.
    vintage_curve = {}
    if not scores.empty:
        modern = scores[scores["sample"] == "1993-2019"]
        for model in modern["model"].unique():
            rows = modern[modern["model"] == model].sort_values("offset_days")
            vintage_curve[model] = {
                "offsets": rows["offset_days"].tolist(),
                "rmse": [round(v, 4) for v in rows["rmse"].tolist()],
            }

    # Leaderboard at the richest offset.
    leaderboard = []
    if not scores.empty:
        modern = scores[scores["sample"] == "1993-2019"]
        richest = modern["offset_days"].max()
        board = modern[modern["offset_days"] == richest].sort_values("rmse")
        benchmark_rmse = board[board["model"] == "rolling_mean"]["rmse"]
        base = float(benchmark_rmse.iloc[0]) if len(benchmark_rmse) else None
        for _, r in board.iterrows():
            improvement = (
                round(100 * (1 - r["rmse"] / base), 1) if base else None
            )
            leaderboard.append(
                {
                    "model": r["model"],
                    "rmse": round(r["rmse"], 4),
                    "mae": round(r["mae"], 4),
                    "bias": round(r["bias"], 4),
                    "improvement_vs_benchmark": improvement,
                }
            )

    # Actual vs predicted time series for the headline model.
    series = {}
    if not forecasts.empty and "actual" in forecasts.columns:
        head = live.get("headline_model", "bridge_average")
        col = head if head in forecasts.columns else None
        if col:
            series = {
                "quarters": forecasts["target"].astype(str).tolist(),
                "actual": [round(v, 3) for v in forecasts["actual"].tolist()],
                "predicted": [round(v, 3) for v in forecasts[col].tolist()],
            }

    news_rows = []
    if not news.empty:
        for _, r in news.iterrows():
            news_rows.append(
                {
                    "month": str(r.get("update date", "")),
                    "indicator": str(r.get("updated variable", "")),
                    "news": round(float(r.get("news", 0)), 3),
                    "weight": round(float(r.get("weight", 0)), 4),
                    "impact": round(float(r.get("impact", 0)), 4),
                }
            )

    payload = {
        "live": live,
        "leaderboard": leaderboard,
        "vintage_curve": vintage_curve,
        "series": series,
        "news": news_rows,
    }

    DASHBOARD.mkdir(parents=True, exist_ok=True)
    (DASHBOARD / "data.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {DASHBOARD / 'data.json'}")
    print(f"  live nowcast : {live.get('headline_nowcast')}% for {live.get('target_quarter')}")
    print(f"  leaderboard  : {len(leaderboard)} models")
    print(f"  vintage curve: {len(vintage_curve)} models")
    print(f"  news rows    : {len(news_rows)}")
    print(f"\nOpen {DASHBOARD / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
