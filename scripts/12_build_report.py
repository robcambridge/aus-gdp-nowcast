"""
-----
    uv run python scripts\\12_build_report.py

OUTPUT
------
    report/Australian_GDP_Nowcasting_Report.docx
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp import report_figures  # noqa: E402
from ausgdp.config import ALL_SPECS, BALANCED_PANEL_START, SHORT_HISTORY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "report"


def _read(name: str) -> pd.DataFrame:
    p = PROCESSED / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(name: str) -> dict:
    p = PROCESSED / name
    return json.loads(p.read_text()) if p.exists() else {}


def collect() -> dict:
    """Pull every number the report needs into one payload."""
    live = _json("live_nowcast.json")
    scores = _read("horserace_scores.csv")
    adf = _read("adf_tests.csv")

    # Leaderboard on the modern sample at the richest offset.
    board = []
    dm_note = ""
    if not scores.empty:
        sample = "1993-2019" if (scores["sample"] == "1993-2019").any() else "full"
        modern = scores[scores["sample"] == sample]
        richest = modern["offset_days"].max()
        b = modern[modern["offset_days"] == richest].sort_values("rmse")
        base = b[b["model"] == "rolling_mean"]["rmse"]
        base = float(base.iloc[0]) if len(base) else None
        for _, r in b.iterrows():
            imp = round(100 * (1 - r["rmse"] / base), 1) if base else None
            board.append(
                {
                    "model": r["model"],
                    "rmse": round(float(r["rmse"]), 4),
                    "mae": round(float(r["mae"]), 4),
                    "bias": round(float(r["bias"]), 4),
                    "improvement": imp,
                }
            )

    # Series registry for the data table.
    series_rows = []
    for s in ALL_SPECS:
        if s.source != "abs":
            continue
        series_rows.append(
            {
                "name": s.name,
                "series_id": s.series_id or "—",
                "collection": s.collection,
                "freq": s.freq,
                "lag_days": s.lag_days,
                "transform": s.transform if s.freq == "Q" else s.aggregation,
            }
        )

    adf_rows = []
    if not adf.empty:
        for _, r in adf.iterrows():
            adf_rows.append(
                {
                    "series": r["series"],
                    "adf_stat": (round(float(r["adf_stat"]), 2)
                                 if pd.notna(r["adf_stat"]) else None),
                    "p_value": (round(float(r["p_value"]), 4)
                                if pd.notna(r["p_value"]) else None),
                    "stationary": (bool(r["stationary_5pct"])
                                   if pd.notna(r["stationary_5pct"]) else None),
                }
            )

    return {
        "live": live,
        "leaderboard": board,
        "adf": adf_rows,
        "series": series_rows,
        "balanced_start": BALANCED_PANEL_START,
        "short_history": sorted(SHORT_HISTORY),
        "dm_note": dm_note,
    }


def main() -> None:
    if not (PROCESSED / "horserace_scores.csv").exists():
        sys.exit(
            "Missing results. Run the pipeline first:\n"
            "  uv run python scripts\\06_build_dataset.py\n"
            "  uv run python scripts\\07_benchmark.py\n"
            "  uv run python scripts\\08_horse_race.py\n"
            "  uv run python scripts\\10_live_nowcast.py"
        )

    REPORT.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    figs = report_figures.generate_all()

    print("Collecting results...")
    payload = collect()
    payload["figures"] = {k: str(v.resolve()) for k, v in figs.items()}
    payload["out_path"] = str((REPORT / "Australian_GDP_Nowcasting_Report.docx").resolve())

    payload_path = REPORT / "_report_data.json"
    payload_path.write_text(json.dumps(payload, indent=2))

    print("Building .docx...")
    node_script = ROOT / "scripts" / "build_report.js"
    result = subprocess.run(
        ["node", str(node_script), str(payload_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit("Report build failed. Is Node.js installed? (node --version)")

    print(f"\nWrote {payload['out_path']}")


if __name__ == "__main__":
    main()
