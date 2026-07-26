"""Run the whole pipeline end to end.

Convenience wrapper: fetch -> build -> benchmark -> horse race -> news ->
live nowcast -> dashboard data. Each step is also runnable on its own; this
just chains them for a clean rebuild.

    uv run python scripts\\run_all.py             # full run (slow: DFM + news)
    uv run python scripts\\run_all.py --fast      # skip DFM-based steps

The DFM steps (horse race DFM block, news) add several minutes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("05_check_series.py", "download + verify series", True),
    ("06_build_dataset.py", "build point-in-time panel", False),
    ("07_benchmark.py", "benchmark backtest", False),
    ("08_horse_race.py", "full horse race (incl. DFM)", False),
    ("09_news.py", "news decomposition", False),
    ("10_live_nowcast.py", "live nowcast", False),
    ("11_build_dashboard_data.py", "assemble dashboard data", False),
    ("12_build_report.py", "generate the research report (.docx)", False),
]


def main() -> None:
    fast = "--fast" in sys.argv
    skip_in_fast = {"08_horse_race.py", "09_news.py"}

    for script, desc, needs_net in STEPS:
        if fast and script in skip_in_fast:
            print(f"\n[skip] {script} ({desc}) -- --fast")
            continue
        print(f"\n{'=' * 70}\n[run] {script}  ({desc})\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(HERE / script)], check=False)
        if result.returncode != 0:
            print(f"\n{script} failed (exit {result.returncode}).")
            if needs_net:
                print("This step needs internet (ABS/RBA). Check your connection.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}\nDone. Open dashboard/index.html in a browser.")
    print("To serve locally:  python -m http.server -d dashboard 8000")


if __name__ == "__main__":
    main()
