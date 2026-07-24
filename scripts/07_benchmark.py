"""Step 7: the benchmark horse race. This produces the number to beat.

Runs one nowcast per GDP release, using only data published by that morning,
and scores four benchmark models against each other.

Read the output carefully. If your later machine-learning models cannot beat
the best number here, say so plainly in your report -- that is a real finding,
not a failure. Most published nowcasting work struggles to beat a well-tuned
AR benchmark by more than 10-25%.

USAGE
-----
    uv run python scripts\\07_benchmark.py

OUTPUT
------
    data/processed/benchmark_forecasts.csv
    data/processed/benchmark_scores.csv
    outputs/figures/benchmark_actual_vs_predicted.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.benchmarks import diebold_mariano, run_backtest  # noqa: E402

PROCESSED = Path("data/processed")
FIGURES = Path("outputs/figures")


def load_panel() -> pd.DataFrame:
    path = PROCESSED / "panel.csv"
    if not path.exists():
        sys.exit(
            f"{path} not found.\n"
            "Run:  uv run python scripts\\06_build_dataset.py"
        )
    return pd.read_csv(path, parse_dates=["ref_end", "available_from"])


def main() -> None:
    panel = load_panel()
    print(f"Loaded panel: {len(panel):,} observations\n")

    res = run_backtest(panel)

    print("=" * 74)
    print("BACKTEST")
    print("=" * 74)
    print(f"  {len(res.forecasts)} nowcasts")
    print(f"  from {res.forecasts.index[0]} to {res.forecasts.index[-1]}")
    print("  one forecast per GDP release, model refit from scratch each time")

    # --- scores ------------------------------------------------------------
    samples = [
        (None, None, "full"),
        (None, "2019Q4", "pre-COVID"),
        ("2021Q1", None, "post-2020"),
    ]
    tables = [res.score(start=a, end=b, label=lab) for a, b, lab in samples]
    scores = pd.concat([t for t in tables if not t.empty])

    print("\n" + "=" * 74)
    print("ACCURACY  (rmse and mae in percentage points of quarterly growth)")
    print("=" * 74)
    for label in scores["sample"].unique():
        block = scores.loc[scores["sample"] == label].drop(columns="sample")
        print(f"\n  {label}:")
        print(block.round(4).to_string())

    best = res.score(label="full").index[0]
    print(f"\n  Best overall: {best}")

    # --- is the gap real? --------------------------------------------------
    print("\n" + "=" * 74)
    print(f"DIEBOLD-MARIANO vs {best}")
    print("=" * 74)
    print("  Negative statistic => the challenger is MORE accurate.")
    print("  p < 0.05 => the difference is unlikely to be noise.\n")

    errors = res.errors()
    rows = []
    for name in errors.columns:
        if name == best:
            continue
        stat, pval = diebold_mariano(errors[name], errors[best])
        rows.append(
            {
                "model": name,
                "vs": best,
                "dm_stat": round(stat, 3) if pd.notna(stat) else None,
                "p_value": round(pval, 4) if pd.notna(pval) else None,
                "verdict": (
                    "no significant difference"
                    if pd.isna(pval) or pval >= 0.05
                    else ("challenger better" if stat < 0 else f"{best} better")
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # --- COVID ------------------------------------------------------------
    err_sq = errors[best] ** 2
    worst = err_sq.nlargest(5)
    print("\n" + "=" * 74)
    print(f"FIVE WORST QUARTERS FOR {best}")
    print("=" * 74)
    for period, _ in worst.items():
        print(
            f"  {period}   actual {res.actuals[period]:>7.2f}   "
            f"forecast {res.forecasts.loc[period, best]:>7.2f}   "
            f"error {errors.loc[period, best]:>7.2f}"
        )
    print("\n  If 2020 dominates this list, that is expected and worth a paragraph.")

    # --- save --------------------------------------------------------------
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = res.forecasts.copy()
    out["actual"] = res.actuals
    out["vintage"] = res.vintages
    out.to_csv(PROCESSED / "benchmark_forecasts.csv")
    scores.to_csv(PROCESSED / "benchmark_scores.csv")

    _plot(res, best)

    print("\n" + "=" * 74)
    print("  Written:")
    print(f"    {PROCESSED / 'benchmark_forecasts.csv'}")
    print(f"    {PROCESSED / 'benchmark_scores.csv'}")
    print(f"    {FIGURES / 'benchmark_actual_vs_predicted.png'}")
    print("\n  This is your baseline. Every later model gets compared to it.")


def _plot(res, best: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not available, skipping figure)")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    x = res.actuals.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, lw=0.8, color="0.7")
    ax.plot(x, res.actuals.to_numpy(), lw=1.6, color="#1b2a4a", label="actual")
    ax.plot(
        x, res.forecasts[best].to_numpy(), lw=1.3, color="#c1440e",
        ls="--", label=f"{best} nowcast",
    )
    ax.set_title(
        "Australian quarterly GDP growth: actual vs point-in-time nowcast",
        fontsize=11,
    )
    ax.set_ylabel("% change on previous quarter")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
