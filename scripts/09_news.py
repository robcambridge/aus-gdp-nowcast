"""Step 9: news decomposition -- why did the nowcast change?

This is what a nowcasting desk produces. As new monthly data arrives through a
quarter, each release differs from what the model expected, and that surprise
moves the nowcast. This script fits the dynamic factor model at two consecutive
vintages and attributes the change in the current-quarter nowcast to the
specific data releases responsible.

The DFM did not win the horse race, but the news decomposition is the single
most report-worthy thing it produces: it turns the model from a black box into
an explanation a person can read.

USAGE
-----
    uv run python scripts\\09_news.py

OUTPUT
------
    data/processed/news_decomposition.csv
    outputs/figures/news_waterfall.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.config import SHORT_HISTORY  # noqa: E402
from ausgdp.dataset import as_of, target_release_dates  # noqa: E402
from ausgdp.factor import news_decomposition  # noqa: E402

PROCESSED = Path("data/processed")
FIGURES = Path("outputs/figures")


def _monthly_levels(panel: pd.DataFrame, when, indicators: list[str]) -> pd.DataFrame:
    snap = as_of(panel, when)
    cols = [c for c in indicators if c in snap.monthly.columns]
    return snap.monthly[cols].dropna(how="all")


def main() -> None:
    path = PROCESSED / "panel.csv"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts\\06_build_dataset.py first.")

    panel = pd.read_csv(path, parse_dates=["ref_end", "available_from"])

    indicators = sorted(set(panel.loc[panel["freq"] == "M", "series"]) - SHORT_HISTORY)
    releases = target_release_dates(panel)

    # The most recent GDP release, and the quarter we are nowcasting from it.
    last_release_period = releases.index[-1]
    last_release_date = releases.iloc[-1]
    impact_quarter = pd.Period(last_release_period, freq="Q") + 1

    # Two vintages: the GDP-release morning, and 45 days later once another
    # month of labour force and other indicators has arrived.
    early = last_release_date
    late = last_release_date + pd.Timedelta(days=45)

    print("=" * 74)
    print("NEWS DECOMPOSITION")
    print("=" * 74)
    print(f"  Nowcasting {impact_quarter} GDP growth.")
    print(f"  Earlier vintage : {early.date()}  (the {last_release_period} GDP release)")
    print(f"  Later vintage   : {late.date()}  (45 days later)")
    print(f"  Indicators      : {', '.join(indicators)}\n")

    q = as_of(panel, early).quarterly[["gdp_growth"]].dropna()
    m_early = _monthly_levels(panel, early, indicators)
    m_late = _monthly_levels(panel, late, indicators)

    new_rows = len(m_late.dropna(how="all")) - len(m_early.dropna(how="all"))
    print(f"  {new_rows} new indicator-months arrived between the vintages.\n")

    details = news_decomposition(m_early, m_late, q, impact_quarter)
    if details is None or details.empty:
        print("  Decomposition could not be computed (EM did not converge, or no")
        print("  new data between the vintages). Try widening the vintage gap.")
        return

    print(details.round(4).to_string(index=False))
    print(f"\n  Total revision to the {impact_quarter} nowcast: "
          f"{details['impact'].sum():+.4f} pp")
    print("\n  Read this as: each release surprised the model by `news`, the")
    print("  filter weighted it by `weight`, and the nowcast moved by `impact`.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    details.to_csv(PROCESSED / "news_decomposition.csv", index=False)

    _waterfall(details, impact_quarter)

    print("\n  Written:")
    print(f"    {PROCESSED / 'news_decomposition.csv'}")
    print(f"    {FIGURES / 'news_waterfall.png'}")


def _waterfall(details: pd.DataFrame, impact_quarter) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_var = details.groupby("updated variable")["impact"].sum().sort_values()
    if by_var.empty:
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = ["#c1440e" if v < 0 else "#1b6b3a" for v in by_var.to_numpy()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(by_var.index, by_var.to_numpy(), color=colors)
    ax.axvline(0, lw=0.8, color="0.4")
    ax.set_xlabel("contribution to change in nowcast (pp)")
    ax.set_title(
        f"What moved the {impact_quarter} GDP nowcast", fontsize=11
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "news_waterfall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
