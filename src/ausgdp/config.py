"""Series registry and publication-lag table.

This module is the single source of truth for:
  1. WHICH series we use,
  2. WHERE each one comes from,
  3. HOW LONG after its reference period each one is actually published.

(3) is what makes or breaks a nowcasting project. If you assume a series was
available earlier than it really was, your backtest sees the future and your
results are fiction.

LAG POLICY: CONSERVATIVE UPPER BOUNDS
-------------------------------------
ABS release dates drift by several days from month to month, so there is no
single "true" lag. We therefore set each lag to the LONGEST recently observed
delay, plus a small buffer.

The two errors are not symmetric:

    lag too SHORT -> the model uses data that did not exist yet.
                     Results inflated. Project worthless.
    lag too LONG  -> we discard a day or two of genuinely available data.
                     Results slightly understated. Project still honest.

So we deliberately err long. Measured performance is a LOWER BOUND on what a
real-time forecaster could have achieved. State this in the README.

All lags below were checked against the ABS release calendar on 2026-07-24.
The observed release dates behind each choice are recorded in `notes` so a
reader can audit the decision rather than take it on trust.

    ABS release calendar: https://www.abs.gov.au/release-calendar

REMAINING CAVEAT
----------------
Lags are held CONSTANT across the whole sample. ABS releases were slower in the
1980s and 1990s than today, so for early observations these bounds are, if
anything, too generous to the model in the safe direction for recent data and
possibly too tight for the oldest data. Disclose this. Fixing it properly means
using true data vintages via readabs' `history=` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Freq = Literal["M", "Q"]
Transform = Literal["level", "pct", "diff", "log_pct"]

# Date on which the publication lags below were checked against the ABS calendar.
LAGS_CHECKED_ON = "2026-07-24"


@dataclass(frozen=True)
class SeriesSpec:
    """One economic time series and everything we need to use it honestly.

    Attributes
    ----------
    name        Short identifier used as the column name throughout the project.
    source      "abs" or "rba".
    collection  ABS catalogue number (e.g. "6202.0") or RBA table id (e.g. "F2").
    search      Metadata search terms, {search_value: metadata_column}. Kept
                even after pinning an ID, so the intent stays readable and the
                series can be re-found if the ID is ever retired.
    series_id   Exact ABS Series ID, confirmed via scripts/01_discover.py.
    freq        Native frequency. Do NOT pre-aggregate monthly data to
                quarterly; the whole point is to keep the monthly timing.
    transform   How to make it stationary. "pct" = percent change on previous
                period, "diff" = first difference, "level" = leave it alone.
    lag_days    Conservative upper bound on days between the END of the
                reference period and publication.
    lag_verified  Date the lag was checked against the official calendar.
                Empty string means UNVERIFIED -- do not trust results.
    notes       Provenance and caveats. Read by humans, not code.
    """

    name: str
    source: Literal["abs", "rba"]
    collection: str
    search: dict[str, str] = field(default_factory=dict)
    series_id: str = ""
    freq: Freq = "M"
    transform: Transform = "pct"
    lag_days: int = 30
    lag_verified: str = ""
    notes: str = ""

    @property
    def is_target(self) -> bool:
        return self.name == "gdp_growth"


# ---------------------------------------------------------------------------
# TARGET
# ---------------------------------------------------------------------------
# Real GDP, chain volume measures, seasonally adjusted, quarterly.
# The ~63 day lag is why this project is a NOWCAST: by the time you learn GDP
# for quarter t, quarter t+1 has already finished.

TARGET = SeriesSpec(
    name="gdp_growth",
    source="abs",
    collection="5206.0",
    search={
        "Gross domestic product: Chain volume measures ;": "Data Item Description",
        "Seasonally Adjusted": "Series Type",
    },
    series_id="A2304402X",
    freq="Q",
    transform="pct",
    lag_days=65,
    lag_verified=LAGS_CHECKED_ON,
    notes=(
        "Table 5206001_Key_Aggregates. 1959Q3 onwards, $ millions. "
        "Observed releases: 2025Q4 -> 4 Mar 2026 (63d); 2026Q1 -> 3 Jun 2026 (64d); "
        "2026Q2 -> 2 Sep 2026 (64d); 2026Q3 -> 2 Dec 2026 (63d); "
        "2026Q4 -> 3 Mar 2027 (62d). Range 62-64, bound set to 65."
    ),
)


# ---------------------------------------------------------------------------
# MONTHLY PREDICTORS
# ---------------------------------------------------------------------------

_LFS_NOTE = (
    "Labour Force Survey, table 62020X28. Observed releases: Apr 2026 -> 21 May (21d); "
    "May -> 25 Jun (25d); Jun -> 23 Jul (23d); Jul -> 20 Aug (20d); Aug -> 24 Sep (24d). "
    "Range 20-25, bound set to 26. NOTE: an earlier assumption of 17 days was WRONG "
    "and would have leaked. Historically the release was the third Thursday "
    "(~15-21d); a constant 26 is conservative for older data."
)

MONTHLY_PREDICTORS: list[SeriesSpec] = [
    SeriesSpec(
        name="employment",
        source="abs",
        collection="6202.0",
        search={
            "Employed total ;  Persons ;  Australia ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A84423043C",
        freq="M",
        transform="pct",
        lag_days=26,
        lag_verified=LAGS_CHECKED_ON,
        notes="Australia total, '000 persons, 1978-02 onwards. " + _LFS_NOTE,
    ),
    SeriesSpec(
        name="unemployment_rate",
        source="abs",
        collection="6202.0",
        search={
            "Unemployment rate ;  Persons ;  Australia ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A84423050A",
        freq="M",
        transform="diff",
        lag_days=26,
        lag_verified=LAGS_CHECKED_ON,
        notes=(
            "Australia total, percent, 1978-02 onwards. A rate, so DIFFERENCE it "
            "rather than taking a percent change. " + _LFS_NOTE
        ),
    ),
    SeriesSpec(
        name="hours_worked",
        source="abs",
        collection="6202.0",
        search={
            "Monthly hours worked in all jobs ;  Persons ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A84426277X",
        freq="M",
        transform="pct",
        lag_days=26,
        lag_verified=LAGS_CHECKED_ON,
        notes=(
            "Table 62020017, '000 hours, 1978-07 onwards. Often tracks output better "
            "than headcount: firms adjust hours before they adjust bodies. " + _LFS_NOTE
        ),
    ),
    SeriesSpec(
        name="building_approvals",
        source="abs",
        collection="8731.0",
        search={
            "Total number of dwelling units ;  Total (Type of Building) ; "
            " Total Sectors ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A422070J",
        freq="M",
        transform="pct",
        lag_days=38,
        lag_verified=LAGS_CHECKED_ON,
        notes=(
            "Australia total dwelling units, table 8731006, 1983-07 onwards. "
            "Magnitude check: ~17,000/month nationally (a single state is ~1,000-2,000). "
            "Observed releases: Jan 2026 -> 3 Mar (31d); Mar -> 4 May (34d); "
            "May -> 1 Jul (31d); Jun -> 30 Jul (30d); Jul -> 1 Sep (32d); "
            "Aug -> 7 Oct (37d). ABS states 4-5 weeks after the FIRST day of the "
            "reference month. Range 30-37, bound set to 38. "
            "Volatile: one large apartment project moves the series."
        ),
    ),
    SeriesSpec(
        name="household_spending",
        source="abs",
        collection="5682.0",
        search={
            "Household spending ;  Total (Household Spending Categories) ; "
            " Australia ;  Current Price ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A130200584T",
        freq="M",
        transform="pct",
        lag_days=40,
        lag_verified=LAGS_CHECKED_ON,
        notes=(
            "Monthly Household Spending Indicator, table 5682001, 2012-07 onwards. "
            "REPLACES Retail Trade (8501.0), which the ABS ceased on 31 Jul 2025. "
            "CURRENT PRICE (nominal) -- the monthly MHSI has no volume measure; "
            "chain volume measures are quarterly only. So this moves with inflation "
            "as well as real activity. Possible extension: deflate by the monthly CPI. "
            "Observed releases: Feb 2026 -> 7 Apr (38d); Apr -> 28 May (28d); "
            "May -> 25 Jun (25d); Jun -> 4 Aug (35d); Jul -> 27 Aug (27d). "
            "Irregular because Mar/Jun/Sep/Dec editions are later to allow volume "
            "processing. Range 25-38, bound set to 40. "
            "SHORT HISTORY: starts 2012-07. Excluded from balanced-panel models."
        ),
    ),
    SeriesSpec(
        name="goods_exports",
        source="abs",
        collection="5368.0",
        search={
            "Credits, Total goods ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        series_id="A2718577A",
        freq="M",
        transform="pct",
        lag_days=40,
        lag_verified=LAGS_CHECKED_ON,
        notes=(
            "International Trade in Goods, table 536801, 1971-07 onwards, $ millions. "
            "'Credits' means exports; 'Debits' means imports. "
            "CURRENT PRICE (nominal), so it moves with commodity PRICES as well as "
            "volumes -- noisier than it looks for a real GDP target. "
            "Observed releases: Oct 2025 -> 4 Dec (34d); Jan 2026 -> 5 Mar (33d); "
            "Apr -> 4 Jun (35d); May -> 2 Jul (32d); Jun -> 6 Aug (37d). "
            "ABS states 4-6 weeks after the end of the reference month. "
            "Range 32-37, bound set to 40 to respect the stated 6-week upper end."
        ),
    ),
]


# ---------------------------------------------------------------------------
# FINANCIAL PREDICTORS  (not yet implemented in fetch.py)
# ---------------------------------------------------------------------------
# Financial variables are observed in real time -- lag_days = 0. Individually
# weak, but they fill the ragged edge at the very start of a quarter when no
# official statistic has arrived yet.

FINANCIAL_PREDICTORS: list[SeriesSpec] = [
    SeriesSpec(
        name="cash_rate",
        source="rba",
        collection="OCR",
        freq="M",
        transform="diff",
        lag_days=0,
        lag_verified="n/a - policy rate is public the day it is set",
        notes="Fetched via readabs.read_rba_ocr(monthly=True).",
    ),
    SeriesSpec(
        name="term_spread",
        source="rba",
        collection="F2",
        freq="M",
        transform="level",
        lag_days=0,
        lag_verified="n/a - market data, observed in real time",
        notes=(
            "10-year bond yield minus cash rate. Classic recession predictor. "
            "CONSTRUCTED in build_features from two RBA series, not fetched directly."
        ),
    ),
    SeriesSpec(
        name="credit_growth",
        source="rba",
        collection="D1",
        freq="M",
        transform="level",
        lag_days=35,
        lag_verified="",
        notes=(
            "RBA financial aggregates, published as a growth rate already, so "
            "transform='level'. LAG NOT YET VERIFIED -- check the RBA release "
            "schedule before using: https://www.rba.gov.au/statistics/tables/"
        ),
    ),
]


ALL_SPECS: list[SeriesSpec] = [TARGET, *MONTHLY_PREDICTORS, *FINANCIAL_PREDICTORS]

SPECS_BY_NAME: dict[str, SeriesSpec] = {s.name: s for s in ALL_SPECS}

# Series with enough history for a balanced panel (see notes on household_spending).
# Everything else goes only into models that handle missing data natively.
BALANCED_PANEL_START = "1983Q3"
SHORT_HISTORY = {"household_spending"}


def unverified_lags() -> list[str]:
    """Names of every series whose publication lag has not been checked."""
    return [s.name for s in ALL_SPECS if not s.lag_verified]


def pinned() -> list[SeriesSpec]:
    """Specs with a confirmed ABS Series ID."""
    return [s for s in ALL_SPECS if s.series_id]


if __name__ == "__main__":
    print(f"{len(ALL_SPECS)} series registered, {len(pinned())} pinned.")
    print(f"Target: {TARGET.name} [{TARGET.series_id}] lag {TARGET.lag_days}d")
    print(f"\nBalanced panel starts {BALANCED_PANEL_START}")
    print(f"Excluded from balanced panel: {', '.join(sorted(SHORT_HISTORY))}")

    missing = unverified_lags()
    if missing:
        print(f"\nUNVERIFIED publication lags ({len(missing)}):")
        for n in missing:
            print(f"  - {n}: {SPECS_BY_NAME[n].lag_days} days (assumed)")
    else:
        print("\nAll publication lags verified.")
