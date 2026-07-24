"""Series registry and publication-lag table.

This module is the single source of truth for:
  1. WHICH series we use,
  2. WHERE each one comes from,
  3. HOW LONG after its reference period each one is actually published.

(3) is the part that makes or breaks a nowcasting project. If you assume a
series is available earlier than it really was, your backtest sees the future
and your results are fiction. Every lag below must be checked against the
ABS/RBA release calendar before you trust a single result.

    ABS release calendar: https://www.abs.gov.au/release-calendar
    RBA statistical tables: https://www.rba.gov.au/statistics/tables/

IMPORTANT CAVEATS
-----------------
* The `lag_days` values below are STARTING ESTIMATES, not verified facts.
  Confirm each one and record the date you checked it in `lag_verified`.
* Publication lags have changed over history. ABS releases were slower in the
  1980s than today. We assume a constant lag, which is mildly optimistic for
  early samples. Disclose this in your README.
* ABS collections get restructured. Retail Trade and the monthly CPI have both
  been reworked in recent years. Run `scripts/01_discover.py` to see what
  actually exists today rather than trusting any hardcoded catalogue number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Freq = Literal["M", "Q"]
Transform = Literal["level", "pct", "diff", "log_pct"]


@dataclass(frozen=True)
class SeriesSpec:
    """One economic time series and everything we need to use it honestly.

    Attributes
    ----------
    name        Short identifier used as the column name throughout the project.
    source      "abs" or "rba".
    collection  ABS catalogue number (e.g. "6202.0") or RBA table id (e.g. "F1").
    search      Metadata search terms used to locate the series. For ABS these
                map {search_value: metadata_column}. Preferred over hardcoded
                series IDs, which are easy to mistype and hard to audit.
    series_id   Optional exact ABS Series ID. Fill this in once you have
                confirmed it via scripts/01_discover.py -- it makes the fetch
                fast and unambiguous.
    freq        Native frequency. Do NOT pre-aggregate monthly data to
                quarterly; the whole point is to keep the monthly timing.
    transform   How to make it stationary. "pct" = percent change on the
                previous period, "diff" = first difference, "level" = leave it.
    lag_days    Days after the END of the reference period before the number is
                published. This is the field that prevents look-ahead bias.
    lag_verified  Date (YYYY-MM-DD) you last checked lag_days against the
                official release calendar. Empty string means UNVERIFIED.
    notes       Anything a reader of your README would want to know.
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
# Real GDP, chain volume measures, seasonally adjusted, from the quarterly
# National Accounts. The ~65 day lag is why this project is a NOWCAST: by the
# time you learn GDP for quarter t, quarter t+1 has already finished.

TARGET = SeriesSpec(
    name="gdp_growth",
    source="abs",
    collection="5206.0",
    search={
        "Gross domestic product: Chain volume measures ;": "Data Item Description",
        "Seasonally Adjusted": "Series Type",
    },
    freq="Q",
    transform="pct",
    lag_days=65,
    lag_verified="",
    notes="June quarter is published in early September => roughly 65 days.",
)


# ---------------------------------------------------------------------------
# MONTHLY PREDICTORS
# ---------------------------------------------------------------------------
# These are the source of your informational edge. Each one tells you something
# about the current quarter well before GDP is published. Ordered roughly by
# how quickly they arrive.

MONTHLY_PREDICTORS: list[SeriesSpec] = [
    SeriesSpec(
        name="employment",
        source="abs",
        collection="6202.0",
        search={
            "Employed total ;  Persons ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="pct",
        lag_days=17,
        notes="Labour Force Survey, released ~3rd Thursday of following month. "
        "One of the fastest and most informative monthly indicators.",
    ),
    SeriesSpec(
        name="unemployment_rate",
        source="abs",
        collection="6202.0",
        search={
            "Unemployment rate ;  Persons ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="diff",
        lag_days=17,
        notes="A rate, so difference it rather than taking a percent change.",
    ),
    SeriesSpec(
        name="hours_worked",
        source="abs",
        collection="6202.0",
        search={
            "Monthly hours worked in all jobs ;  Persons ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="pct",
        lag_days=17,
        notes="Often tracks output better than headcount, because firms adjust "
        "hours before they adjust bodies.",
    ),
    SeriesSpec(
        name="building_approvals",
        source="abs",
        collection="8731.0",
        search={
            "Total number of dwelling units ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="pct",
        lag_days=33,
        notes="Volatile but genuinely forward-looking for dwelling investment.",
    ),
    SeriesSpec(
        name="retail_turnover",
        source="abs",
        collection="8501.0",
        search={
            "Turnover ;  Total (State) ;  Total (Industry) ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="pct",
        lag_days=32,
        notes="CHECK THIS ONE. ABS has been transitioning retail trade toward "
        "the Monthly Household Spending Indicator. Run 01_discover.py.",
    ),
    SeriesSpec(
        name="goods_exports",
        source="abs",
        collection="5368.0",
        search={
            "Goods ;  Credits ;": "Data Item Description",
            "Seasonally Adjusted": "Series Type",
        },
        freq="M",
        transform="pct",
        lag_days=35,
        notes="Australia is a commodity exporter; net exports swing GDP a lot.",
    ),
]


# ---------------------------------------------------------------------------
# FINANCIAL / DAILY-ISH PREDICTORS
# ---------------------------------------------------------------------------
# Financial variables are observed essentially in real time -- lag_days = 0.
# They are weak predictors individually but they fill the ragged edge at the
# very start of a quarter, when nothing else has arrived yet.

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
        notes="10-year bond yield minus cash rate. Classic recession predictor. "
        "Constructed in build_features, not fetched directly.",
    ),
    SeriesSpec(
        name="credit_growth",
        source="rba",
        collection="D1",
        freq="M",
        transform="level",
        lag_days=32,
        notes="RBA financial aggregates. Already published as a growth rate, "
        "so transform='level'.",
    ),
]


ALL_SPECS: list[SeriesSpec] = [TARGET, *MONTHLY_PREDICTORS, *FINANCIAL_PREDICTORS]

# Convenience lookup
SPECS_BY_NAME: dict[str, SeriesSpec] = {s.name: s for s in ALL_SPECS}


def unverified_lags() -> list[str]:
    """Names of every series whose publication lag you have not yet checked.

    Call this before you believe any backtest result. If the list is non-empty,
    your evaluation may be leaking future information.
    """
    return [s.name for s in ALL_SPECS if not s.lag_verified]


if __name__ == "__main__":
    print(f"{len(ALL_SPECS)} series registered.")
    print(f"Target: {TARGET.name} (lag {TARGET.lag_days}d)")
    missing = unverified_lags()
    if missing:
        print(f"\nUNVERIFIED publication lags ({len(missing)}):")
        for n in missing:
            print(f"  - {n}: {SPECS_BY_NAME[n].lag_days} days (assumed)")
        print("\nVerify these against https://www.abs.gov.au/release-calendar")
