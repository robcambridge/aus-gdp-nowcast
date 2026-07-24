"""Point-in-time data handling: the ragged edge.

THE PROBLEM
-----------
On 3 September 2024 the ABS publishes June-quarter GDP. Standing at that
moment, what did you actually know?

  * June-quarter GDP            -> just published, you know it
  * September-quarter GDP       -> the quarter is not even over; unknown
  * July labour force           -> published mid-August, you know it
  * August labour force         -> not until mid-September, unknown
  * July retail turnover        -> published early September, you know it
  * July building approvals     -> published early September, maybe

Different series stop at different points. Lay them out as a table and the
bottom-right corner is missing in an uneven, staircase pattern. That pattern is
called the RAGGED EDGE, and handling it correctly is the whole game. Get it
wrong -- let one August observation leak into a forecast made in early
September -- and your model looks brilliant for reasons that have nothing to do
with economics.

THE APPROACH
------------
We never store a plain wide table. We store a LONG table where every single
observation carries the date it became public:

    series      ref_period  ref_end     value  available_from
    employment  2024-07     2024-07-31   14.2  2024-08-17
    employment  2024-08     2024-08-31   14.3  2024-09-17
    gdp_growth  2024Q2      2024-06-30    0.3  2024-09-03

Then `as_of(panel, "2024-09-03")` keeps only rows whose `available_from` is on
or before that date. The ragged edge falls out automatically, and leakage
becomes structurally impossible rather than something you have to remember.

Monthly and quarterly observations are kept in SEPARATE frames, because
(a) pandas cannot sensibly index one axis with two different Period
frequencies, and (b) statsmodels' DynamicFactorMQ wants them separately
anyway: DynamicFactorMQ(endog=<monthly>, endog_quarterly=<quarterly>).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SeriesSpec

LONG_COLUMNS = ["series", "ref_period", "ref_end", "freq", "value", "available_from"]


def availability_date(period: pd.Period, lag_days: int) -> pd.Timestamp:
    """The date an observation for `period` became publicly known.

    A statistic describing July 2024 cannot be published before July ends.
    So we take the last day of the reference period and add the publication
    lag.

    >>> availability_date(pd.Period("2024-07", freq="M"), 17)
    Timestamp('2024-08-17 00:00:00')
    """
    end_of_period = period.end_time.normalize()
    return end_of_period + pd.Timedelta(days=lag_days)


@dataclass(frozen=True)
class Snapshot:
    """Everything that was publicly known on a given date. Nothing more.

    Attributes
    ----------
    vintage   : the as-of date this snapshot represents.
    monthly   : wide frame, PeriodIndex freq="M", one column per monthly series.
    quarterly : wide frame, PeriodIndex freq="Q", one column per quarterly series.
    """

    vintage: pd.Timestamp
    monthly: pd.DataFrame
    quarterly: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Snapshot(vintage={self.vintage.date()}, "
            f"monthly={self.monthly.shape}, quarterly={self.quarterly.shape})"
        )

    @property
    def n_values(self) -> int:
        """Total number of non-missing observations visible at this vintage."""
        return int(
            self.monthly.notna().sum().sum() + self.quarterly.notna().sum().sum()
        )

    @property
    def is_empty(self) -> bool:
        return self.monthly.empty and self.quarterly.empty


def to_long(series: pd.Series, spec: SeriesSpec) -> pd.DataFrame:
    """Convert one series with a PeriodIndex into the long, stamped format.

    Parameters
    ----------
    series : pd.Series indexed by pd.PeriodIndex (what readabs gives you).
    spec   : the SeriesSpec describing it, including its publication lag.
    """
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError(
            f"{spec.name}: expected a PeriodIndex, got {type(series.index).__name__}. "
            "readabs returns PeriodIndex; if you built this series yourself use "
            "pd.PeriodIndex(..., freq='M') or freq='Q'."
        )

    clean = series.dropna()
    return pd.DataFrame(
        {
            "series": spec.name,
            "ref_period": [str(p) for p in clean.index],
            "ref_end": [p.end_time.normalize() for p in clean.index],
            "freq": spec.freq,
            "value": clean.to_numpy(),
            "available_from": [
                availability_date(p, spec.lag_days) for p in clean.index
            ],
        }
    )[LONG_COLUMNS]


def build_panel(
    series_map: dict[str, pd.Series], specs: dict[str, SeriesSpec]
) -> pd.DataFrame:
    """Stack many series into one long panel.

    Parameters
    ----------
    series_map : {name: pd.Series}
    specs      : {name: SeriesSpec}
    """
    missing = set(series_map) - set(specs)
    if missing:
        raise KeyError(f"No SeriesSpec registered for: {sorted(missing)}")

    parts = [to_long(s, specs[name]) for name, s in series_map.items()]
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["series", "ref_end"]).reset_index(drop=True)


def _pivot(visible: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Pivot one frequency block into a wide frame with a PeriodIndex."""
    block = visible.loc[visible["freq"] == freq]
    if block.empty:
        return pd.DataFrame()

    wide = block.pivot_table(
        index="ref_period", columns="series", values="value", aggfunc="last"
    )
    wide.index = pd.PeriodIndex(wide.index, freq=freq)
    wide.index.name = "ref_period"
    wide.columns.name = None
    return wide.sort_index()


def as_of(panel: pd.DataFrame, when: str | pd.Timestamp) -> Snapshot:
    """Everything that was publicly known on `when`, and nothing else.

    Parameters
    ----------
    panel : long panel from build_panel()
    when  : the vintage date, e.g. "2024-09-03"

    Returns
    -------
    Snapshot, whose `.monthly` frame has a ragged bottom edge -- later-
    publishing series carry more trailing NaNs than fast ones.
    """
    cutoff = pd.Timestamp(when)
    visible = panel.loc[panel["available_from"] <= cutoff]
    return Snapshot(
        vintage=cutoff,
        monthly=_pivot(visible, "M"),
        quarterly=_pivot(visible, "Q"),
    )


def ragged_edge_report(panel: pd.DataFrame, when: str | pd.Timestamp) -> pd.DataFrame:
    """Diagnostic: how stale is each series at a given vintage date?

    This is the table to eyeball whenever a result looks too good. If a series
    shows a last observation it could not plausibly have had, your lag is wrong.
    """
    cutoff = pd.Timestamp(when)
    visible = panel.loc[panel["available_from"] <= cutoff]
    if visible.empty:
        return pd.DataFrame()

    rows = []
    for name, grp in visible.groupby("series"):
        last = grp.loc[grp["ref_end"].idxmax()]
        rows.append(
            {
                "series": name,
                "freq": last["freq"],
                "last_ref_period": last["ref_period"],
                "published_on": last["available_from"].date(),
                "days_stale": (cutoff - last["available_from"]).days,
                "n_obs": len(grp),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("published_on", ascending=False)
        .reset_index(drop=True)
    )


def target_release_dates(panel: pd.DataFrame, target: str = "gdp_growth") -> pd.Series:
    """The dates on which each GDP figure was published.

    These are your natural backtest vintage dates: one forecast per GDP
    release, using exactly the information available that morning.
    """
    tgt = panel.loc[panel["series"] == target]
    if tgt.empty:
        raise ValueError(f"Series '{target}' not found in panel.")
    return tgt.set_index("ref_period")["available_from"].sort_values()
