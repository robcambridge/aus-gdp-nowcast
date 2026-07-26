"""Turn raw levels into stationary series, and assemble the point-in-time panel.

WHEN TO TRANSFORM: BEFORE BUILDING THE PANEL
--------------------------------------------
A percent change for month t uses months t and t-1. Both were published by the
time month t was published, so the transformed value becomes available exactly
when the raw value for month t did. Transforming first and stamping second is
therefore correct and does not leak.

The order is:
    raw levels -> transform -> attach publication dates -> panel -> as_of()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SHORT_HISTORY, SeriesSpec
from .dataset import build_panel


def transform_series(series: pd.Series, how: str) -> pd.Series:
    """Apply one stationarity transform.

    "pct"      percent change on the previous period (levels: employment, GDP)
    "diff"     first difference (rates: unemployment rate, cash rate)
    "log_pct"  100 * change in logs (similar to pct, better for volatile series)
    "level"    leave alone (already a growth rate, e.g. RBA credit growth)
    """
    if how == "pct":
        out = series.pct_change() * 100
    elif how == "diff":
        out = series.diff()
    elif how == "log_pct":
        if (series <= 0).any():
            raise ValueError("log_pct needs strictly positive values")
        out = np.log(series).diff() * 100
    elif how == "level":
        out = series.copy()
    else:
        raise ValueError(f"Unknown transform: {how!r}")

    out.name = series.name
    return out.dropna()


def transform_all(
    raw: dict[str, pd.Series], specs: dict[str, SeriesSpec]
) -> dict[str, pd.Series]:
    """Transform every raw series according to its SeriesSpec.

    Kept for direct use and testing. Note this applies `spec.transform` to
    EVERYTHING, which is only right for quarterly series. For building the
    panel use `prepare_for_panel` instead.
    """
    missing = set(raw) - set(specs)
    if missing:
        raise KeyError(f"No SeriesSpec for: {sorted(missing)}")
    return {name: transform_series(s, specs[name].transform) for name, s in raw.items()}


def prepare_for_panel(
    raw: dict[str, pd.Series], specs: dict[str, SeriesSpec]
) -> dict[str, pd.Series]:
    """Put each series into the form the panel should store.

    QUARTERLY series (the target) are transformed here: GDP growth for quarter
    q uses q and q-1, both known when q is published, so transforming before
    stamping is correct.

    MONTHLY series are left as RAW LEVELS. Their transform happens later, after
    quarterly aggregation, because averaging three month-on-month growth rates
    is a different quantity from the change in the quarterly average level --
    and the latter is how GDP itself is measured. Averaging levels first also
    smooths out survey noise that would otherwise dominate a single month.
    """
    missing = set(raw) - set(specs)
    if missing:
        raise KeyError(f"No SeriesSpec for: {sorted(missing)}")

    out = {}
    for name, series in raw.items():
        spec = specs[name]
        out[name] = series.dropna() if spec.stored_as_level else transform_series(
            series, spec.transform
        )
    return out


def make_panel(raw: dict[str, pd.Series], specs: dict[str, SeriesSpec]) -> pd.DataFrame:
    """Full pipeline: raw -> panel-ready form -> stamped long panel.

    Monthly columns in the resulting panel hold LEVELS. Convert them with
    `quarterly_regressor` (or bridge.partial_quarterly_growth) at model time.
    """
    return build_panel(prepare_for_panel(raw, specs), specs)


def quarterly_regressor(
    monthly_levels: pd.Series, aggregation: str, n_months: int = 3
) -> pd.Series:
    """Turn monthly levels into the quarterly regressor a bridge equation wants.

    The first `n_months` of each quarter are averaged, then compared against the
    PREVIOUS quarter's FULL three-month average:

        mean_then_pct   100 * (partial_t / full_{t-1} - 1)
        mean_then_diff  partial_t - full_{t-1}
        mean            partial_t   (already a rate of change)

    Comparing a partial current quarter against a complete previous quarter is
    the standard bridge construction, and it is what makes a single month
    useful: one noisy month is measured against a smooth three-month base.

    Applied identically across the whole history, so estimation and prediction
    see the same kind of number.
    """
    from .bridge import partial_quarterly_mean

    partial = partial_quarterly_mean(monthly_levels, n_months)
    if aggregation == "mean":
        return partial

    full = partial_quarterly_mean(monthly_levels, 3)
    base = full.shift(1).reindex(partial.index)

    if aggregation == "mean_then_pct":
        out = (partial / base - 1) * 100
    elif aggregation == "mean_then_diff":
        out = partial - base
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# Stationarity testing
# ---------------------------------------------------------------------------


def adf_table(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Augmented Dickey-Fuller test on each series.

    The null hypothesis is that the series has a unit root (is NOT stationary).
    A p-value below 0.05 lets you reject that, which is what you want after
    transforming.

    Put this table in your report. If something fails, say so and explain what
    you did about it rather than quietly ignoring it.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for name, s in sorted(series_map.items()):
        clean = s.dropna()
        if len(clean) < 20:
            rows.append({"series": name, "n": len(clean), "adf_stat": np.nan,
                         "p_value": np.nan, "stationary_5pct": None,
                         "note": "too short to test"})
            continue
        stat, pval, *_ = adfuller(clean.to_numpy(), autolag="AIC")
        rows.append(
            {
                "series": name,
                "n": len(clean),
                "adf_stat": round(stat, 3),
                "p_value": round(pval, 4),
                "stationary_5pct": bool(pval < 0.05),
                "note": "" if pval < 0.05 else "FAILS - investigate",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monthly -> quarterly, respecting the ragged edge
# ---------------------------------------------------------------------------


def monthly_to_quarterly(
    monthly: pd.DataFrame, min_months: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average monthly observations within each quarter.

    The ragged edge means the most recent quarter is usually INCOMPLETE -- you
    might hold one or two of its three months. We average whatever is there
    rather than discarding the quarter, because a partial read on the current
    quarter is the whole point of nowcasting.

    Returns
    -------
    values : quarterly averages, PeriodIndex freq="Q"
    counts : how many months went into each average (1, 2 or 3)

    `counts` matters. A quarter built from one month is a weaker signal than one
    built from three, and you should be able to show that in your results.
    """
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()

    q_index = monthly.index.asfreq("Q")
    grouped = monthly.groupby(q_index)

    values = grouped.mean()
    counts = grouped.count()

    values = values.where(counts >= min_months)
    values.index = pd.PeriodIndex(values.index, freq="Q")
    counts.index = pd.PeriodIndex(counts.index, freq="Q")
    values.index.name = counts.index.name = "ref_period"
    return values, counts


def balanced_features(
    snapshot_monthly: pd.DataFrame, exclude: set[str] | None = None
) -> pd.DataFrame:
    """Quarterly features for models that need a complete rectangle.

    Drops the short-history series (household spending) so the panel can start
    in 1983 rather than 2012. Models that handle missing data natively -- the
    dynamic factor model -- should use everything instead.
    """
    exclude = SHORT_HISTORY if exclude is None else exclude
    keep = [c for c in snapshot_monthly.columns if c not in exclude]
    values, _ = monthly_to_quarterly(snapshot_monthly[keep])
    return values
