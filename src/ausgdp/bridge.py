"""Bridge equations: the first models that use the monthly indicators.

THE IDEA
--------
GDP is quarterly and late. Employment is monthly and early. A bridge equation
"bridges" the frequency gap in the simplest possible way:

    1. Average the monthly indicator within each quarter.
    2. Regress quarterly GDP growth on that quarterly average.
    3. For the quarter you are nowcasting, plug in the average of whatever
       months have arrived so far.

Standing on the day June-quarter GDP is published (early September), you may
already hold July and August employment. The September quarter is not over,
but you have a partial read on it that GDP itself cannot give you for another
three months. That is the entire informational edge of this project.

THE SUBTLETY THAT MOST IMPLEMENTATIONS GET WRONG
------------------------------------------------
If you estimate the regression on three-month averages but predict from a
one-month average, you have a mismatch: a one-month average is noisier, so the
coefficient you estimated is wrong for the input you are feeding it. The model
will be overconfident.

The fix used here is to make estimation and prediction consistent. If only two
months of the target quarter have arrived, we rebuild the ENTIRE history using
two-month averages, then estimate. The regression then sees exactly the kind of
input it will be asked to predict from.

This means re-estimating whenever the number of available months changes, which
is why `partial_quarterly_mean` takes `n_months`.

WHY BRIDGES BEFORE ANYTHING FANCIER
-----------------------------------
They are transparent, fast, and surprisingly hard to beat. If a dynamic factor
model cannot beat a simple average of bridge equations, the factor model is not
earning its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmarks import Context, f_mean


def partial_quarterly_mean(monthly: pd.Series, n_months: int) -> pd.Series:
    """Average the FIRST `n_months` of each quarter.

    n_months=3 gives the usual full-quarter average. n_months=1 uses only the
    first month of each quarter -- which is what you hold early in a quarter.

    >>> s = pd.Series(range(6), index=pd.period_range("2024-01", periods=6, freq="M"))
    >>> partial_quarterly_mean(s, 1).to_list()   # Jan and Apr only
    [0.0, 3.0]
    """
    if monthly.empty:
        return pd.Series(dtype=float)
    if not 1 <= n_months <= 3:
        raise ValueError(f"n_months must be 1, 2 or 3; got {n_months}")

    clean = monthly.dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    position_in_quarter = (clean.index.month - 1) % 3  # 0, 1, 2
    keep = position_in_quarter < n_months
    kept = clean[keep]
    if kept.empty:
        return pd.Series(dtype=float)

    quarters = kept.index.asfreq("Q")
    out = kept.groupby(quarters).mean()
    out.index = pd.PeriodIndex(out.index, freq="Q")
    return out.sort_index()


def months_available(monthly: pd.Series, quarter: pd.Period) -> int:
    """How many months of `quarter` have been published."""
    if monthly.empty:
        return 0
    in_quarter = monthly.dropna().index.asfreq("Q") == quarter
    return int(in_quarter.sum())


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Least squares with an intercept. Returns None if the fit is degenerate."""
    design = np.column_stack([np.ones(len(X)), X])
    if design.shape[0] <= design.shape[1] + 2:
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


def bridge_forecast(
    ctx: Context,
    indicator: str,
    add_ar: bool = True,
    min_obs: int = 40,
    window: int | None = None,
    aggregation: str | None = None,
) -> float:
    """One bridge equation for one indicator.

    Model:   gdp_t = a + b * indicator_t (+ c * gdp_{t-1})

    where indicator_t is built from monthly LEVELS: average the first n months
    of quarter t, then compare against quarter t-1's full three-month average.
    n is however many months of the TARGET quarter are currently available, and
    the whole history is rebuilt the same way so estimation and prediction see
    the same kind of number.

    Parameters
    ----------
    window : if set, estimate on the last `window` quarters only.

        WHY A ROLLING WINDOW MATTERS HERE
        ---------------------------------
        The intercept `a` absorbs average growth over the estimation sample.
        Estimated over an expanding window back to the 1980s, it anchors on a
        period when Australian trend growth was materially higher, and the
        forecast inherits an upward bias. This is the same effect that makes a
        rolling mean beat an expanding mean on this data -- so any regression
        with a fixed intercept needs the same treatment.

    add_ar : include a lag of GDP growth. Worth turning OFF when the target is
        near white noise: an AR coefficient that is really zero still costs
        estimation variance.

    Falls back to the historical mean if the indicator is missing, the target
    quarter has no months yet, or the sample is too short.
    """
    from .config import SPECS_BY_NAME
    from .transforms import quarterly_regressor

    monthly = ctx.snapshot.monthly
    if monthly.empty or indicator not in monthly.columns:
        return f_mean(ctx.y)

    series = monthly[indicator].dropna()
    n_avail = months_available(series, ctx.target)
    if n_avail == 0:
        return f_mean(ctx.y)

    if aggregation is None:
        spec = SPECS_BY_NAME.get(indicator)
        aggregation = spec.aggregation if spec else "mean_then_pct"

    x = quarterly_regressor(series, aggregation, n_avail)
    if ctx.target not in x.index or not np.isfinite(x.loc[ctx.target]):
        return f_mean(ctx.y)

    x_target = float(x.loc[ctx.target])
    y = ctx.y

    frame = pd.DataFrame({"y": y, "x": x})
    if add_ar:
        frame["y_lag"] = y.shift(1)
    frame = frame.dropna()

    if len(frame) < min_obs:
        return f_mean(ctx.y)

    if window is not None:
        frame = frame.iloc[-window:]
        if len(frame) < min_obs:
            return f_mean(ctx.y)

    cols = ["x", "y_lag"] if add_ar else ["x"]
    beta = _ols(frame[cols].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float))
    if beta is None:
        return f_mean(ctx.y)

    inputs = [x_target]
    if add_ar:
        inputs.append(float(y.iloc[-1]))

    pred = beta[0] + float(np.dot(beta[1:], inputs))
    return pred if np.isfinite(pred) else f_mean(ctx.y)


def make_bridge(indicator: str, add_ar: bool = True, window: int | None = None):
    """Build a named forecaster for one indicator."""

    def forecaster(ctx: Context) -> float:
        return bridge_forecast(ctx, indicator, add_ar=add_ar, window=window)

    forecaster.__name__ = f"bridge_{indicator}"
    forecaster.__doc__ = (
        f"Bridge equation on {indicator} "
        f"(window={window or 'expanding'}, ar={add_ar})."
    )
    return forecaster


def make_bridge_average(
    indicators: list[str], add_ar: bool = True, window: int | None = None
):
    """Average the forecasts of several bridge equations.

    Forecast combination is one of the most reliable results in the whole
    forecasting literature: a simple average of several decent models usually
    beats trying to pick the single best one in advance, because it cancels
    idiosyncratic errors. Expect this to be among your strongest models.
    """

    def forecaster(ctx: Context) -> float:
        preds = [
            bridge_forecast(ctx, name, add_ar=add_ar, window=window)
            for name in indicators
        ]
        preds = [p for p in preds if np.isfinite(p)]
        return float(np.mean(preds)) if preds else f_mean(ctx.y)

    forecaster.__name__ = "bridge_average"
    forecaster.__doc__ = f"Equal-weighted average of bridges on {indicators}."
    return forecaster


def make_ridge(
    indicators: list[str],
    alpha: float = 1.0,
    min_obs: int = 40,
    window: int | None = None,
    add_ar: bool = True,
):
    """Ridge regression on all indicators at once, plus a lag of GDP.

    Bridges use one indicator each; this uses them jointly. Ridge rather than
    OLS because the indicators are correlated with each other and the sample is
    short -- shrinkage keeps the coefficients stable.

    Standardisation happens INSIDE this function, on the visible sample only.
    Scaling on the full sample first would leak the future into every forecast.
    """

    def forecaster(ctx: Context) -> float:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from .config import SPECS_BY_NAME
        from .transforms import quarterly_regressor

        monthly = ctx.snapshot.monthly
        if monthly.empty:
            return f_mean(ctx.y)

        available = [c for c in indicators if c in monthly.columns]
        if not available:
            return f_mean(ctx.y)

        # Use the smallest month-count across indicators so every column is
        # built the same way for the target quarter.
        counts = [months_available(monthly[c].dropna(), ctx.target) for c in available]
        n_avail = min(counts)
        if n_avail == 0:
            return f_mean(ctx.y)

        cols = {}
        for c in available:
            spec = SPECS_BY_NAME.get(c)
            agg = spec.aggregation if spec else "mean_then_pct"
            cols[c] = quarterly_regressor(monthly[c].dropna(), agg, n_avail)

        X = pd.DataFrame(cols)
        if ctx.target not in X.index:
            return f_mean(ctx.y)

        frame = X.join(ctx.y.rename("y"), how="inner")
        if add_ar:
            frame["y_lag"] = ctx.y.shift(1)
        frame = frame.dropna()
        if len(frame) < min_obs:
            return f_mean(ctx.y)

        if window is not None:
            frame = frame.iloc[-window:]
            if len(frame) < min_obs:
                return f_mean(ctx.y)

        features = [*available, "y_lag"] if add_ar else list(available)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(frame[features].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float))

        target_row = X.loc[[ctx.target]].copy()
        if add_ar:
            target_row["y_lag"] = float(ctx.y.iloc[-1])
        if target_row[features].isna().any(axis=None):
            return f_mean(ctx.y)

        pred = float(model.predict(target_row[features].to_numpy(dtype=float))[0])
        return pred if np.isfinite(pred) else f_mean(ctx.y)

    forecaster.__name__ = "ridge"
    forecaster.__doc__ = f"Ridge (alpha={alpha}) on {indicators} plus a GDP lag."
    return forecaster
