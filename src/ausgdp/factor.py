"""Dynamic factor model: pool all indicators through one latent factor.

WHY A FACTOR MODEL, AFTER BRIDGES
---------------------------------
A bridge equation reads each indicator in isolation. But employment, hours,
approvals and exports are all noisy reflections of one underlying thing -- the
state of the business cycle. A dynamic factor model says exactly that: a small
number of unobserved factors drive every series, and each observed series is
the factor plus idiosyncratic noise. Estimating the common factor averages
away noise that no single indicator can escape.

The MQ ("mixed-frequency") variant is the reason this is worth the complexity:

  * it takes monthly and quarterly series together, at their native
    frequencies, with no pre-aggregation;
  * it handles the ragged edge natively via the Kalman filter -- a series that
    stops early simply contributes no observation for its missing months, and
    the filter fills the gap from the factor;
  * it handles the short left edge too, so household spending (from 2012) can
    join the panel and contribute only where it exists.

This is the architecture behind the New York Fed Staff Nowcast.

WHAT TO EXPECT
--------------
On this data, do not expect the DFM to crush the bridge combination on RMSE.
The value is twofold: a single principled model that uses every series at once
including the short one, and the `news` decomposition -- attributing a change
in the nowcast to the specific data release that caused it, which is what a
research analyst actually does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .benchmarks import Context, f_mean

# statsmodels' state-space code emits a ValueWarning about non-monotonic
# PeriodIndex on every forecast call. It is harmless here -- our index is
# monotonic; statsmodels just cannot tell for a PeriodIndex -- and at hundreds
# of vintages it drowns the console. Silence this one specific warning.
_SS_WARN = "A date index has been provided"


def _fit_quietly(model, maxiter: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", message=_SS_WARN)
        return model.fit(disp=False, maxiter=maxiter)


def _build_inputs(
    ctx: Context,
    monthly_series: list[str],
    quarterly_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Assemble the monthly and quarterly frames the DFM needs, as LEVELS.

    Monthly indicators are stored as levels in the panel; the DFM wants levels
    (it models the common factor directly), so we pass them through. The target
    is quarterly growth. Everything is trimmed to what the snapshot makes
    visible, preserving the ragged edge.
    """
    monthly = ctx.snapshot.monthly
    available = [c for c in monthly_series if c in monthly.columns]
    if not available:
        return None

    m = monthly[available].copy()
    m = m.loc[m.dropna(how="all").index]
    if m.empty or len(m) < 24:
        return None

    q = ctx.y.rename(quarterly_target).to_frame()
    return m, q


def dfm_forecast(
    ctx: Context,
    monthly_series: list[str],
    target: str = "gdp_growth",
    factors: int = 1,
    factor_orders: int = 2,
    maxiter: int = 100,
) -> float:
    """Nowcast the target quarter with a mixed-frequency dynamic factor model.

    Falls back to the historical mean on any failure -- a non-converging EM run
    or a degenerate snapshot should never crash the backtest.
    """
    try:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    except ImportError:
        return f_mean(ctx.y)

    built = _build_inputs(ctx, monthly_series, target)
    if built is None:
        return f_mean(ctx.y)
    m, q = built

    try:
        model = DynamicFactorMQ(
            m,
            endog_quarterly=q,
            factors=factors,
            factor_orders=factor_orders,
            idiosyncratic_ar1=False,  # simpler, more stable on short samples
        )
        res = _fit_quietly(model, maxiter)
    except Exception:  # noqa: BLE001 - any estimation failure -> fall back
        return f_mean(ctx.y)

    # How many months past the last monthly observation is the target quarter?
    steps = _steps_to_quarter(m.index[-1], ctx.target)
    if steps <= 0:
        return f_mean(ctx.y)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = res.forecast(steps=steps)
    except Exception:  # noqa: BLE001
        return f_mean(ctx.y)

    pred = _extract_quarter(fc, target, ctx.target)
    if pred is None or not np.isfinite(pred):
        return f_mean(ctx.y)
    return float(pred)


def _steps_to_quarter(last_month: pd.Period, target_q: pd.Period) -> int:
    """Number of monthly steps from last_month to the end of target_q."""
    target_end_month = pd.Period(target_q.end_time, freq="M")
    return (target_end_month.year - last_month.year) * 12 + (
        target_end_month.month - last_month.month
    )


def _extract_quarter(forecast: pd.DataFrame, target: str, target_q: pd.Period):
    """Pull the target-quarter value of the target series from a DFM forecast.

    DynamicFactorMQ returns a monthly-indexed frame; the quarterly series is
    repeated across the quarter's months, so any month of the target quarter
    carries its value.
    """
    if target not in forecast.columns:
        return None
    q_of_index = forecast.index.asfreq("Q")
    mask = q_of_index == target_q
    if not mask.any():
        return None
    return forecast.loc[mask, target].iloc[-1]


def make_dfm(
    monthly_series: list[str],
    target: str = "gdp_growth",
    factors: int = 1,
    factor_orders: int = 2,
):
    """Build a named DFM forecaster for the horse race."""

    def forecaster(ctx: Context) -> float:
        return dfm_forecast(
            ctx, monthly_series, target=target,
            factors=factors, factor_orders=factor_orders,
        )

    forecaster.__name__ = f"dfm_{factors}f"
    forecaster.__doc__ = (
        f"Dynamic factor model: {factors} factor(s), order {factor_orders}, "
        f"on {monthly_series} + {target}."
    )
    return forecaster


def news_decomposition(
    monthly_old: pd.DataFrame,
    monthly_new: pd.DataFrame,
    quarterly: pd.DataFrame,
    impact_quarter: pd.Period,
    target: str = "gdp_growth",
    factors: int = 1,
    factor_orders: int = 2,
    maxiter: int = 100,
) -> pd.DataFrame | None:
    """Attribute a change in the nowcast to the data releases that caused it.

    This is what a nowcasting desk actually produces. Between two vintages new
    monthly numbers arrive; each differs from what the model expected; that
    surprise ("news") moves the nowcast by an amount equal to the surprise times
    the weight the Kalman filter places on that series. This function returns
    one row per (release month, indicator) with:

        news    the surprise: observed value minus the model's prior forecast
        weight  how much the filter lets that series move the nowcast
        impact  news * weight -- the contribution to the change in the nowcast

    Sum of `impact` = total revision to the nowcast between the two vintages.

    Parameters
    ----------
    monthly_old : monthly panel at the earlier vintage (fewer observations)
    monthly_new : monthly panel at the later vintage (more observations)
    quarterly   : the quarterly target, identical for both (only monthly data
                  arrived between the vintages)
    impact_quarter : the quarter whose nowcast we are decomposing

    Returns None on any estimation failure.
    """
    try:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    except ImportError:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_old = DynamicFactorMQ(
                monthly_old, endog_quarterly=quarterly, factors=factors,
                factor_orders=factor_orders, idiosyncratic_ar1=False,
            ).fit(disp=False, maxiter=maxiter)
            res_new = DynamicFactorMQ(
                monthly_new, endog_quarterly=quarterly, factors=factors,
                factor_orders=factor_orders, idiosyncratic_ar1=False,
            ).fit(disp=False, maxiter=maxiter)
            news = res_new.news(
                res_old, impact_date=impact_quarter, impacted_variable=target
            )
    except Exception:  # noqa: BLE001
        return None

    details = news.details_by_update.copy()
    details = details.reset_index()

    # Flatten to the columns a reader cares about.
    keep = {}
    for col in details.columns:
        name = col if isinstance(col, str) else col[-1] if isinstance(col, tuple) else str(col)
        keep[col] = name
    details = details.rename(columns=keep)

    out_cols = [c for c in ["update date", "updated variable", "observed",
                            "forecast (prev)", "news", "weight", "impact"]
                if c in details.columns]
    return details[out_cols].reset_index(drop=True)
