"""Benchmark forecasts, and the backtest that scores them honestly.

THE BENCHMARKS
--------------
Before any machine learning, you need to know what "good" is. These three are
the bar. If a fancy model cannot beat them, the fancy model is not earning its
complexity -- and reporting that clearly is a legitimate, publishable result.

    mean          forecast the historical average growth rate
    random_walk   forecast that next quarter equals this quarter
    ar1           first-order autoregression, refit every step
    ar_aic        AR(p), p chosen by AIC each step

For quarterly GDP growth the AR(1) is a genuinely hard benchmark. Australian
growth is weakly autocorrelated and mostly unforecastable from its own past, so
the mean is also surprisingly competitive.

THE BACKTEST
------------
One forecast per GDP release. Standing on the morning quarter t's GDP is
published, we nowcast quarter t+1 -- the quarter that has just finished but is
not yet measured. The model is refit from scratch each step on exactly the data
visible at that vintage.

There is no train/test split, no shuffling, and no cross-validation. Those all
assume exchangeable observations, which time series are not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import Snapshot, as_of, target_release_dates


@dataclass(frozen=True)
class Context:
    """Everything a model is allowed to see when making one nowcast.

    Benchmarks only need `y`. Bridge equations and factor models need the
    monthly block too. Bundling them means adding a richer model later does
    not require changing the backtest.

    Attributes
    ----------
    y        visible history of the target (quarterly GDP growth)
    snapshot everything published as at `vintage`, monthly and quarterly
    target   the quarter being nowcast -- guaranteed absent from `y`
    vintage  the date this forecast is being made
    """

    y: pd.Series
    snapshot: Snapshot
    target: pd.Period
    vintage: pd.Timestamp


Forecaster = Callable[[Context], float]


def univariate(fn: Callable[[pd.Series], float]) -> Forecaster:
    """Adapt a target-only forecaster to the Context interface.

    Lets the simple benchmarks stay simple functions of a series, which keeps
    them easy to reason about and to test directly.
    """

    def wrapped(ctx: Context) -> float:
        return fn(ctx.y)

    wrapped.__name__ = getattr(fn, "__name__", "univariate")
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ---------------------------------------------------------------------------
# Benchmark forecasters: each takes the visible history, returns one number
# ---------------------------------------------------------------------------


def f_mean(y: pd.Series) -> float:
    """Historical average. Hard to beat when a series is near white noise."""
    return float(y.mean())


def f_rolling_mean(y: pd.Series, window: int = 40) -> float:
    """Average of the last `window` quarters (default 10 years).

    Motivated by an empirical finding: the expanding-window mean is biased
    UPWARD by roughly 0.2pp, because it averages in the high-growth 1960s-70s
    and Australian trend growth has since declined. A rolling window tracks the
    slowly-moving local mean instead of anchoring on distant history.
    """
    return float(y.iloc[-window:].mean()) if len(y) > window else float(y.mean())


def f_random_walk(y: pd.Series) -> float:
    """Next quarter equals this quarter."""
    return float(y.iloc[-1])


def _fit_ar(y: pd.Series, lags: int) -> float:
    from statsmodels.tsa.ar_model import AutoReg

    values = y.to_numpy(dtype=float)
    res = AutoReg(values, lags=lags, old_names=False).fit()
    return float(res.forecast(steps=1)[0])


def f_ar1(y: pd.Series) -> float:
    """AR(1), refit on every vintage."""
    if len(y) < 10:
        return f_mean(y)
    return _fit_ar(y, lags=1)


def f_ar_aic(y: pd.Series, max_lags: int = 4) -> float:
    """AR(p) with p chosen by AIC on the data visible at this vintage.

    Choosing p inside the loop is deliberate. Picking one p on the full sample
    and reusing it everywhere would leak information from the future into every
    earlier forecast.
    """
    from statsmodels.tsa.ar_model import AutoReg

    if len(y) < 20:
        return f_mean(y)

    values = y.to_numpy(dtype=float)
    best_p, best_aic = 1, np.inf
    for p in range(1, max_lags + 1):
        try:
            aic = AutoReg(values, lags=p, old_names=False).fit().aic
        except Exception:  # noqa: BLE001 - a failed fit just loses the contest
            continue
        if aic < best_aic:
            best_p, best_aic = p, aic
    return _fit_ar(y, lags=best_p)


BENCHMARKS: dict[str, Forecaster] = {
    "mean": univariate(f_mean),
    "rolling_mean": univariate(f_rolling_mean),
    "random_walk": univariate(f_random_walk),
    "ar1": univariate(f_ar1),
    "ar_aic": univariate(f_ar_aic),
}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Forecasts and outcomes, one row per GDP release."""

    forecasts: pd.DataFrame  # index = target quarter, columns = model names
    actuals: pd.Series  # index = target quarter
    vintages: pd.Series  # index = target quarter, value = vintage date used

    def errors(self) -> pd.DataFrame:
        return self.forecasts.sub(self.actuals, axis=0)

    def score(self, start=None, end=None, label: str = "full") -> pd.DataFrame:
        """RMSE and MAE per model over an optional sub-period."""
        err = self.errors()
        if start is not None:
            err = err.loc[err.index >= pd.Period(start, freq="Q")]
        if end is not None:
            err = err.loc[err.index <= pd.Period(end, freq="Q")]

        if err.empty:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "rmse": np.sqrt((err**2).mean()),
                "mae": err.abs().mean(),
                "bias": err.mean(),
                "n": err.notna().sum(),
            }
        )
        out.insert(0, "sample", label)
        return out.sort_values("rmse")


def run_backtest(
    panel: pd.DataFrame,
    models: dict[str, Forecaster] | None = None,
    target: str = "gdp_growth",
    min_history: int = 40,
    vintage_offset_days: int = 0,
) -> BacktestResult:
    """One nowcast per GDP release, using only what was visible that morning.

    Parameters
    ----------
    panel       long point-in-time panel (already transformed)
    models      {name: forecaster}; defaults to BENCHMARKS
    min_history at least this many quarters before we start forecasting
    vintage_offset_days
                days to wait AFTER the GDP release before forecasting.

    On `vintage_offset_days`: standing exactly on the day quarter t's GDP is
    published, quarter t+1 is only about two-thirds over, and with a 26-day
    publication lag you typically hold just ONE month of it. Waiting longer
    buys more monthly data while quarter t+1's own GDP is still unpublished
    (it does not arrive until roughly 156 days after quarter t ended).

    Running the same backtest at offsets 0, 30 and 60 traces how accuracy
    improves as data arrives -- the single most informative chart in a
    nowcasting project. The leak assertion below guards the upper end: push the
    offset too far and it will fire rather than silently cheat.

    Returns
    -------
    BacktestResult
    """
    models = models or BENCHMARKS
    releases = target_release_dates(panel, target)
    offset = pd.Timedelta(days=vintage_offset_days)

    # The full (final-vintage) target series, used only to score forecasts.
    truth = panel.loc[panel["series"] == target].copy()
    truth["period"] = pd.PeriodIndex(truth["ref_period"], freq="Q")
    actual_by_period = truth.set_index("period")["value"].sort_index()

    rows, actuals, vintages = [], {}, {}

    for ref_period_str, release_date in releases.items():
        published_q = pd.Period(ref_period_str, freq="Q")
        target_q = published_q + 1
        vintage = release_date + offset

        if target_q not in actual_by_period.index:
            continue  # the quarter we would nowcast has no outcome yet

        snap = as_of(panel, vintage)
        if snap.quarterly.empty or target not in snap.quarterly.columns:
            continue

        y_visible = snap.quarterly[target].dropna()
        if len(y_visible) < min_history:
            continue

        # Safety net: the model must not be able to see the quarter it forecasts.
        if target_q in y_visible.index:
            raise AssertionError(
                f"LEAK at vintage {vintage.date()}: {target_q} already visible. "
                f"vintage_offset_days={vintage_offset_days} is too large."
            )

        row = {"target": target_q}
        ctx = Context(y=y_visible, snapshot=snap, target=target_q, vintage=vintage)
        for name, fn in models.items():
            try:
                row[name] = float(fn(ctx))
            except Exception:  # noqa: BLE001 - record the miss, keep going
                row[name] = np.nan
        rows.append(row)
        actuals[target_q] = actual_by_period.loc[target_q]
        vintages[target_q] = vintage

    if not rows:
        raise ValueError(
            "No forecasts produced. Check min_history and that the panel "
            "contains the target series."
        )

    forecasts = pd.DataFrame(rows).set_index("target").sort_index()
    return BacktestResult(
        forecasts=forecasts,
        actuals=pd.Series(actuals).sort_index(),
        vintages=pd.Series(vintages).sort_index(),
    )


def diebold_mariano(
    e1: pd.Series, e2: pd.Series, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test of equal squared-error accuracy.

    Returns (statistic, two-sided p-value). A negative statistic with a small
    p-value means the FIRST model is significantly more accurate.

    RMSE alone cannot tell you whether a gap is real or noise. With ~150
    forecasts, a 5% RMSE improvement usually is not significant. Report this
    alongside every headline comparison.
    """
    from scipy import stats

    d = (e1.dropna() ** 2) - (e2.dropna() ** 2)
    d = d.dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_bar = d.mean()
    gamma0 = d.var(ddof=0)
    var = gamma0
    for lag in range(1, h):
        cov = d.autocorr(lag) * gamma0
        var += 2 * (1 - lag / h) * cov

    if var <= 0:
        return np.nan, np.nan

    stat = d_bar / np.sqrt(var / n)
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)
