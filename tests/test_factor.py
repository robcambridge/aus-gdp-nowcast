"""Tests for the dynamic factor model wrapper.

The DFM is slow to fit, so these use small samples and capped iterations. They
check that it (a) recovers a genuine common factor, (b) handles the ragged edge,
and (c) falls back to the mean rather than crashing on any degenerate input.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ausgdp.benchmarks import Context, f_mean
from ausgdp.dataset import Snapshot
from ausgdp.factor import _steps_to_quarter, dfm_forecast, make_dfm

warnings.filterwarnings("ignore")


def _factor_context(n_q=90, months_of_target=2, n_ind=4, seed=3, signal=0.3):
    """Context where one latent factor drives the indicators and GDP."""
    rng = np.random.default_rng(seed)
    m_idx = pd.period_range("2000-01", periods=n_q * 3, freq="M")
    q_idx = pd.period_range("2000Q1", periods=n_q, freq="Q")

    factor = np.cumsum(rng.normal(0, 0.4, len(m_idx)))
    monthly = pd.DataFrame(
        {
            f"ind{i}": factor * rng.uniform(0.5, 1.5) + rng.normal(0, 0.6, len(m_idx))
            for i in range(n_ind)
        },
        index=m_idx,
    )
    gdp = pd.Series(
        0.5 + signal * factor[2::3][:n_q] + rng.normal(0, 0.3, n_q), index=q_idx
    )

    target = q_idx[-1]
    y_vis = gdp.iloc[:-1]
    keep = (n_q - 1) * 3 + months_of_target
    snap = Snapshot(
        vintage=pd.Timestamp("2022-06-01"),
        monthly=pd.DataFrame(monthly.iloc[:keep]),
        quarterly=pd.DataFrame({"gdp_growth": y_vis}),
    )
    ctx = Context(y=y_vis, snapshot=snap, target=target, vintage=snap.vintage)
    return ctx, gdp, [f"ind{i}" for i in range(n_ind)]


def test_steps_to_quarter():
    may = pd.Period("2024-05", freq="M")
    aug = pd.Period("2024-08", freq="M")
    q3 = pd.Period("2024Q3", freq="Q")
    assert _steps_to_quarter(may, q3) == 4
    assert _steps_to_quarter(aug, q3) == 1


def test_dfm_beats_mean_with_common_factor():
    ctx, gdp, names = _factor_context(signal=0.4)
    actual = gdp.loc[ctx.target]
    dfm_err = abs(dfm_forecast(ctx, names, factor_orders=2, maxiter=80) - actual)
    mean_err = abs(f_mean(ctx.y) - actual)
    assert dfm_err < mean_err


def test_dfm_handles_ragged_edge():
    """One month of the target quarter should still produce a finite forecast."""
    ctx, _, names = _factor_context(months_of_target=1)
    assert np.isfinite(dfm_forecast(ctx, names, maxiter=60))


def test_dfm_falls_back_on_empty_monthly():
    ctx, _, _ = _factor_context()
    empty_ctx = Context(
        y=ctx.y,
        snapshot=Snapshot(ctx.vintage, pd.DataFrame(), ctx.snapshot.quarterly),
        target=ctx.target,
        vintage=ctx.vintage,
    )
    assert dfm_forecast(empty_ctx, ["ind0"]) == pytest.approx(f_mean(ctx.y))


def test_dfm_falls_back_when_no_indicators_match():
    ctx, _, _ = _factor_context()
    assert dfm_forecast(ctx, ["nonexistent"]) == pytest.approx(f_mean(ctx.y))


def test_dfm_falls_back_on_short_monthly():
    q_idx = pd.period_range("2010Q1", periods=40, freq="Q")
    y = pd.Series(np.random.default_rng(0).normal(0.6, 0.5, 40), index=q_idx)
    short = pd.DataFrame(
        {"ind0": range(12)}, index=pd.period_range("2010-01", periods=12, freq="M")
    )
    ctx = Context(
        y=y,
        snapshot=Snapshot(pd.Timestamp("2020-01-01"), short, pd.DataFrame({"gdp_growth": y})),
        target=q_idx[-1] + 1,
        vintage=pd.Timestamp("2020-01-01"),
    )
    assert dfm_forecast(ctx, ["ind0"]) == pytest.approx(f_mean(y))


def test_make_dfm_names_itself():
    assert make_dfm(["a", "b"], factors=1).__name__ == "dfm_1f"


@pytest.mark.slow
def test_dfm_two_factors_runs():
    ctx, _, names = _factor_context(n_ind=6)
    assert np.isfinite(dfm_forecast(ctx, names, factors=2, maxiter=60))


# --- news decomposition ----------------------------------------------------


def test_news_decomposition_attributes_impact():
    """Sum of per-release impacts equals the total nowcast revision."""
    from ausgdp.factor import news_decomposition

    rng = np.random.default_rng(1)
    m_idx = pd.period_range("2000-01", periods=180, freq="M")
    q_idx = pd.period_range("2000Q1", periods=58, freq="Q")
    f = np.cumsum(rng.normal(0, 0.4, 180))
    monthly = pd.DataFrame(
        {f"ind{i}": f * rng.uniform(0.5, 1.5) + rng.normal(0, 0.6, 180) for i in range(3)},
        index=m_idx,
    )
    q = pd.DataFrame({"gdp_growth": f[2::3][:58] * 0.3 + rng.normal(0, 0.3, 58)}, index=q_idx)

    details = news_decomposition(
        monthly.iloc[:176], monthly.iloc[:178], q, pd.Period("2014Q4", freq="Q")
    )
    assert details is not None
    assert {"news", "weight", "impact"} <= set(details.columns)
    # impact should equal news * weight, row by row
    np.testing.assert_allclose(
        details["impact"].to_numpy(),
        (details["news"] * details["weight"]).to_numpy(),
        atol=1e-6,
    )


def test_news_decomposition_falls_back_to_none():
    from ausgdp.factor import news_decomposition

    q = pd.DataFrame(
        {"gdp_growth": [0.5, 0.6]}, index=pd.period_range("2014Q1", periods=2, freq="Q")
    )
    assert news_decomposition(
        pd.DataFrame(), pd.DataFrame(), q, pd.Period("2014Q2", freq="Q")
    ) is None
