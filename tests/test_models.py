"""Tests for transforms, quarterly aggregation, and the backtest.

The important one is test_backtest_detects_injected_leak: it deliberately
corrupts the panel so the target quarter becomes visible, and checks that the
backtest refuses to run. A safety net you have never seen fire is not a safety
net.
"""

import numpy as np
import pandas as pd
import pytest

from ausgdp.benchmarks import (
    diebold_mariano,
    f_ar1,
    f_mean,
    f_random_walk,
    run_backtest,
)
from ausgdp.config import SeriesSpec
from ausgdp.dataset import build_panel
from ausgdp.transforms import (
    monthly_to_quarterly,
    transform_all,
    transform_series,
)

# --- transforms ------------------------------------------------------------


def test_pct_change():
    s = pd.Series([100.0, 110.0, 99.0], index=pd.period_range("2024-01", periods=3, freq="M"))
    out = transform_series(s, "pct")
    assert out.iloc[0] == pytest.approx(10.0)
    assert out.iloc[1] == pytest.approx(-10.0)


def test_diff_for_rates():
    """A rate moving 4.0 -> 4.4 is +0.4 points, not +10 percent."""
    s = pd.Series([4.0, 4.4], index=pd.period_range("2024-01", periods=2, freq="M"))
    assert transform_series(s, "diff").iloc[0] == pytest.approx(0.4)


def test_level_passthrough():
    s = pd.Series([1.0, 2.0], index=pd.period_range("2024-01", periods=2, freq="M"))
    pd.testing.assert_series_equal(transform_series(s, "level"), s)


def test_log_pct_rejects_nonpositive():
    s = pd.Series([1.0, 0.0], index=pd.period_range("2024-01", periods=2, freq="M"))
    with pytest.raises(ValueError, match="positive"):
        transform_series(s, "log_pct")


def test_unknown_transform_raises():
    s = pd.Series([1.0], index=pd.period_range("2024-01", periods=1, freq="M"))
    with pytest.raises(ValueError, match="Unknown transform"):
        transform_series(s, "wiggle")


def test_transform_all_requires_specs():
    s = pd.Series([1.0, 2.0], index=pd.period_range("2024-01", periods=2, freq="M"))
    with pytest.raises(KeyError):
        transform_all({"x": s}, {})


# --- monthly -> quarterly --------------------------------------------------


def test_quarterly_average_and_counts():
    idx = pd.period_range("2024-01", periods=5, freq="M")  # Q1 full, Q2 partial
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0, 20.0]}, index=idx)
    values, counts = monthly_to_quarterly(df)

    assert values.loc[pd.Period("2024Q1", freq="Q"), "a"] == pytest.approx(2.0)
    assert counts.loc[pd.Period("2024Q1", freq="Q"), "a"] == 3
    # partial quarter is kept, averaged over what exists
    assert values.loc[pd.Period("2024Q2", freq="Q"), "a"] == pytest.approx(15.0)
    assert counts.loc[pd.Period("2024Q2", freq="Q"), "a"] == 2


def test_quarterly_min_months_filter():
    idx = pd.period_range("2024-01", periods=5, freq="M")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0, 20.0]}, index=idx)
    values, _ = monthly_to_quarterly(df, min_months=3)
    assert not np.isnan(values.loc[pd.Period("2024Q1", freq="Q"), "a"])
    assert np.isnan(values.loc[pd.Period("2024Q2", freq="Q"), "a"])


def test_quarterly_empty_input():
    values, counts = monthly_to_quarterly(pd.DataFrame())
    assert values.empty and counts.empty


# --- forecasters -----------------------------------------------------------


def test_random_walk_returns_last_value():
    y = pd.Series([1.0, 2.0, 3.5])
    assert f_random_walk(y) == pytest.approx(3.5)


def test_mean_returns_average():
    y = pd.Series([1.0, 2.0, 3.0])
    assert f_mean(y) == pytest.approx(2.0)


def test_ar1_falls_back_on_short_history():
    y = pd.Series([1.0, 2.0, 3.0])
    assert f_ar1(y) == pytest.approx(f_mean(y))


def test_ar1_recovers_persistence():
    """On a strongly persistent series AR(1) should forecast near the last value."""
    rng = np.random.default_rng(0)
    n = 300
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.9 * v[t - 1] + rng.normal(0, 0.5)
    y = pd.Series(v)
    assert abs(f_ar1(y) - 0.9 * y.iloc[-1]) < 0.4


# --- backtest --------------------------------------------------------------


def _toy_panel(n_quarters=80, gdp_lag=65):
    rng = np.random.default_rng(3)
    idx = pd.period_range("2000Q1", periods=n_quarters, freq="Q")
    y = pd.Series(rng.normal(0.6, 0.8, n_quarters).round(3), index=idx)
    spec = SeriesSpec(
        name="gdp_growth", source="abs", collection="5206.0",
        freq="Q", transform="pct", lag_days=gdp_lag,
    )
    return build_panel({"gdp_growth": y}, {"gdp_growth": spec})


def test_backtest_runs_and_scores():
    res = run_backtest(_toy_panel(), min_history=20)
    assert len(res.forecasts) > 20
    assert set(res.forecasts.columns) == {
        "mean", "rolling_mean", "random_walk", "ar1", "ar_aic",
    }

    scores = res.score(label="full")
    assert (scores["rmse"] > 0).all()
    assert scores["n"].iloc[0] == len(res.forecasts)


def test_backtest_never_forecasts_a_visible_quarter():
    """Every target must post-date the last quarter visible at its vintage."""
    panel = _toy_panel()
    res = run_backtest(panel, min_history=20)

    for target_q, vintage in res.vintages.items():
        visible = panel.loc[
            (panel["available_from"] <= vintage) & (panel["series"] == "gdp_growth")
        ]
        last_visible = pd.Period(visible["ref_period"].iloc[-1], freq="Q")
        assert target_q > last_visible


def test_backtest_detects_injected_leak():
    """Corrupt the panel so the target is visible early; backtest must refuse."""
    panel = _toy_panel()
    # Make every observation available immediately at the start of the sample:
    # now the "future" quarter is visible at every vintage.
    panel.loc[:, "available_from"] = pd.Timestamp("2000-01-01")

    with pytest.raises(AssertionError, match="LEAK"):
        run_backtest(panel, min_history=20)


def test_backtest_scoring_subperiods():
    res = run_backtest(_toy_panel(n_quarters=100), min_history=20)
    early = res.score(end="2010Q4", label="pre2011")
    late = res.score(start="2011Q1", label="post2011")
    assert early["n"].iloc[0] + late["n"].iloc[0] == len(res.forecasts)


def test_backtest_raises_without_target():
    panel = _toy_panel()
    panel = panel.loc[panel["series"] != "gdp_growth"]
    with pytest.raises(ValueError):
        run_backtest(panel, min_history=20)


# --- Diebold-Mariano -------------------------------------------------------


def test_dm_detects_a_clearly_better_model():
    rng = np.random.default_rng(11)
    good = pd.Series(rng.normal(0, 0.5, 200))
    bad = pd.Series(rng.normal(0, 2.0, 200))
    stat, pval = diebold_mariano(good, bad)
    assert stat < 0  # first model has smaller squared errors
    assert pval < 0.01


def test_dm_finds_no_difference_between_equals():
    rng = np.random.default_rng(12)
    a = pd.Series(rng.normal(0, 1.0, 300))
    b = pd.Series(rng.normal(0, 1.0, 300))
    _, pval = diebold_mariano(a, b)
    assert pval > 0.05


def test_dm_short_sample_returns_nan():
    a = pd.Series([0.1, 0.2])
    stat, pval = diebold_mariano(a, a)
    assert np.isnan(stat) and np.isnan(pval)
