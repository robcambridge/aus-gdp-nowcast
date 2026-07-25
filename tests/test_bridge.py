"""Tests for bridge equations and the levels-then-growth regressor they use.

Two properties matter here.

CONSISTENCY: when only n months of the target quarter are available, the whole
history must be rebuilt using n-month averages. Estimating on three-month
averages and predicting from a one-month average is the classic bridge bug.

CONSTRUCTION: monthly indicators enter the panel as LEVELS. The regressor is
the change in the quarterly average level, not the average of monthly growth
rates. These are different quantities and only the first matches how GDP is
measured.
"""

import numpy as np
import pandas as pd
import pytest

from ausgdp.benchmarks import Context, f_mean
from ausgdp.bridge import (
    bridge_forecast,
    make_bridge,
    make_bridge_average,
    make_ridge,
    months_available,
    partial_quarterly_mean,
)
from ausgdp.dataset import Snapshot
from ausgdp.transforms import quarterly_regressor

# --- partial quarterly aggregation -----------------------------------------


def _monthly(n=9, start="2024-01"):
    return pd.Series(
        range(n), index=pd.period_range(start, periods=n, freq="M"), dtype=float
    )


def test_partial_mean_first_month_only():
    assert partial_quarterly_mean(_monthly(), 1).to_list() == [0.0, 3.0, 6.0]


def test_partial_mean_two_months():
    assert partial_quarterly_mean(_monthly(), 2).to_list() == [0.5, 3.5, 6.5]


def test_partial_mean_full_quarter():
    assert partial_quarterly_mean(_monthly(), 3).to_list() == [1.0, 4.0, 7.0]


def test_partial_mean_rejects_bad_n():
    with pytest.raises(ValueError, match="n_months"):
        partial_quarterly_mean(_monthly(), 4)


def test_partial_mean_empty_input():
    assert partial_quarterly_mean(pd.Series(dtype=float), 3).empty


def test_months_available_counts_ragged_edge():
    s = _monthly(7)  # Jan..Jul: Q3 holds only July
    assert months_available(s, pd.Period("2024Q1", freq="Q")) == 3
    assert months_available(s, pd.Period("2024Q3", freq="Q")) == 1
    assert months_available(s, pd.Period("2024Q4", freq="Q")) == 0


# --- quarterly regressor ---------------------------------------------------


def _levels(n=36, start="2020-01", base=1000.0, drift=0.004):
    idx = pd.period_range(start, periods=n, freq="M")
    return pd.Series(base * np.exp(np.arange(n) * drift), index=idx)


def test_regressor_full_quarter_equals_qoq_growth_of_averages():
    lvl = _levels()
    auto = quarterly_regressor(lvl, "mean_then_pct", 3)

    full_q = lvl.groupby(lvl.index.asfreq("Q")).mean()
    manual = ((full_q / full_q.shift(1) - 1) * 100).dropna()
    manual.index = pd.PeriodIndex(manual.index, freq="Q")

    pd.testing.assert_series_equal(auto, manual, check_names=False)


def test_regressor_diff_for_rates():
    """A rate: quarterly average differenced against the previous quarter."""
    idx = pd.period_range("2024-01", periods=6, freq="M")
    rate = pd.Series([4.0, 4.0, 4.0, 4.5, 4.5, 4.5], index=idx)
    out = quarterly_regressor(rate, "mean_then_diff", 3)
    assert out.loc[pd.Period("2024Q2", freq="Q")] == pytest.approx(0.5)


def test_regressor_mean_passthrough():
    """'mean' aggregation just averages; it is already a rate of change."""
    idx = pd.period_range("2024-01", periods=3, freq="M")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    out = quarterly_regressor(s, "mean", 3)
    assert out.loc[pd.Period("2024Q1", freq="Q")] == pytest.approx(2.0)


def test_regressor_partial_uses_full_previous_quarter_as_base():
    idx = pd.period_range("2024-01", periods=6, freq="M")
    lvl = pd.Series([100.0, 100.0, 100.0, 110.0, 999.0, 999.0], index=idx)
    # Q2 with n=1 -> 110 against Q1's full average of 100 -> +10%
    out = quarterly_regressor(lvl, "mean_then_pct", 1)
    assert out.loc[pd.Period("2024Q2", freq="Q")] == pytest.approx(10.0)


def test_regressor_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        quarterly_regressor(_levels(), "sideways", 3)


def test_regressor_recovers_truth_better_than_averaging_growth_rates():
    """The reason for the whole redesign, pinned as a test.

    Under monthly survey noise, taking the change in quarterly average LEVELS
    tracks true quarterly growth much better than averaging monthly growth
    rates -- especially when only one month has arrived.
    """
    rng = np.random.default_rng(42)
    n = 480
    idx = pd.period_range("1986-01", periods=n, freq="M")
    true_g = 0.15 + 0.35 * np.sin(np.arange(n) / 17)
    true_lvl = 1000 * np.exp(np.cumsum(true_g / 100))
    observed = pd.Series(true_lvl * (1 + rng.normal(0, 0.003, n)), index=idx)

    tq = pd.Series(true_lvl, index=idx).groupby(idx.asfreq("Q")).mean()
    truth = ((tq / tq.shift(1) - 1) * 100).dropna()
    truth.index = pd.PeriodIndex(truth.index, freq="Q")

    growth = observed.pct_change() * 100
    keep = ((growth.index.month - 1) % 3) < 1
    old = growth[keep].groupby(growth[keep].index.asfreq("Q")).mean() * 3
    old.index = pd.PeriodIndex(old.index, freq="Q")

    new = quarterly_regressor(observed, "mean_then_pct", 1)

    def rmse(est):
        j = pd.concat([truth.rename("t"), est.rename("e")], axis=1).dropna()
        return np.sqrt(((j["t"] - j["e"]) ** 2).mean())

    assert rmse(new) < 0.6 * rmse(old)


# --- bridge forecasts ------------------------------------------------------


def _context(n_quarters=80, indicator_beta=1.0, months_of_target=2, seed=1):
    """Context where a monthly LEVEL series genuinely predicts GDP growth."""
    rng = np.random.default_rng(seed)

    q_idx = pd.period_range("2000Q1", periods=n_quarters, freq="Q")
    m_idx = pd.period_range("2000-01", periods=n_quarters * 3, freq="M")

    steps = rng.normal(0.004, 0.006, len(m_idx))
    levels = pd.Series(1000 * np.exp(np.cumsum(steps)), index=m_idx)

    x_q = quarterly_regressor(levels, "mean_then_pct", 3)
    y = pd.Series(
        0.5 + indicator_beta * x_q.reindex(q_idx).fillna(0.0).to_numpy()
        + rng.normal(0, 0.3, n_quarters),
        index=q_idx,
    )

    target = q_idx[-1]
    y_visible = y.iloc[:-1]  # target quarter's GDP not yet published

    keep_to = (n_quarters - 1) * 3 + months_of_target
    monthly = pd.DataFrame({"ind": levels.iloc[:keep_to]})

    snap = Snapshot(
        vintage=pd.Timestamp("2020-01-01"),
        monthly=monthly,
        quarterly=pd.DataFrame({"gdp_growth": y_visible}),
    )
    return Context(y=y_visible, snapshot=snap, target=target, vintage=snap.vintage), y


def test_bridge_beats_the_mean_when_indicator_is_informative():
    ctx, truth = _context(indicator_beta=3.0, months_of_target=3, seed=2)
    actual = truth.loc[ctx.target]
    assert abs(bridge_forecast(ctx, "ind") - actual) < abs(f_mean(ctx.y) - actual)


def test_bridge_works_with_only_one_month():
    ctx, _ = _context(months_of_target=1)
    assert np.isfinite(bridge_forecast(ctx, "ind"))


def test_bridge_falls_back_when_indicator_missing():
    ctx, _ = _context()
    assert bridge_forecast(ctx, "does_not_exist") == pytest.approx(f_mean(ctx.y))


def test_bridge_falls_back_when_target_has_no_months():
    ctx, _ = _context(months_of_target=0)
    assert bridge_forecast(ctx, "ind") == pytest.approx(f_mean(ctx.y))


def test_bridge_falls_back_on_short_sample():
    ctx, _ = _context(n_quarters=12)
    assert bridge_forecast(ctx, "ind") == pytest.approx(f_mean(ctx.y))


def test_bridge_uses_consistent_aggregation():
    """A 1-month vintage must estimate on 1-month averages, not 3-month ones."""
    ctx1, _ = _context(months_of_target=1, seed=9)
    ctx3, _ = _context(months_of_target=3, seed=9)
    assert bridge_forecast(ctx1, "ind") != pytest.approx(bridge_forecast(ctx3, "ind"))


def test_make_bridge_names_itself():
    assert make_bridge("employment").__name__ == "bridge_employment"


def test_bridge_average_sits_between_its_members():
    ctx, _ = _context(seed=3)
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"].iloc[::-1].to_numpy()

    a = bridge_forecast(ctx, "ind")
    b = bridge_forecast(ctx, "ind2")
    avg = make_bridge_average(["ind", "ind2"])(ctx)
    assert min(a, b) - 1e-9 <= avg <= max(a, b) + 1e-9


def test_bridge_average_falls_back_cleanly():
    ctx, _ = _context()
    assert make_bridge_average(["nope1", "nope2"])(ctx) == pytest.approx(f_mean(ctx.y))


# --- ridge -----------------------------------------------------------------


def test_ridge_runs_and_is_finite():
    ctx, _ = _context()
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * 1.5
    assert np.isfinite(make_ridge(["ind", "ind2"])(ctx))


def test_ridge_falls_back_without_indicators():
    ctx, _ = _context()
    assert make_ridge(["nope"])(ctx) == pytest.approx(f_mean(ctx.y))


def test_ridge_beats_mean_with_informative_indicators():
    ctx, truth = _context(indicator_beta=3.0, months_of_target=3, seed=4)
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * 1.2
    actual = truth.loc[ctx.target]
    assert abs(make_ridge(["ind", "ind2"])(ctx) - actual) < abs(f_mean(ctx.y) - actual)


# --- rolling-window estimation ---------------------------------------------


def test_window_changes_the_forecast():
    """Estimating on the last 40 quarters must differ from using all history."""
    ctx, _ = _context(n_quarters=120, months_of_target=3, seed=7)
    expanding = bridge_forecast(ctx, "ind", window=None)
    rolling = bridge_forecast(ctx, "ind", window=40)
    assert expanding != pytest.approx(rolling)


def test_window_tracks_a_drifting_intercept():
    """With drifting mean growth, a rolling window beats an expanding one.

    This is the whole justification for the rolling estimation: an intercept
    estimated over decades anchors on a stale average.
    """
    rng = np.random.default_rng(21)
    n_q = 160
    q_idx = pd.period_range("1985Q1", periods=n_q, freq="Q")
    m_idx = pd.period_range("1985-01", periods=n_q * 3, freq="M")

    levels = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0.004, 0.006, len(m_idx)))),
                       index=m_idx)
    x_q = quarterly_regressor(levels, "mean_then_pct", 3).reindex(q_idx).fillna(0.0)

    drift = np.linspace(2.0, 0.2, n_q)  # trend growth falls steadily
    y = pd.Series(drift + 0.5 * x_q.to_numpy() + rng.normal(0, 0.3, n_q), index=q_idx)

    target = q_idx[-1]
    snap = Snapshot(
        vintage=pd.Timestamp("2024-01-01"),
        monthly=pd.DataFrame({"ind": levels}),
        quarterly=pd.DataFrame({"gdp_growth": y.iloc[:-1]}),
    )
    ctx = Context(y=y.iloc[:-1], snapshot=snap, target=target, vintage=snap.vintage)

    actual = y.loc[target]
    err_expanding = abs(bridge_forecast(ctx, "ind", window=None) - actual)
    err_rolling = abs(bridge_forecast(ctx, "ind", window=40) - actual)
    assert err_rolling < err_expanding


def test_window_too_small_falls_back():
    ctx, _ = _context(months_of_target=3)
    assert bridge_forecast(ctx, "ind", window=10) == pytest.approx(f_mean(ctx.y))


def test_bridge_without_ar_term_runs():
    ctx, _ = _context(months_of_target=3)
    assert np.isfinite(bridge_forecast(ctx, "ind", add_ar=False))


def test_ridge_accepts_window_and_no_ar():
    ctx, _ = _context(n_quarters=120, months_of_target=3, seed=8)
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * 1.3
    pred = make_ridge(["ind", "ind2"], window=40, add_ar=False)(ctx)
    assert np.isfinite(pred)
