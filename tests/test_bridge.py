"""Tests for bridge equations and the partial-quarter aggregation they rely on.

The key property under test is CONSISTENCY: when only n months of the target
quarter are available, the estimation sample must be rebuilt using n-month
averages too. Estimating on 3-month averages and predicting from a 1-month
average is the classic bridge-equation bug.
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

# --- partial quarterly aggregation -----------------------------------------


def _monthly(n=9, start="2024-01"):
    return pd.Series(
        range(n), index=pd.period_range(start, periods=n, freq="M"), dtype=float
    )


def test_partial_mean_first_month_only():
    out = partial_quarterly_mean(_monthly(), 1)
    assert out.to_list() == [0.0, 3.0, 6.0]  # Jan, Apr, Jul


def test_partial_mean_two_months():
    out = partial_quarterly_mean(_monthly(), 2)
    assert out.to_list() == [0.5, 3.5, 6.5]


def test_partial_mean_full_quarter():
    out = partial_quarterly_mean(_monthly(), 3)
    assert out.to_list() == [1.0, 4.0, 7.0]


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


# --- bridge forecasts ------------------------------------------------------


def _context(n_quarters=80, indicator_beta=1.0, months_of_target=2, seed=1):
    """Build a Context where the indicator genuinely predicts GDP."""
    rng = np.random.default_rng(seed)

    q_idx = pd.period_range("2000Q1", periods=n_quarters, freq="Q")
    m_idx = pd.period_range("2000-01", periods=n_quarters * 3, freq="M")

    x_monthly = pd.Series(rng.normal(0, 1, len(m_idx)), index=m_idx)
    x_q = partial_quarterly_mean(x_monthly, 3)
    y = pd.Series(
        0.5 + indicator_beta * x_q.to_numpy() + rng.normal(0, 0.3, n_quarters),
        index=q_idx,
    )

    target = q_idx[-1]
    y_visible = y.iloc[:-1]  # target quarter's GDP not yet published

    # Truncate the monthly indicator to `months_of_target` months of the target
    keep_to = (n_quarters - 1) * 3 + months_of_target
    monthly = pd.DataFrame({"ind": x_monthly.iloc[:keep_to]})

    snap = Snapshot(
        vintage=pd.Timestamp("2020-01-01"),
        monthly=monthly,
        quarterly=pd.DataFrame({"gdp_growth": y_visible}),
    )
    return Context(y=y_visible, snapshot=snap, target=target, vintage=snap.vintage), y


def test_bridge_beats_the_mean_when_indicator_is_informative():
    ctx, truth = _context(indicator_beta=1.5)
    actual = truth.loc[ctx.target]

    bridge_err = abs(bridge_forecast(ctx, "ind") - actual)
    mean_err = abs(f_mean(ctx.y) - actual)
    assert bridge_err < mean_err


def test_bridge_recovers_the_coefficient():
    """With a strong signal the bridge forecast should land near the truth."""
    ctx, truth = _context(indicator_beta=2.0, months_of_target=3, seed=5)
    pred = bridge_forecast(ctx, "ind", add_ar=False)
    assert abs(pred - truth.loc[ctx.target]) < 1.0


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
    # Different information sets should generally give different forecasts.
    assert bridge_forecast(ctx1, "ind") != pytest.approx(bridge_forecast(ctx3, "ind"))


def test_make_bridge_names_itself():
    fn = make_bridge("employment")
    assert fn.__name__ == "bridge_employment"


def test_bridge_average_sits_between_its_members():
    ctx, _ = _context()
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * -0.5

    a = bridge_forecast(ctx, "ind")
    b = bridge_forecast(ctx, "ind2")
    avg = make_bridge_average(["ind", "ind2"])(ctx)
    assert min(a, b) <= avg <= max(a, b)


def test_bridge_average_falls_back_cleanly():
    ctx, _ = _context()
    avg = make_bridge_average(["nope1", "nope2"])(ctx)
    assert avg == pytest.approx(f_mean(ctx.y))


# --- ridge -----------------------------------------------------------------


def test_ridge_runs_and_is_finite():
    ctx, _ = _context()
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * 0.5
    pred = make_ridge(["ind", "ind2"])(ctx)
    assert np.isfinite(pred)


def test_ridge_falls_back_without_indicators():
    ctx, _ = _context()
    assert make_ridge(["nope"])(ctx) == pytest.approx(f_mean(ctx.y))


def test_ridge_beats_mean_with_informative_indicators():
    ctx, truth = _context(indicator_beta=1.5, months_of_target=3, seed=4)
    ctx.snapshot.monthly["ind2"] = ctx.snapshot.monthly["ind"] * 0.8
    actual = truth.loc[ctx.target]
    assert abs(make_ridge(["ind", "ind2"])(ctx) - actual) < abs(f_mean(ctx.y) - actual)
