"""Tests for the point-in-time logic.

These are not decorative. The leakage test below is the one thing standing
between you and a backtest that quietly reports a fake 60% improvement.
"""

import pandas as pd
import pytest

from ausgdp.config import SeriesSpec
from ausgdp.dataset import (
    as_of,
    availability_date,
    build_panel,
    ragged_edge_report,
    to_long,
)


def monthly(name: str, lag_days: int, start="2024-01", n=12) -> tuple[pd.Series, SeriesSpec]:
    idx = pd.period_range(start=start, periods=n, freq="M")
    s = pd.Series(range(n), index=idx, dtype=float)
    spec = SeriesSpec(name=name, source="abs", collection="X", freq="M", lag_days=lag_days)
    return s, spec


def quarterly(name: str, lag_days: int, start="2024Q1", n=4) -> tuple[pd.Series, SeriesSpec]:
    idx = pd.period_range(start=start, periods=n, freq="Q")
    s = pd.Series(range(n), index=idx, dtype=float)
    spec = SeriesSpec(name=name, source="abs", collection="X", freq="Q", lag_days=lag_days)
    return s, spec


def test_availability_is_end_of_period_plus_lag():
    p = pd.Period("2024-07", freq="M")
    assert availability_date(p, 17) == pd.Timestamp("2024-08-17")


def test_quarterly_availability():
    p = pd.Period("2024Q2", freq="Q")  # ends 30 June
    assert availability_date(p, 65) == pd.Timestamp("2024-09-03")


def test_to_long_requires_period_index():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    spec = SeriesSpec(name="bad", source="abs", collection="X")
    with pytest.raises(TypeError, match="PeriodIndex"):
        to_long(s, spec)


def test_no_leakage_every_visible_row_was_published():
    """THE critical invariant: nothing in a snapshot postdates the vintage."""
    fast, fast_spec = monthly("fast", lag_days=17)
    slow, slow_spec = monthly("slow", lag_days=45)
    gdp, gdp_spec = quarterly("gdp_growth", lag_days=65)

    panel = build_panel(
        {"fast": fast, "slow": slow, "gdp_growth": gdp},
        {"fast": fast_spec, "slow": slow_spec, "gdp_growth": gdp_spec},
    )

    for vintage in ["2024-03-15", "2024-06-01", "2024-09-03", "2024-12-31"]:
        cutoff = pd.Timestamp(vintage)
        visible = panel.loc[panel["available_from"] <= cutoff]
        assert (visible["available_from"] <= cutoff).all()
        # the snapshot must contain exactly the visible rows -- no more
        snap = as_of(panel, vintage)
        assert snap.n_values == len(visible)


def test_ragged_edge_is_actually_ragged():
    """A fast series must reach further than a slow one at the same vintage."""
    fast, fast_spec = monthly("fast", lag_days=17)
    slow, slow_spec = monthly("slow", lag_days=45)

    panel = build_panel(
        {"fast": fast, "slow": slow}, {"fast": fast_spec, "slow": slow_spec}
    )
    snap = as_of(panel, "2024-07-20").monthly

    last_fast = snap["fast"].last_valid_index()
    last_slow = snap["slow"].last_valid_index()
    assert last_fast > last_slow, "faster-publishing series should extend further"


def test_gdp_unknown_for_current_quarter():
    """At the June-quarter release, September-quarter GDP must be absent."""
    gdp, gdp_spec = quarterly("gdp_growth", lag_days=65, start="2024Q1", n=4)
    panel = build_panel({"gdp_growth": gdp}, {"gdp_growth": gdp_spec})

    snap = as_of(panel, "2024-09-03").quarterly  # day June-qtr GDP is released
    assert pd.Period("2024Q2", freq="Q") in snap.index
    assert pd.Period("2024Q3", freq="Q") not in snap.index


def test_empty_snapshot_before_anything_published():
    fast, fast_spec = monthly("fast", lag_days=17)
    panel = build_panel({"fast": fast}, {"fast": fast_spec})
    assert as_of(panel, "2023-01-01").is_empty


def test_ragged_edge_report_shape():
    fast, fast_spec = monthly("fast", lag_days=17)
    slow, slow_spec = monthly("slow", lag_days=45)
    panel = build_panel(
        {"fast": fast, "slow": slow}, {"fast": fast_spec, "slow": slow_spec}
    )
    rep = ragged_edge_report(panel, "2024-07-20")
    assert set(rep["series"]) == {"fast", "slow"}
    assert (rep["days_stale"] >= 0).all()


def test_build_panel_rejects_unregistered_series():
    fast, fast_spec = monthly("fast", lag_days=17)
    with pytest.raises(KeyError):
        build_panel({"fast": fast}, {"other": fast_spec})
