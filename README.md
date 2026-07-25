# Australian GDP Nowcasting

Point-in-time nowcasting of Australian quarterly real GDP growth using monthly
ABS and RBA indicators.

**Status:** Phase 2. Point-in-time data layer complete; benchmark models running.

---

## The problem, stated precisely

The ABS publishes quarterly GDP roughly nine weeks after the quarter ends — the
June quarter lands in early September. So at the moment GDP for quarter *t* is
released, quarter *t+1* is already over but unmeasured.

This project forecasts **quarter *t+1* growth, using only information published
on or before a given date**. That is a nowcast of a completed-but-unpublished
quarter, which is what the RBA and Treasury actually do, and it is the version
of the problem where monthly indicators can add value over a univariate
benchmark.

## Why the plumbing comes first

The informational edge comes from monthly data arriving before GDP does. That
edge is only real if the backtest respects **when each number was actually
published**. Every observation in this project carries an `available_from`
date, and snapshots are constructed by filtering on it — so look-ahead bias is
structurally prevented rather than manually avoided.

```
series      ref_period  ref_end     value  available_from
employment  2024-07     2024-07-31   14.2  2024-08-17
employment  2024-08     2024-08-31   14.3  2024-09-17
gdp_growth  2024Q2      2024-06-30    0.3  2024-09-03
```

`as_of(panel, "2024-09-03")` returns only rows published by that date. Because
series publish at different speeds, the bottom edge is a staircase — the
**ragged edge**.

## Setup

```bash
uv sync                                  # install dependencies
uv run python scripts/02_demo_ragged_edge.py   # synthetic demo, no network
uv run pytest                            # 9 tests, including a leakage check
uv run python scripts/01_discover.py     # find real ABS series IDs
uv run python scripts/05_check_series.py # download and verify
uv run python scripts/06_build_dataset.py
uv run python scripts/07_benchmark.py    # the number to beat
uv run python scripts/08_horse_race.py   # do the indicators help?
```

## Layout

```
src/ausgdp/config.py       Series registry + verified publication lags
src/ausgdp/fetch.py        ABS/RBA download via readabs
src/ausgdp/dataset.py      Long panel, Snapshot, as_of(), ragged-edge diagnostics
src/ausgdp/transforms.py   Panel prep, ADF tests, levels-to-quarterly regressors
src/ausgdp/benchmarks.py   Benchmarks, Context, backtest, Diebold-Mariano
src/ausgdp/bridge.py       Bridge equations and ridge on monthly indicators

scripts/01_discover.py     Confirm ABS series IDs            (network)
scripts/02_demo_*.py       Offline demo of point-in-time logic
scripts/03_search_meta.py  Search downloaded ABS metadata    (offline)
scripts/04_list_catalogues.py  Find ABS collections          (network)
scripts/05_check_series.py Verify pinned series, save raw    (network)
scripts/06_build_dataset.py  Transform + build panel         (offline)
scripts/07_benchmark.py    Benchmarks only                   (offline)
scripts/08_horse_race.py   Full comparison at 3 vintages     (offline)

tests/                     59 tests: leakage, ragged edge, transforms, backtest
```

## Limitations (read before believing any result)

1. **Publication lags are conservative upper bounds, held constant.**
   Verified against the [ABS release calendar](https://www.abs.gov.au/release-calendar)
   on 2026-07-24; observed release dates are recorded in `config.py`. Each lag is
   set to the longest recently observed delay plus a buffer, so the model
   occasionally discards data it could legitimately have used. Measured accuracy
   is therefore a **lower bound** on real-time performance. Lags are also held
   constant across the sample, though ABS releases were slower historically.

2. **Final-vintage data, not real-time vintages.** Values used are the latest
   revised estimates, not the first prints a forecaster would have seen.
   Revisions to Australian GDP can be several tenths of a percentage point.
   This likely flatters measured performance. `readabs` supports
   `read_abs_cat(cat, history="dec-2023")` for true vintage data — a planned
   extension.

3. **COVID.** June-quarter 2020 is roughly a ten-sigma observation. Results
   will be reported for the full sample, pre-2020, and post-2021 separately.

4. **Small sample.** ~265 quarterly observations since 1959. Flexible models
   are expected to struggle relative to simple benchmarks; that comparison is
   a result, not a failure.

## Planned

- [x] Confirm series IDs and verify every publication lag
- [x] Point-in-time panel with ragged-edge handling
- [x] Stationarity transforms and ADF tests
- [x] Benchmarks: historical mean, random walk, AR(1), AR(p)
- [x] Expanding-window backtest at each GDP release date
- [x] Diebold-Mariano tests
- [x] Bridge equations (partial-quarter consistent, levels-then-growth)
- [x] Ridge on all indicators
- [x] Rolling-window estimation (drifting trend growth)
- [x] RMSE by days-into-quarter
- [ ] Dynamic factor model (`statsmodels.tsa.statespace.dynamic_factor_mq`)
- [ ] Ridge / gradient boosting comparison
- [ ] RMSE by days-into-quarter against AR(1)

## Data sources

Australian Bureau of Statistics and Reserve Bank of Australia, retrieved via
[`readabs`](https://pypi.org/project/readabs/). All data is publicly available.
