# Australian GDP Nowcasting

Point-in-time nowcasting of Australian quarterly real GDP growth using monthly
ABS and RBA indicators.

**Status:** Phase 1 (data infrastructure). Modelling not yet implemented.

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
```

## Layout

```
src/ausgdp/config.py    Series registry + publication-lag table
src/ausgdp/fetch.py     ABS/RBA download via readabs
src/ausgdp/dataset.py   Long panel, Snapshot, as_of(), ragged-edge diagnostics
scripts/01_discover.py  Confirm ABS series IDs (needs network)
scripts/02_demo_*.py    Offline demonstration of the point-in-time logic
tests/                  Leakage and ragged-edge invariants
```

## Limitations (read before believing any result)

1. **Publication lags are assumed constant and are currently UNVERIFIED.**
   Run `python -m ausgdp.config` to list unverified series. Each must be
   checked against the [ABS release calendar](https://www.abs.gov.au/release-calendar)
   before results mean anything. Lags were also longer historically than today.

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

- [ ] Confirm series IDs and verify every publication lag
- [ ] Benchmarks: historical mean, random walk, AR(1), AR(p)
- [ ] Bridge equations
- [ ] Dynamic factor model (`statsmodels.tsa.statespace.dynamic_factor_mq`)
- [ ] Ridge / gradient boosting comparison
- [ ] Expanding-window backtest at each GDP release date
- [ ] RMSE by days-into-quarter
- [ ] Diebold–Mariano tests against AR(1)

## Data sources

Australian Bureau of Statistics and Reserve Bank of Australia, retrieved via
[`readabs`](https://pypi.org/project/readabs/). All data is publicly available.
