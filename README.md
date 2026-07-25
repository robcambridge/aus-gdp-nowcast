# Australian GDP Nowcasting System

A point-in-time nowcasting system for Australian quarterly real GDP growth,
built from monthly ABS and RBA indicators. Includes a backtesting framework, a
model horse race, a mixed-frequency dynamic factor model with news
decomposition, a live current-quarter nowcast, and a browser dashboard.

**Live headline:** a rolling-window combination of bridge equations nowcasts the
current (unpublished) quarter, cross-checked against a dynamic factor model and
univariate benchmarks.

---

## The problem, stated precisely

The ABS publishes quarterly GDP about nine weeks after the quarter ends — the
June quarter lands in early September. So when GDP for quarter *t* is released,
quarter *t+1* is already over but unmeasured. This system nowcasts *t+1* using
only data published on or before a given date — the way the RBA and Treasury
actually operate.

## Why point-in-time is the whole game

The informational edge comes from monthly data arriving before GDP does. That
edge is only real if the backtest respects **when each number was actually
published**. Every observation carries an `available_from` date, and snapshots
are built by filtering on it, so look-ahead bias is structurally impossible
rather than something to remember. Publication lags are conservative upper
bounds verified against the ABS release calendar; each lag records the observed
release dates behind it in `config.py`.

## Headline result

On an out-of-sample backtest (1993–2019, one nowcast per GDP release):

- A **rolling-window average of bridge equations** is the most accurate model,
  ~2.5% better than a rolling-mean benchmark, with accuracy improving as more
  monthly data arrives through the quarter.
- The improvement is **not statistically significant** (Diebold–Mariano
  p ≈ 0.17).
- A **mixed-frequency dynamic factor model** does not beat the simpler
  combination, and adding a short-history spending series does not help.

The honest conclusion: standard monthly indicators offer limited, non-significant
uplift for Australian GDP at a one-quarter horizon, and flexible models do not
outperform simple ones on a ~130-quarter sample. The framework is built so that
this is a credible finding rather than a disappointing one.

## Quickstart

```bash
uv sync
uv run pytest                              # 69 tests

# one-time: confirm ABS series IDs, then download
uv run python scripts/01_discover.py
uv run python scripts/05_check_series.py

# build + evaluate
uv run python scripts/06_build_dataset.py  # point-in-time panel
uv run python scripts/07_benchmark.py      # benchmarks
uv run python scripts/08_horse_race.py     # full horse race (incl. DFM, slow)
uv run python scripts/09_news.py           # news decomposition
uv run python scripts/10_live_nowcast.py   # current-quarter nowcast
uv run python scripts/11_build_dashboard_data.py

# or chain them all
uv run python scripts/run_all.py

# view the dashboard
python -m http.server -d dashboard 8000    # then open localhost:8000
```

## The dashboard

`dashboard/index.html` is a static page (no server, no build step) that reads
`dashboard/data.json`. It shows the live nowcast, the accuracy-by-vintage curve,
the model horse race, nowcast-vs-outcome over history, and the news
decomposition. Deployable as-is to GitHub Pages.

## Models

| Model | Idea |
|---|---|
| mean / rolling mean | historical average of GDP growth |
| random walk | next quarter = this quarter |
| AR(1) / AR(p) | GDP's own past only |
| bridge equations | regress GDP on each monthly indicator's quarterly reading |
| **bridge average** | equal-weight combination of bridges (headline) |
| ridge | all indicators jointly, shrinkage |
| dynamic factor model | pool all indicators through a latent factor (Kalman/EM) |

## Layout

```
src/ausgdp/
  config.py       series registry + verified publication lags
  fetch.py        ABS/RBA download via readabs
  dataset.py      long panel, Snapshot, as_of(), ragged-edge diagnostics
  transforms.py   panel prep, ADF tests, levels-to-quarterly regressors
  benchmarks.py   Context, backtest, Diebold-Mariano, benchmark models
  bridge.py       bridge equations, ridge, forecast combination
  factor.py       mixed-frequency dynamic factor model + news decomposition
scripts/          01-11 pipeline + run_all.py
dashboard/        static HTML dashboard + data.json
tests/            69 tests: leakage, ragged edge, transforms, models, news
```

## Method notes worth reading

- **Bridge construction.** Monthly indicators enter the panel as levels; the
  regressor is the change in the quarterly average level (matching how GDP is
  measured), not the average of monthly growth rates. A controlled test
  (`test_regressor_recovers_truth_better_than_averaging_growth_rates`) shows this
  recovers true quarterly growth with ~60% lower error at the one-month edge.
- **Rolling estimation.** Bridge intercepts are estimated on a 40-quarter window,
  because Australian trend growth has declined and an expanding window anchors on
  a stale average — the same reason a rolling mean beats an expanding one here.
- **Vintage timing.** The backtest runs at 0/30/60/80 days after each GDP
  release; a leak assertion fires if the offset ever reaches the target quarter's
  own publication.

## Limitations

1. **Final-vintage data**, not real-time vintages — measured accuracy is
   optimistic versus a true real-time system. `readabs` supports historical
   vintages (`history=`) as a future extension.
2. **Constant publication lags** across the sample, though ABS releases were
   slower historically.
3. **Short sample** (~130 quarters post-1983), which limits flexible models.
4. **COVID:** 2020Q2 is a ~10σ observation; results are reported full-sample,
   pre-COVID, and 1993–2019 separately.

## Data sources

Australian Bureau of Statistics and Reserve Bank of Australia, via
[`readabs`](https://pypi.org/project/readabs/). All data is publicly available.
This is a personal research project and not investment advice.
