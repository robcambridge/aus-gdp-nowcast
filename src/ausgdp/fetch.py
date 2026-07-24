"""Download series from the ABS and RBA.

Thin wrapper over `readabs`. Two jobs:

  1. Turn a SeriesSpec into an actual pandas Series with a PeriodIndex.
  2. Fail loudly and legibly when the ABS has renamed something, rather than
     silently returning the wrong series.

`readabs` caches downloads to ./.readabs_cache/, so re-running is cheap.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .config import ALL_SPECS, SeriesSpec

CACHE_DIR = Path("data/raw")


def _require_readabs():
    try:
        import readabs as ra
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "readabs is not installed. Run:  uv add readabs"
        ) from exc
    return ra


def fetch_abs(spec: SeriesSpec, verbose: bool = False) -> pd.Series:
    """Fetch one ABS series described by `spec`.

    Uses the exact Series ID if `spec.series_id` is set (fast, unambiguous),
    otherwise falls back to a metadata search on `spec.search`.
    """
    ra = _require_readabs()
    from readabs import metacol as mc  # noqa: F401  (available for your own searches)

    if spec.series_id:
        data, meta = ra.read_abs_series(cat=spec.collection, series_id=spec.series_id)
        col = spec.series_id
        if col not in data.columns:
            raise KeyError(
                f"{spec.name}: series ID {spec.series_id} not found in {spec.collection}. "
                "It may have been discontinued. Re-run scripts/01_discover.py."
            )
        series = data[col]
    else:
        data, meta = ra.read_abs_cat(spec.collection, verbose=verbose)
        try:
            table, series_id, units = ra.find_abs_id(meta, spec.search)
        except ValueError as exc:
            matches = ra.search_abs_meta(meta, spec.search)
            raise ValueError(
                f"{spec.name}: search terms did not identify a unique series in "
                f"{spec.collection} ({len(matches)} matches). "
                "Run scripts/01_discover.py to inspect candidates and then pin an "
                "exact series_id in config.py."
            ) from exc
        series = data[table][series_id]
        if verbose:
            print(f"  {spec.name}: table={table} id={series_id} units={units}")

    series = series.dropna()
    series.name = spec.name
    _check_frequency(series, spec)
    return series


def fetch_rba(spec: SeriesSpec, verbose: bool = False) -> pd.Series:
    """Fetch one RBA series.

    Special-cases the cash rate, which readabs exposes directly.
    """
    ra = _require_readabs()

    if spec.collection.upper() == "OCR":
        series = ra.read_rba_ocr(monthly=(spec.freq == "M"))
    else:
        data, meta = ra.read_rba_table(spec.collection)
        raise NotImplementedError(
            f"{spec.name}: RBA table {spec.collection} downloaded ({data.shape}), but "
            "you must pick the column you want. Inspect `meta` and pin the column "
            "name in fetch_rba(). RBA tables vary too much to guess safely."
        )

    series = series.dropna()
    series.name = spec.name
    _check_frequency(series, spec)
    return series


def _check_frequency(series: pd.Series, spec: SeriesSpec) -> None:
    """Warn if what came back is not the frequency we registered."""
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError(f"{spec.name}: expected PeriodIndex, got {type(series.index)}")
    actual = series.index.freqstr[0]
    if actual != spec.freq:
        warnings.warn(
            f"{spec.name}: registered as freq='{spec.freq}' but ABS returned "
            f"'{series.index.freqstr}'. Fix config.py before modelling.",
            stacklevel=2,
        )


def fetch_one(spec: SeriesSpec, verbose: bool = False) -> pd.Series:
    return fetch_abs(spec, verbose) if spec.source == "abs" else fetch_rba(spec, verbose)


def fetch_all(
    specs: list[SeriesSpec] | None = None, verbose: bool = True
) -> dict[str, pd.Series]:
    """Fetch everything in the registry, skipping (loudly) whatever fails.

    Partial failure is normal: the ABS restructures collections. Better to get
    eight series and a clear list of two problems than to crash on the first.
    """
    specs = specs or ALL_SPECS
    out: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}

    for spec in specs:
        try:
            out[spec.name] = fetch_one(spec, verbose=verbose)
            if verbose:
                s = out[spec.name]
                print(f"  OK   {spec.name:<20} {len(s):>4} obs  {s.index[0]} to {s.index[-1]}")
        except Exception as exc:  # noqa: BLE001 - we want every failure reported
            failures[spec.name] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  FAIL {spec.name:<20} {type(exc).__name__}")

    if failures and verbose:
        print(f"\n{len(failures)} series failed:")
        for name, msg in failures.items():
            print(f"\n  {name}\n    {msg}")

    return out


def save_raw(series_map: dict[str, pd.Series], directory: Path = CACHE_DIR) -> None:
    """Persist each raw series to CSV so your repo is reproducible offline."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, series in series_map.items():
        path = directory / f"{name}.csv"
        series.to_frame("value").to_csv(path, index_label="ref_period")
    print(f"Saved {len(series_map)} series to {directory}/")


def load_raw(directory: Path = CACHE_DIR) -> dict[str, pd.Series]:
    """Reload previously saved raw series (no network needed)."""
    out = {}
    for path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(path)
        name = path.stem
        idx = pd.PeriodIndex(df["ref_period"], freq="infer")
        out[name] = pd.Series(df["value"].to_numpy(), index=idx, name=name)
    return out
