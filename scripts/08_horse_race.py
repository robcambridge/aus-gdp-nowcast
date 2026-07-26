

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgdp.benchmarks import BENCHMARKS, diebold_mariano, run_backtest  # noqa: E402
from ausgdp.bridge import make_bridge, make_bridge_average, make_ridge  # noqa: E402
from ausgdp.config import SHORT_HISTORY  # noqa: E402
from ausgdp.factor import make_dfm  # noqa: E402

PROCESSED = Path("data/processed")
FIGURES = Path("outputs/figures")

# Days after the GDP release at which to form the nowcast.
#
# Publication-lag arithmetic, counting from the end of the PREVIOUS quarter:
#   the GDP release lands on day 65; the target quarter ends on day 91;
#   its own GDP is not published until day 156.
# So a vintage at offset d sits on day 65 + d, and any d up to ~90 is safe.
#
#   0  = GDP release morning (day 65). Only labour force has any of the
#        target quarter; slower indicators have nothing at all.
#   30 = day 95. Target quarter just ended.
#   60 = day 125. Labour force complete, but building approvals (month 3
#        arrives day 129) and exports (day 131) are still one month short.
#   80 = day 145. EVERY indicator complete, still 11 days before GDP.
OFFSETS = [0, 30, 60, 80]

# Estimation window for the regression models, in quarters.
# 40 quarters = 10 years, matching the rolling_mean benchmark. Chosen a priori
# for the same reason: a fixed intercept estimated back to the 1980s anchors on
# a period of higher trend growth and biases forecasts upward.
WINDOW = 40

# The DFM is ~100x slower than a bridge. Forecast every DFM_STRIDE-th quarter so
# a full pass stays a few minutes rather than tens of minutes. Set to 1 for the
# final, definitive run if you are willing to wait ~7 minutes.
DFM_STRIDE = 2


def build_fast_models(indicators: list[str]) -> dict:
    """Benchmarks and bridges: cheap enough to run at every vintage offset.

    The AR term is dropped from the tuned bridges because ar1 was not
    significantly better than the mean on this data (p = 0.30): a coefficient
    that is really zero still costs estimation variance. `bridge_avg_expanding`
    keeps the original specification so the effect stays visible.
    """
    models = dict(BENCHMARKS)
    for name in indicators:
        models[f"bridge_{name}"] = make_bridge(name, add_ar=False, window=WINDOW)
    models["bridge_average"] = make_bridge_average(
        indicators, add_ar=False, window=WINDOW
    )
    models["bridge_avg_expanding"] = make_bridge_average(indicators, add_ar=True)
    models["ridge"] = make_ridge(indicators, window=WINDOW, add_ar=False)
    return models


def build_slow_models(indicators: list[str], all_monthly: list[str]) -> dict:
    """Dynamic factor models: re-estimated by EM at every vintage, so slow.

    Run at a single offset only. One uses the balanced indicator set; the other
    adds the short-history series (household spending) that the bridges dropped,
    since the DFM handles a ragged LEFT edge natively.
    """
    return {
        "dfm_1f": make_dfm(indicators, factors=1),
        "dfm_1f_all": make_dfm(all_monthly, factors=1),
    }


def main() -> None:
    path = PROCESSED / "panel.csv"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts\\06_build_dataset.py first.")

    panel = pd.read_csv(path, parse_dates=["ref_end", "available_from"])

    all_monthly = sorted(set(panel.loc[panel["freq"] == "M", "series"]))
    monthly_series = [s for s in all_monthly if s not in SHORT_HISTORY]
    if not monthly_series:
        sys.exit("No monthly indicators in the panel. Check 06_build_dataset.py.")

    print(f"Panel: {len(panel):,} observations")
    print(f"Balanced indicators: {', '.join(monthly_series)}")
    print(f"DFM-only (short history): {', '.join(sorted(SHORT_HISTORY))}\n")
    print("Note: the DFM models are slow (~1-3 min each). Please wait.\n")

    fast_models = build_fast_models(monthly_series)
    slow_models = build_slow_models(monthly_series, all_monthly)
    richest_offset = max(OFFSETS)

    all_scores, all_results = [], {}

    for offset in OFFSETS:
        res = run_backtest(panel, models=fast_models, vintage_offset_days=offset)
        all_results[offset] = res

        for start, end, label in [
            (None, None, "full"),
            (None, "2019Q4", "pre-COVID"),
            ("1993Q1", "2019Q4", "1993-2019"),
        ]:
            block = res.score(start=start, end=end, label=label)
            if block.empty:
                continue
            block = block.reset_index().rename(columns={"index": "model"})
            block.insert(0, "offset_days", offset)
            all_scores.append(block)

        print("=" * 74)
        print(f"VINTAGE = GDP RELEASE + {offset} DAYS   ({len(res.forecasts)} nowcasts)")
        print("=" * 74)
        table = res.score(start="1993Q1", end="2019Q4", label="1993-2019")
        if table.empty:
            table = res.score(label="full")
        print(table.drop(columns="sample").round(4).to_string())
        print()

    scores = pd.concat(all_scores, ignore_index=True)

    # --- DFM comparison, on a matched (strided) sample ---------------------
    # The DFM refits by EM at every vintage and is ~100x slower than a bridge.
    # We run it every STRIDE-th quarter at the richest offset, and run the fast
    # models on the SAME strided vintages so the RMSE comparison is fair.
    print("\n" + "=" * 74)
    print(f"DYNAMIC FACTOR MODEL   (offset {richest_offset}d, every {DFM_STRIDE} quarters)")
    print("=" * 74)
    print("  Fitting -- the DFM re-estimates by EM at each vintage, please wait.\n")

    compare_models = dict(slow_models)
    compare_models["bridge_average"] = fast_models["bridge_average"]
    compare_models["rolling_mean"] = fast_models["rolling_mean"]

    dfm_res = run_backtest(
        panel, models=compare_models,
        vintage_offset_days=richest_offset, stride=DFM_STRIDE,
    )
    dfm_scores = dfm_res.score(start="1993Q1", end="2019Q4", label="1993-2019")
    if dfm_scores.empty:
        dfm_scores = dfm_res.score(label="full")
    print(f"  {len(dfm_res.forecasts)} nowcasts on the strided sample\n")
    print(dfm_scores.drop(columns="sample").round(4).to_string())

    dfm_err = dfm_res.errors()
    ref = "bridge_average" if "bridge_average" in dfm_err else "rolling_mean"
    print(f"\n  Diebold-Mariano vs {ref} (same strided sample):")
    for name in dfm_err.columns:
        if name == ref:
            continue
        stat, pval = diebold_mariano(dfm_err[name], dfm_err[ref])
        if pd.notna(stat):
            verdict = ("better" if stat < 0 and pval < 0.05
                       else "worse" if stat > 0 and pval < 0.05
                       else "no sig. difference")
            print(f"    {name:<14} dm={stat:>6.3f}  p={pval:.4f}  {verdict}")

    # --- the headline: does more data help? --------------------------------
    print("=" * 74)
    print("RMSE BY HOW MUCH DATA HAS ARRIVED   (sample 1993-2019)")
    print("=" * 74)
    print("  If the indicators carry real signal, these fall left to right.\n")

    modern = scores.loc[scores["sample"] == "1993-2019"]
    if not modern.empty:
        pivot = modern.pivot(index="model", columns="offset_days", values="rmse")
        pivot = pivot.sort_values(pivot.columns[-1], na_position="last")
        print(pivot.round(4).to_string())
        print("\n  (DFM rows show a value only at the richest offset; they are")
        print("   too slow to run at every vintage.)")

    # --- best model vs best benchmark --------------------------------------
    res60 = all_results[OFFSETS[-1]]
    table = res60.score(start="1993Q1", end="2019Q4")
    if table.empty:
        table = res60.score()

    benchmark_names = set(BENCHMARKS)
    best_benchmark = next(m for m in table.index if m in benchmark_names)
    best_overall = table.index[0]

    print("\n" + "=" * 74)
    print(f"DIEBOLD-MARIANO vs best benchmark ({best_benchmark}), offset {OFFSETS[-1]}d")
    print("=" * 74)
    print("  Negative statistic => the challenger is MORE accurate.\n")

    errors = res60.errors()
    mask = (errors.index >= pd.Period("1993Q1", freq="Q")) & (
        errors.index <= pd.Period("2019Q4", freq="Q")
    )
    errors = errors.loc[mask] if mask.any() else errors

    rows = []
    for name in errors.columns:
        if name == best_benchmark:
            continue
        stat, pval = diebold_mariano(errors[name], errors[best_benchmark])
        rows.append(
            {
                "model": name,
                "dm_stat": round(stat, 3) if pd.notna(stat) else None,
                "p_value": round(pval, 4) if pd.notna(pval) else None,
                "verdict": (
                    "no significant difference"
                    if pd.isna(pval) or pval >= 0.05
                    else ("BEATS benchmark" if stat < 0 else "worse")
                ),
            }
        )
    dm = pd.DataFrame(rows).sort_values("dm_stat")
    print(dm.to_string(index=False))

    base = table.loc[best_benchmark, "rmse"]
    top = table.loc[best_overall, "rmse"]
    print(f"\n  Best benchmark : {best_benchmark:<16} rmse {base:.4f}")
    print(f"  Best overall   : {best_overall:<16} rmse {top:.4f}")
    if best_overall != best_benchmark:
        print(f"  Improvement    : {100 * (1 - top / base):.1f}%")
    else:
        print("  No model beat the benchmark. Report that honestly.")

    # --- save --------------------------------------------------------------
    PROCESSED.mkdir(parents=True, exist_ok=True)
    scores.to_csv(PROCESSED / "horserace_scores.csv", index=False)

    out = res60.forecasts.copy()
    out["actual"] = res60.actuals
    out["vintage"] = res60.vintages
    out.to_csv(PROCESSED / "horserace_by_vintage.csv")

    _plots(all_results, modern, best_overall)

    print("\n" + "=" * 74)
    print("  Written:")
    print(f"    {PROCESSED / 'horserace_scores.csv'}")
    print(f"    {PROCESSED / 'horserace_by_vintage.csv'}")
    print(f"    {FIGURES / 'rmse_by_vintage.png'}")
    print(f"    {FIGURES / 'horserace_actual_vs_predicted.png'}")


def _plots(all_results, modern, best_overall) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not available, skipping figures)")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)

    # RMSE by vintage offset
    if not modern.empty:
        pivot = modern.pivot(index="offset_days", columns="model", values="rmse")
        fig, ax = plt.subplots(figsize=(9, 5))
        for col in pivot.columns:
            style = "-" if col == best_overall else "--"
            width = 2.2 if col == best_overall else 1.0
            ax.plot(pivot.index, pivot[col], style, lw=width, label=col, alpha=0.9)
        ax.set_xlabel("days after the GDP release that the nowcast is made")
        ax.set_ylabel("RMSE (percentage points)")
        ax.set_title("Nowcast accuracy as monthly data arrives, 1993-2019", fontsize=11)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / "rmse_by_vintage.png", dpi=150)
        plt.close(fig)

    # Actual vs predicted at the richest information set
    res = all_results[max(all_results)]
    x = res.actuals.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, lw=0.8, color="0.7")
    ax.plot(x, res.actuals.to_numpy(), lw=1.6, color="#1b2a4a", label="actual")
    ax.plot(
        x, res.forecasts[best_overall].to_numpy(), lw=1.3, color="#c1440e",
        ls="--", label=f"{best_overall} nowcast",
    )
    ax.set_title(
        "Australian quarterly GDP growth: actual vs point-in-time nowcast", fontsize=11
    )
    ax.set_ylabel("% change on previous quarter")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "horserace_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
