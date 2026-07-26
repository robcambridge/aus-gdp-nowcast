"""Generate publication-quality figures for the research report.

Reads the CSVs your pipeline produced (data/processed/) and writes PNGs to
report/figures/. Kept separate from the dashboard figures so the report can have
its own print-friendly styling (white background, serif-compatible).

Called by scripts/12_build_report.py; also runnable alone:
    uv run python -m ausgdp.report_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROCESSED = Path("data/processed")
FIGDIR = Path("report/figures")

# Print-friendly palette
INK = "#1a1a2e"
ACCENT = "#c1440e"
BLUE = "#2b6cb0"
GREEN = "#2f7d4f"
GREY = "#9aa5b1"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fig_vintage_curve() -> Path | None:
    """RMSE vs vintage offset for each model, modern sample."""
    scores = _read("horserace_scores.csv")
    if scores.empty:
        return None
    modern = scores[scores["sample"] == "1993-2019"]
    if modern.empty:
        modern = scores[scores["sample"] == "full"]
    if modern.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for model in modern["model"].unique():
        rows = modern[modern["model"] == model].sort_values("offset_days")
        if model == "bridge_average":
            ax.plot(rows["offset_days"], rows["rmse"], "-o", lw=2.4, color=ACCENT,
                    label="bridge average", zorder=5, ms=5)
        elif model == "rolling_mean":
            ax.plot(rows["offset_days"], rows["rmse"], "--", lw=1.6, color=BLUE,
                    label="rolling mean", zorder=4)
        else:
            ax.plot(rows["offset_days"], rows["rmse"], "-", lw=0.9, color=GREY,
                    alpha=0.6, zorder=1)

    ax.set_xlabel("days after prior GDP release")
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_title("Nowcast accuracy as monthly data arrives (1993–2019)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "vintage_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_actual_vs_pred() -> Path | None:
    """Headline model nowcast vs actual over history."""
    fc = _read("horserace_by_vintage.csv")
    if fc.empty or "actual" not in fc.columns:
        return None
    col = "bridge_average" if "bridge_average" in fc.columns else None
    if col is None:
        return None

    x = pd.PeriodIndex(fc["target"], freq="Q").to_timestamp()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axhline(0, lw=0.7, color=GREY)
    ax.plot(x, fc["actual"], lw=1.6, color=INK, label="actual")
    ax.plot(x, fc[col], lw=1.3, color=ACCENT, ls="--", label="bridge-average nowcast")
    ax.set_ylabel("% change on previous quarter")
    ax.set_title("Point-in-time nowcast vs actual GDP growth")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "actual_vs_pred.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_news() -> Path | None:
    """News decomposition waterfall, if available."""
    news = _read("news_decomposition.csv")
    if news.empty or "updated variable" not in news.columns:
        return None
    by_var = news.groupby("updated variable")["impact"].sum().sort_values()
    if by_var.empty:
        return None

    colors = [ACCENT if v < 0 else GREEN for v in by_var.to_numpy()]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(by_var.index.str.replace("_", " "), by_var.to_numpy(), color=colors)
    ax.axvline(0, lw=0.8, color="#444")
    ax.set_xlabel("contribution to change in nowcast (pp)")
    ax.set_title("News decomposition: what moved the nowcast")
    fig.tight_layout()
    out = FIGDIR / "news.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_all() -> dict[str, Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = {}
    for name, fn in [
        ("vintage_curve", fig_vintage_curve),
        ("actual_vs_pred", fig_actual_vs_pred),
        ("news", fig_news),
    ]:
        path = fn()
        if path is not None:
            figs[name] = path
            print(f"  wrote {path}")
        else:
            print(f"  skipped {name} (no data)")
    return figs


if __name__ == "__main__":
    generate_all()
