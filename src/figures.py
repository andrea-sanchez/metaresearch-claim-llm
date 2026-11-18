"""
Figures of ``reports/paper`` (file names match the LaTeX \\includegraphics).

    35ScoresResults2.png     overall score per paper and evaluator      (Fig. 1)
    4ScoresDistribution2.png distribution of overall scores             (Fig. 2a)
    5ScoresDomain2.png       mean score per domain and evaluator        (Fig. 2b)
    2Spearmanrho.png         Spearman ρ heatmaps                        (Fig. 3)
    6MetadataresultsAll.png  overall score by study metadata            (Appendix B)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import HUMANS, LLMS, OUTCOME_LABELS, OUTCOMES, wide

EVALUATORS = [HUMANS, *LLMS]
COLORS = {  # palette of the original figures
    HUMANS: "#B2182B",
    "ChatGPT": "#0B2E5C",
    "Claude": "#3A78B8",
    "Gemini": "#A6D6E6",
}
OVERALL = "all_domains"
DPI = 300

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
})


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def scores_per_paper(scores: pd.DataFrame, path: Path) -> None:
    w = wide(scores, OVERALL).sort_index()
    x = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(16, 4.2))
    for ev in EVALUATORS:
        ax.plot(x, w[ev], marker="o", ms=4, lw=1.6, color=COLORS[ev], label=ev)
    ax.set_xticks(x, w.index.astype(str), rotation=45, fontsize=8)
    ax.set_xlim(-1, len(w))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Paper ID")
    ax.set_ylabel("Overall CLAIM score (%)")
    ax.legend(ncol=4, loc="upper right", frameon=False)
    _save(fig, path)


def score_distribution(scores: pd.DataFrame, path: Path, seed: int = 0) -> None:
    w = wide(scores, OVERALL)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [w[ev].dropna() for ev in EVALUATORS]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops={"color": "#1F1F1F"})
    for patch, ev in zip(bp["boxes"], EVALUATORS):
        patch.set_facecolor(COLORS[ev])
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14,
                   color="#1F1F1F", alpha=0.5, zorder=3, linewidths=0)
    ax.set_xticks(range(1, len(EVALUATORS) + 1), EVALUATORS)
    ax.set_xlabel("Evaluator")
    ax.set_ylabel("Overall CLAIM score (%)")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def domain_means(scores: pd.DataFrame, path: Path) -> None:
    means = scores.groupby(["outcome", "evaluator"])["score"].mean().unstack()
    means = means.loc[OUTCOMES, [*LLMS, HUMANS]]
    x = np.arange(len(OUTCOMES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, ev in enumerate(means.columns):
        bars = ax.bar(x + (j - 1.5) * width, means[ev], width * 0.95,
                      color=COLORS[ev], label=ev)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in OUTCOMES])
    ax.set_xlabel("CLAIM domain")
    ax.set_ylabel("Mean score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", frameon=False)
    _save(fig, path)


def spearman_heatmaps(rho_h: pd.DataFrame, rho_llm: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    panels = [(rho_h, "(a) LLM vs humans"), (rho_llm, "(b) LLM vs LLM")]
    for ax, (df, title) in zip(axes, panels):
        rho = df.pivot(index="outcome", columns="pair", values="rho")
        sig = df.pivot(index="outcome", columns="pair", values="significant")
        cols = list(dict.fromkeys(df["pair"]))
        rho, sig = rho.loc[OUTCOMES, cols], sig.loc[OUTCOMES, cols]
        im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                v = rho.iat[i, j]
                ax.text(j, i, f"{v:.2f}{'*' if sig.iat[i, j] else ''}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(v) > 0.5 else "#1F1F1F")
        ax.set_xticks(range(len(cols)), cols, rotation=20)
        ax.set_yticks(range(len(OUTCOMES)), [OUTCOME_LABELS[o] for o in OUTCOMES])
        ax.set_title(f"{title} (* = significant, Holm-adjusted)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, label="Spearman ρ", shrink=0.9)
    _save(fig, path)


METADATA_PANELS = [  # (column, max categories shown)
    ("performance_evaluation", None), ("public_code", None),
    ("year", None), ("journal", None),
    ("country", 12), ("images", None),
    ("segmentation", 12), ("cases", 12),
    ("dataset", None), ("validation", None),
]


def metadata_strips(scores: pd.DataFrame, full: pd.DataFrame, path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    meta = full.loc[full["evaluator"] == HUMANS, ["paper_id", *(c for c, _ in METADATA_PANELS)]].set_index("paper_id")
    overall = scores[scores["outcome"] == OVERALL].join(meta, on="paper_id")

    fig, axes = plt.subplots(5, 2, figsize=(16, 24), constrained_layout=True)
    offsets = dict(zip(EVALUATORS, np.linspace(-0.27, 0.27, len(EVALUATORS))))
    for ax, (col, top) in zip(axes.ravel(), METADATA_PANELS):
        values = overall[col]
        if pd.api.types.is_float_dtype(values):  # e.g. cases: 20.0 -> "20"
            values = values.astype("Int64")
        values = values.astype("string").fillna("NR")
        counts = values[overall["evaluator"] == HUMANS].value_counts()
        cats = counts.index[:top].tolist() if top else counts.index.tolist()
        ypos = {c: i for i, c in enumerate(cats)}
        for ev in EVALUATORS:
            sel = (overall["evaluator"] == ev) & values.isin(cats)
            y = values[sel].map(ypos).to_numpy(float) + offsets[ev]
            ax.scatter(overall.loc[sel, "score"], y + rng.uniform(-0.04, 0.04, len(y)),
                       s=22, alpha=0.75, color=COLORS[ev], label=ev, linewidths=0)
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        title = col.replace("_", " ").capitalize()
        ax.set_title(f"{title} (top {top})" if top and len(counts) > top else title)
        ax.set_xlabel("Overall CLAIM score (%)")
    axes[0, 0].legend(title="Evaluator", loc="lower left", frameon=True)
    _save(fig, path)


def make_all(scores: pd.DataFrame, full: pd.DataFrame, results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_per_paper(scores, out_dir / "35ScoresResults2.png")
    score_distribution(scores, out_dir / "4ScoresDistribution2.png")
    domain_means(scores, out_dir / "5ScoresDomain2.png")
    spearman_heatmaps(results["spearman_llm_humans"], results["spearman_llm_llm"],
                      out_dir / "2Spearmanrho.png")
    metadata_strips(scores, full, out_dir / "6MetadataresultsAll.png")
