"""
Figures of the paper, written to ``reports/paper/images/``.

    fig1_overall_scores_per_paper.png   overall score per paper and evaluator  (Fig. 1)
    fig2a_overall_score_distribution.png distribution of overall scores       (Fig. 2a)
    fig2b_mean_score_by_domain.png      mean score per domain and evaluator    (Fig. 2b)
    fig3_spearman_heatmaps.png          Spearman rho, LLM-humans and LLM-LLM    (Fig. 3)
    figC1_scores_by_metadata.png        overall score by study characteristic  (Appendix C)

All figures are deterministic: point jitter uses a fixed random seed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis import EVALUATORS, OUTCOME_LABELS, OUTCOMES, REFERENCE, wide

PALETTE = {
    "Humans": "#B2182B",
    "ChatGPT": "#053061",
    "Claude": "#3677B8",
    "Gemini": "#A6D9E8",
}
OVERALL = "all_domains"
DPI = 300
SEED = 0

FIGURE_FILES = {
    "scores_per_paper": "fig1_overall_scores_per_paper.png",
    "score_distribution": "fig2a_overall_score_distribution.png",
    "domain_means": "fig2b_mean_score_by_domain.png",
    "spearman": "fig3_spearman_heatmaps.png",
    "metadata": "figC1_scores_by_metadata.png",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scores_per_paper(scores: pd.DataFrame, outcome: str = OVERALL) -> plt.Figure:
    """Score of every paper (sorted by ID) for each evaluator."""
    w = wide(scores, outcome).sort_index()
    x = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(18, 4.5))
    for ev in EVALUATORS:
        ax.plot(x, w[ev], marker="o", markersize=4, linewidth=1.5,
                color=PALETTE[ev], label=ev)
    ax.set_xticks(x, w.index.astype(str), rotation=45)
    ax.set_xlim(-1, len(w))
    ax.set_ylim(0, 105)
    ax.set_title(OUTCOME_LABELS[outcome], fontsize=19, pad=10)
    ax.set_xlabel("Paper ID", fontsize=17)
    ax.set_ylabel("Score (%)", fontsize=17)
    ax.grid(alpha=0.3)
    ax.legend(ncol=len(EVALUATORS), loc="upper right", fontsize=12)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: pd.DataFrame, outcome: str = OVERALL) -> plt.Figure:
    """Boxplot of the scores by evaluator, with one point per paper."""
    data = scores[scores["outcome"] == outcome]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x="evaluator", y="score", order=EVALUATORS, hue="evaluator",
                palette=PALETTE, legend=False, showfliers=False, ax=ax)
    for i, ev in enumerate(EVALUATORS):
        y = data.loc[data["evaluator"] == ev, "score"].dropna()
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(y)), y, s=30, color="black",
                   alpha=0.5, linewidths=0, zorder=3)
    ax.set_title("Distribution of CLAIM scores by evaluator", fontsize=19, pad=15)
    ax.set_xlabel("Evaluator", fontsize=17)
    ax.set_ylabel("CLAIM score (%)", fontsize=17)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    return fig


def plot_domain_means(scores: pd.DataFrame) -> plt.Figure:
    """Mean score per CLAIM domain (and overall) for each evaluator."""
    means = (scores.groupby(["outcome", "evaluator"])["score"].mean()
             .unstack().loc[OUTCOMES, EVALUATORS])
    means.index = [OUTCOME_LABELS[o] for o in means.index]

    fig, ax = plt.subplots(figsize=(13, 7))
    means.plot(kind="bar", ax=ax, color=PALETTE, width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=4, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Average CLAIM domain scores by evaluator", fontsize=19, pad=15)
    ax.set_xlabel("CLAIM domain", fontsize=17)
    ax.set_ylabel("Mean score (%)", fontsize=17)
    ax.tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(ncol=len(EVALUATORS), loc="upper right", fontsize=12, frameon=True)
    fig.tight_layout()
    return fig


def plot_spearman_heatmaps(rho_h: pd.DataFrame, rho_llm: pd.DataFrame) -> plt.Figure:
    """Spearman rho per outcome: (a) LLM vs humans, (b) LLM vs LLM.
    Asterisk = significant after Holm correction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def draw(ax, df, title, show_cbar):
        cols = list(dict.fromkeys(df["pair"]))
        rho = df.pivot(index="outcome", columns="pair", values="rho").loc[OUTCOMES, cols]
        sig = df.pivot(index="outcome", columns="pair", values="significant").loc[OUTCOMES, cols]
        annot = rho.map("{:.2f}".format) + sig.map(lambda s: "*" if s else "")
        sns.heatmap(rho.rename(index=OUTCOME_LABELS), annot=annot.to_numpy(), fmt="",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5,
                    cbar=show_cbar, cbar_kws={"label": "Spearman rho"} if show_cbar else None,
                    ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)

    draw(axes[0], rho_h, "(a) LLM vs humans (* = significant, adjusted)", False)
    draw(axes[1], rho_llm, "(b) LLM vs LLM (* = significant, adjusted)", True)
    fig.tight_layout()
    return fig


METADATA = [  # (column of the full dataset, max. number of categories shown)
    ("performance_evaluation", None), ("public_code", None),
    ("year", None), ("journal", None),
    ("country", 12), ("images", None),
    ("segmentation", 12), ("cases", 12),
    ("dataset", None), ("validation", None),
]


def plot_metadata(scores: pd.DataFrame, full: pd.DataFrame, outcome: str = OVERALL) -> plt.Figure:
    """Scores by study characteristic (strip plots, most frequent categories)."""
    rng = np.random.default_rng(SEED)
    meta_cols = [c for c, _ in METADATA]
    meta = full.loc[full["evaluator"] == REFERENCE, ["paper_id", *meta_cols]].set_index("paper_id")
    data = scores[scores["outcome"] == outcome].join(meta, on="paper_id")
    offsets = dict(zip(EVALUATORS, np.linspace(-0.27, 0.27, len(EVALUATORS))))

    fig, axes = plt.subplots(int(np.ceil(len(METADATA) / 2)), 2, figsize=(25, 35))
    for ax, (col, top) in zip(axes.ravel(), METADATA):
        values = data[col]
        if pd.api.types.is_float_dtype(values):  # e.g. cases: 20.0 -> "20"
            values = values.astype("Int64")
        values = values.astype("string").fillna("NR")  # NR = not reported
        counts = values[data["evaluator"] == REFERENCE].value_counts()
        cats = counts.index[:top].tolist() if top else counts.index.tolist()
        if pd.api.types.is_numeric_dtype(data[col]):  # numeric: ascending order, NR last
            cats.sort(key=lambda c: (c == "NR", float(c) if c != "NR" else 0))
        ypos = {c: i for i, c in enumerate(cats)}

        for ev in EVALUATORS:
            sel = (data["evaluator"] == ev) & values.isin(cats)
            y = values[sel].map(ypos).to_numpy(float) + offsets[ev]
            ax.scatter(data.loc[sel, "score"], y + rng.uniform(-0.04, 0.04, len(y)),
                       s=60, alpha=0.7, color=PALETTE[ev], label=ev, linewidths=0)

        title = col.replace("_", " ").capitalize()
        ax.set_title(f"{title} (top {top} most frequent)" if top and len(counts) > top else title,
                     fontsize=18, pad=12)
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        ax.set_xlabel(f"{OUTCOME_LABELS[outcome]} CLAIM score (%)", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes.ravel()[0].legend(title="Evaluator", fontsize=13, title_fontsize=14)
    for ax in axes.ravel()[len(METADATA):]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig


def make_all(scores: pd.DataFrame, full: pd.DataFrame, results: dict, out_dir: Path) -> None:
    """Write every figure of the paper to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "scores_per_paper": plot_scores_per_paper(scores),
        "score_distribution": plot_score_distribution(scores),
        "domain_means": plot_domain_means(scores),
        "spearman": plot_spearman_heatmaps(results["spearman_llm_humans"],
                                           results["spearman_llm_llm"]),
        "metadata": plot_metadata(scores, full),
    }
    for name, fig in figures.items():
        _save(fig, out_dir / FIGURE_FILES[name])
