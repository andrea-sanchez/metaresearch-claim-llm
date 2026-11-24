"""
Statistical analysis: LLM vs. human CLAIM compliance scores (Methods, Section 2.2).

    agreement           MAE, ME and Wilcoxon signed-rank test vs. humans,
                        Holm over 15 tests                                   -> Table 1
    model_differences   Friedman test on absolute errors (Holm over 5 outcomes)
                        and pairwise Wilcoxon tests (Holm within outcome)    -> Table 2
    correlations        Spearman rho LLM-humans and LLM-LLM, Holm per family -> Fig. 3
    descriptives        mean, SD and quartiles per evaluator and outcome     -> Section 3

Input: tidy score table from ``src.data.prepare_scores`` with columns
``paper_id, evaluator, outcome, score`` (score in whole-number percent).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from src import data as D

REFERENCE = D.HUMAN_EVALUATOR
LLMS = ["ChatGPT", "Claude", "Gemini"]  # order used in the paper
EVALUATORS = [REFERENCE, *LLMS]
OUTCOMES = D.SCORE_COLS
OUTCOME_LABELS = {
    "study_description_domain": "Study description",
    "dataset_description_domain": "Dataset description",
    "model_description_domain": "Model description",
    "performance_domain": "Model performance",
    "all_domains": "Overall",
}
ALPHA = 0.05
CORRECTION = "holm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wide(scores: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Papers x evaluators score matrix for one outcome."""
    w = scores[scores["outcome"] == outcome].pivot(
        index="paper_id", columns="evaluator", values="score"
    )
    return w[EVALUATORS]


def adjust(pvalues) -> np.ndarray:
    """Multiple-comparison adjustment (Holm)."""
    return multipletests(pvalues, method=CORRECTION)[1]


# ---------------------------------------------------------------------------
# Agreement (MAE) and strictness (ME + Wilcoxon)
# ---------------------------------------------------------------------------


def agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """MAE, ME and two-sided Wilcoxon signed-rank test of LLM - human
    differences (zero differences excluded); Holm over all 15 tests."""
    rows = []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        for model in LLMS:
            diff = (w[model] - w[REFERENCE]).dropna()
            p = wilcoxon(diff, zero_method="wilcox").pvalue if diff.any() else 1.0
            rows.append({
                "outcome": outcome, "model": model, "n": len(diff),
                "mae": diff.abs().mean(), "me": diff.mean(), "p_value": p,
            })
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_value"])
    df["significant"] = df["p_adj"] < ALPHA
    df["direction"] = np.select(
        [df["me"] < 0, df["me"] > 0], ["Stricter", "More lenient"], default="No bias"
    )
    return df


def ranking(agreement_df: pd.DataFrame, outcome: str = "all_domains") -> pd.DataFrame:
    """Models ranked by MAE (rank 1 = closest to humans) for one outcome."""
    rank = (agreement_df.loc[agreement_df["outcome"] == outcome, ["model", "mae"]]
            .sort_values("mae").reset_index(drop=True))
    rank.insert(0, "rank", range(1, len(rank) + 1))
    return rank


# ---------------------------------------------------------------------------
# Differences between models
# ---------------------------------------------------------------------------


def model_differences(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test on the absolute errors of the LLMs (papers as blocks;
    Holm over outcomes) and pairwise Wilcoxon tests on the paired absolute
    errors (Holm within each outcome)."""
    friedman_rows, pairwise_rows = [], []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        abs_err = w[LLMS].sub(w[REFERENCE], axis=0).abs().dropna()

        chi2, p = friedmanchisquare(*(abs_err[m] for m in LLMS))
        friedman_rows.append({"outcome": outcome, "n": len(abs_err), "chi2": chi2, "p_value": p})

        pairs = list(combinations(LLMS, 2))
        pvals = [wilcoxon(abs_err[a], abs_err[b], zero_method="wilcox").pvalue for a, b in pairs]
        for (a, b), p_raw, p_adj in zip(pairs, pvals, adjust(pvals)):
            mae_a, mae_b = abs_err[a].mean(), abs_err[b].mean()
            pairwise_rows.append({
                "outcome": outcome, "model_a": a, "model_b": b, "n": len(abs_err),
                "mae_a": mae_a, "mae_b": mae_b, "p_value": p_raw, "p_adj": p_adj,
                "significant": p_adj < ALPHA,
                "closer_to_humans": a if mae_a < mae_b else b if mae_b < mae_a else "Equal",
            })

    friedman = pd.DataFrame(friedman_rows)
    friedman["p_adj"] = adjust(friedman["p_value"])
    friedman["significant"] = friedman["p_adj"] < ALPHA
    return friedman, pd.DataFrame(pairwise_rows)


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def correlations(scores: pd.DataFrame, reference: str | None = REFERENCE) -> pd.DataFrame:
    """Spearman rho per outcome (average ranks for ties).

    With ``reference``: each LLM vs the reference; with ``None``: every pair
    of LLMs. Holm correction over the whole family (15 tests).
    """
    pairs = [(m, reference) for m in LLMS] if reference else list(combinations(LLMS, 2))
    rows = []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        for a, b in pairs:
            pair = w[[a, b]].dropna()
            rho, p = spearmanr(pair[a], pair[b])
            rows.append({"outcome": outcome, "pair": f"{a} vs {b}", "n": len(pair),
                         "rho": rho, "p_value": p})
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_value"])
    df["significant"] = df["p_adj"] < ALPHA
    return df


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def descriptives(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and quartiles per outcome and evaluator."""
    g = scores.groupby(["outcome", "evaluator"])["score"]
    out = g.describe().rename(columns={"25%": "q1", "50%": "median", "75%": "q3"}).reset_index()
    out["outcome"] = pd.Categorical(out["outcome"], OUTCOMES, ordered=True)
    out["evaluator"] = pd.Categorical(out["evaluator"], EVALUATORS, ordered=True)
    return out.sort_values(["outcome", "evaluator"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration and reporting
# ---------------------------------------------------------------------------


def run_analysis(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every statistic reported in the paper."""
    friedman, pairwise = model_differences(scores)
    return {
        "descriptives": descriptives(scores),
        "agreement": agreement(scores),
        "friedman": friedman,
        "pairwise": pairwise,
        "spearman_llm_humans": correlations(scores, reference=REFERENCE),
        "spearman_llm_llm": correlations(scores, reference=None),
    }


def summary(results: dict[str, pd.DataFrame]) -> str:
    """Key numbers of the paper, one per line."""
    agr, fr = results["agreement"], results["friedman"]
    rho_h, rho_llm = results["spearman_llm_humans"], results["spearman_llm_llm"]
    return "\n".join([
        f"Agreement   MAE {agr['mae'].min():.1f}-{agr['mae'].max():.1f} pp",
        f"Strictness  {(agr['me'] < 0).sum()}/{len(agr)} stricter, "
        f"{agr['significant'].sum()}/{len(agr)} significant biases",
        f"Models      differ in {fr['significant'].sum()}/{len(fr)} outcomes (Friedman)",
        f"Spearman    LLM-humans {rho_h['rho'].min():.2f} to {rho_h['rho'].max():.2f} "
        f"({rho_h['significant'].sum()}/{len(rho_h)} significant); "
        f"LLM-LLM {rho_llm['rho'].min():.2f} to {rho_llm['rho'].max():.2f} "
        f"({rho_llm['significant'].sum()}/{len(rho_llm)} significant)",
    ])


def _sci(p: float) -> str:
    mant, exp = f"{p:.2e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def agreement_to_latex(df: pd.DataFrame) -> str:
    """Tabular of Table 1."""
    lines = [
        r"\begin{tabular}{llrrrrl}", r"\hline",
        r"\textbf{Target} & \textbf{Model} & \textbf{MAE} & \textbf{ME} & "
        r"\textbf{Wilcoxon p} & \textbf{Adjusted p} & \textbf{Direction} \\",
        r"\hline",
    ]
    for _, g in df.groupby("outcome", sort=False):
        for r in g.itertuples():
            lines.append(
                f"{OUTCOME_LABELS[r.outcome]} & {r.model} & {r.mae:.3f} & {r.me:.3f} & "
                f"{_sci(r.p_value)} & {_sci(r.p_adj)} & {r.direction} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def friedman_to_latex(friedman: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    """Tabular of Table 2 (model closer to humans in bold)."""
    lines = [
        r"\begin{tabular}{lccc}", r"\hline",
        r"\textbf{Domain} & \textbf{Friedman $\chi^2$} & \textbf{Adjusted p} & "
        r"\textbf{Significant pairwise comparisons} \\",
        r"\hline",
    ]
    for r in friedman.itertuples():
        sig = pairwise[(pairwise["outcome"] == r.outcome) & pairwise["significant"]]
        pairs = [
            " vs ".join(f"\\textbf{{{m}}}" if m == s.closer_to_humans else m
                        for m in (s.model_a, s.model_b))
            for s in sig.itertuples()
        ]
        p = "$<0.001$" if r.p_adj < 0.001 else f"{r.p_adj:.3f}"
        lines.append(f"{OUTCOME_LABELS[r.outcome]} & {r.chi2:.2f} & {p} & "
                     f"{', '.join(pairs) or 'None'} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
