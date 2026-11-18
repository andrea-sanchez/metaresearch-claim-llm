"""
Statistical analysis: LLM vs. human CLAIM compliance scores.

Reproduces the tables and statistics of ``reports/paper``:

    descriptives   mean / quartiles per evaluator and outcome
    agreement      MAE, ME, Wilcoxon signed-rank vs. humans, Holm (15 tests)   → Table 1
    friedman       Friedman on absolute errors, Holm (5 tests)                 → Table 2
    pairwise       Wilcoxon on paired absolute errors, Holm within outcome     → Table 2
    spearman       ρ LLM–human and LLM–LLM, Holm per family (15 + 15)          → Figure 3

Score preparation (Methodology, §2):
    - compliance score = #Yes / (#Yes + #No) × 100
    - every score rounded to a whole-number percentage (half up)
    - error = LLM − human, in percentage points
    - domain scores taken as in the source spreadsheet (SPREADSHEET_OVERRIDES)

NOTE — PLACEHOLDER IMPLEMENTATION
    Reconstructed from the paper and validated against its reported numbers.
    To be replaced by the original analysis code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

from src import data as D

LLMS = ["ChatGPT", "Claude", "Gemini"]  # order used in the paper's tables
HUMANS = D.HUMAN_EVALUATOR
OUTCOMES = D.SCORE_COLS
OUTCOME_LABELS = {
    "study_description_domain": "Study description",
    "dataset_description_domain": "Dataset description",
    "model_description_domain": "Model description",
    "performance_domain": "Performance",
    "all_domains": "Results",  # overall score
}
ALPHA = 0.05


@dataclass(frozen=True)
class Override:
    paper_id: int
    evaluator: str
    outcome: str
    value: float
    reason: str


# Domain scores taken as recorded in the original database (source spreadsheet).
# Always applied.
SPREADSHEET_OVERRIDES = [
    Override(57, "Gemini", "model_description_domain", 0.0,
             "value recorded in the original database"),
    Override(56, "Claude", "model_description_domain", 0.0,
             "all items 'Not applicable' (#DIV/0! in the spreadsheet) scored as 0"),
]


# ---------------------------------------------------------------------------
# Score preparation
# ---------------------------------------------------------------------------


def to_percent(x: pd.Series | pd.DataFrame):
    """Fraction → whole-number percentage, rounding half up (as Excel ROUND).

    The intermediate ``round(…, 6)`` removes floating-point noise
    (e.g. 0.58 × 100 = 57.999…) before rounding.
    """
    return np.floor(np.round(x * 100, 6) + 0.5)


def prepare_scores(full: pd.DataFrame) -> pd.DataFrame:
    """Long table: paper_id, evaluator, outcome, score (integer %),
    with SPREADSHEET_OVERRIDES applied."""
    full = full.copy()
    for o in SPREADSHEET_OVERRIDES:
        mask = (full["paper_id"] == o.paper_id) & (full["evaluator"] == o.evaluator)
        assert mask.sum() == 1, o
        full.loc[mask, o.outcome] = o.value
    long = full.melt(
        id_vars=["paper_id", "evaluator"], value_vars=OUTCOMES,
        var_name="outcome", value_name="score",
    )
    long["score"] = to_percent(long["score"])
    return long


def wide(scores: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """papers × evaluators matrix for one outcome."""
    w = scores[scores["outcome"] == outcome].pivot(
        index="paper_id", columns="evaluator", values="score"
    )
    return w[[HUMANS, *LLMS]]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _paired(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    ok = a.notna() & b.notna()
    return a[ok], b[ok]


def _wilcoxon(a: pd.Series, b: pd.Series) -> float:
    """Two-sided Wilcoxon signed-rank; zero differences excluded ('wilcox')."""
    a, b = _paired(a, b)
    return wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def descriptives(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and quartiles per evaluator and outcome."""
    g = scores.groupby(["outcome", "evaluator"])["score"]
    out = pd.DataFrame({
        "n": g.count(), "mean": g.mean(), "sd": g.std(),
        "q1": g.quantile(0.25), "median": g.median(), "q3": g.quantile(0.75),
        "min": g.min(), "max": g.max(),
    }).reset_index()
    return _order(out)


def agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """MAE, ME and Wilcoxon (LLM vs. humans); Holm over all 15 tests."""
    rows = []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        for model in LLMS:
            llm, hum = _paired(w[model], w[HUMANS])
            err = llm - hum
            rows.append({
                "outcome": outcome, "model": model, "n": len(err),
                "mae": err.abs().mean(), "me": err.mean(),
                "p_value": _wilcoxon(llm, hum),
            })
    df = pd.DataFrame(rows)
    df["p_adj"] = holm(df["p_value"])
    df["direction"] = np.where(df["me"] < 0, "Stricter", "More lenient")
    df["significant"] = df["p_adj"] < ALPHA
    return df


def model_differences(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test on absolute errors (Holm over 5 outcomes) and pairwise
    Wilcoxon on absolute errors (Holm within each outcome)."""
    fr_rows, pw_rows = [], []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        abs_err = w[LLMS].sub(w[HUMANS], axis=0).abs().dropna()
        stat, p = friedmanchisquare(*(abs_err[m] for m in LLMS))
        fr_rows.append({"outcome": outcome, "n": len(abs_err), "chi2": stat, "p_value": p})

        pairs = list(itertools.combinations(LLMS, 2))
        pvals = [_wilcoxon(abs_err[a], abs_err[b]) for a, b in pairs]
        for (a, b), p_raw, p_adj in zip(pairs, pvals, holm(pvals)):
            closer = a if abs_err[a].mean() < abs_err[b].mean() else b
            pw_rows.append({
                "outcome": outcome, "model_a": a, "model_b": b, "n": len(abs_err),
                "mae_a": abs_err[a].mean(), "mae_b": abs_err[b].mean(),
                "p_value": p_raw, "p_adj": p_adj, "significant": p_adj < ALPHA,
                "closer_to_humans": closer,
            })
    friedman = pd.DataFrame(fr_rows)
    friedman["p_adj"] = holm(friedman["p_value"])
    friedman["significant"] = friedman["p_adj"] < ALPHA
    return friedman, pd.DataFrame(pw_rows)


def spearman(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ρ LLM–humans and LLM–LLM; Holm applied separately to each family."""
    def corr(a, b):
        a, b = _paired(a, b)
        r = spearmanr(a, b)
        return r.statistic, r.pvalue

    vs_h, vs_llm = [], []
    for outcome in OUTCOMES:
        w = wide(scores, outcome)
        for m in LLMS:
            rho, p = corr(w[m], w[HUMANS])
            vs_h.append({"outcome": outcome, "pair": f"{m} vs Humans", "rho": rho, "p_value": p})
        for a, b in itertools.combinations(LLMS, 2):
            rho, p = corr(w[a], w[b])
            vs_llm.append({"outcome": outcome, "pair": f"{a} vs {b}", "rho": rho, "p_value": p})
    out = []
    for rows in (vs_h, vs_llm):
        df = pd.DataFrame(rows)
        df["p_adj"] = holm(df["p_value"])
        df["significant"] = df["p_adj"] < ALPHA
        out.append(df)
    return out[0], out[1]


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df["outcome"] = pd.Categorical(df["outcome"], OUTCOMES, ordered=True)
    return df.sort_values(list(df.columns[:2])).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LaTeX export (same layout as the paper)
# ---------------------------------------------------------------------------


def _sci(p: float) -> str:
    mant, exp = f"{p:.2e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def agreement_to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrl}", r"\hline",
        r"\textbf{Target} & \textbf{Model} & \textbf{MAE} & \textbf{ME} & "
        r"\textbf{Wilcoxon p} & \textbf{Adjusted p} & \textbf{Direction} \\",
        r"\hline",
    ]
    n_max = df["n"].max()
    for outcome, g in df.groupby("outcome", sort=False):
        for _, r in g.iterrows():
            model = r.model + ("$^{a}$" if r.n < n_max else "")  # reduced n, see note
            lines.append(
                f"{OUTCOME_LABELS[outcome]} & {model} & {r.mae:.3f} & {r.me:.3f} & "
                f"{_sci(r.p_value)} & {_sci(r.p_adj)} & {r.direction} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def friedman_to_latex(friedman: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lccc}", r"\hline",
        r"\textbf{Domain} & \textbf{Friedman $\chi^2$} & \textbf{Adjusted p} & "
        r"\textbf{Significant pairwise comparisons} \\",
        r"\hline",
    ]
    for _, r in friedman.iterrows():
        sig = pairwise[(pairwise.outcome == r.outcome) & pairwise.significant]
        pairs = [
            " vs ".join(f"\\textbf{{{m}}}" if m == s.closer_to_humans else m
                        for m in (s.model_a, s.model_b))
            for _, s in sig.iterrows()
        ]
        p = "$<0.001$" if r.p_adj < 0.001 else f"{r.p_adj:.3f}"
        lines.append(
            f"{OUTCOME_LABELS[r.outcome]} & {r.chi2:.2f} & {p} & "
            f"{', '.join(pairs) or 'None'} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    friedman, pairwise = model_differences(scores)
    rho_h, rho_llm = spearman(scores)
    return {
        "descriptives": descriptives(scores),
        "agreement": agreement(scores),
        "friedman": friedman,
        "pairwise": pairwise,
        "spearman_llm_humans": rho_h,
        "spearman_llm_llm": rho_llm,
    }
