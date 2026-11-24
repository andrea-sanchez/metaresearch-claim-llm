"""Regression tests: the pipeline must reproduce the numbers reported in the paper."""

import pandas as pd
import pytest

from src import analysis as A
from src import data as D

# Table 1 (MAE, ME), reports/paper/sections/3Results.tex
TABLE_1 = {
    ("study_description_domain", "ChatGPT"): (34.600, -34.400),
    ("study_description_domain", "Claude"): (25.383, -20.817),
    ("study_description_domain", "Gemini"): (29.083, -23.117),
    ("dataset_description_domain", "ChatGPT"): (39.700, -39.600),
    ("dataset_description_domain", "Claude"): (26.983, -23.917),
    ("dataset_description_domain", "Gemini"): (35.150, -28.383),
    ("model_description_domain", "ChatGPT"): (68.500, -67.833),
    ("model_description_domain", "Claude"): (61.900, -57.367),
    ("model_description_domain", "Gemini"): (42.483, -29.583),
    ("performance_domain", "ChatGPT"): (20.517, 3.983),
    ("performance_domain", "Claude"): (25.300, 13.467),
    ("performance_domain", "Gemini"): (22.200, 9.900),
    ("all_domains", "ChatGPT"): (33.133, -33.100),
    ("all_domains", "Claude"): (21.700, -18.633),
    ("all_domains", "Gemini"): (24.850, -19.317),
}
# Table 2 (Friedman chi2)
FRIEDMAN = {
    "study_description_domain": 16.58, "dataset_description_domain": 25.93,
    "model_description_domain": 20.79, "performance_domain": 5.93, "all_domains": 32.67,
}


@pytest.fixture(scope="module")
def scores():
    return D.prepare_scores(D.load_full_dataset())


@pytest.fixture(scope="module")
def results(scores):
    return A.run_analysis(scores)


def test_scores_table(scores):
    assert len(scores) == 60 * len(A.EVALUATORS) * len(A.OUTCOMES)
    assert scores["score"].notna().all()
    assert (scores["score"] % 1 == 0).all()  # whole-number percentages


def test_table1(results):
    agr = results["agreement"].set_index(["outcome", "model"])
    for key, (mae, me) in TABLE_1.items():
        assert round(agr.loc[key, "mae"], 3) == pytest.approx(mae), key
        assert round(agr.loc[key, "me"], 3) == pytest.approx(me), key
    assert agr["significant"].sum() == 14
    assert (agr["me"] < 0).sum() == 12
    assert not agr.loc[("performance_domain", "ChatGPT"), "significant"]


def test_table2_friedman(results):
    fr = results["friedman"].set_index("outcome")
    for outcome, chi2 in FRIEDMAN.items():
        assert round(fr.loc[outcome, "chi2"], 2) == pytest.approx(chi2), outcome
    assert not fr.loc["performance_domain", "significant"]
    assert fr["significant"].sum() == 4


def test_spearman(results):
    rho_h, rho_llm = results["spearman_llm_humans"], results["spearman_llm_llm"]
    assert not rho_h["significant"].any()
    assert round(rho_h["rho"].min(), 2) == -0.29
    assert round(rho_h["rho"].max(), 2) == 0.02
    assert rho_llm["significant"].sum() == 6
    assert round(rho_llm["rho"].max(), 2) == 0.65


def test_descriptives(results):
    d = results["descriptives"].set_index(["outcome", "evaluator"])
    assert d.loc[("model_description_domain", "Humans"), "mean"] == pytest.approx(87.0)
    assert d.loc[("dataset_description_domain", "Humans"), "mean"] == pytest.approx(61.7)


def test_ranking(results):
    assert A.ranking(results["agreement"])["model"].tolist() == ["Claude", "Gemini", "ChatGPT"]


def test_to_percent_half_up():
    assert D.to_percent(pd.Series([0.58, 0.125, 0.285])).tolist() == [58, 13, 29]


def test_spreadsheet_overrides(scores):
    def score(pid, ev):
        sel = ((scores["paper_id"] == pid) & (scores["evaluator"] == ev)
               & (scores["outcome"] == "model_description_domain"))
        return scores.loc[sel, "score"].item()
    assert score(57, "Gemini") == 0
    assert score(56, "Claude") == 0


def test_figures_are_written(scores, results, tmp_path):
    from src import figures as F
    F.make_all(scores, D.load_full_dataset(), results, tmp_path)
    for name in F.FIGURE_FILES.values():
        assert (tmp_path / name).stat().st_size > 0, name
