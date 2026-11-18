"""Regression tests: the pipeline must reproduce the numbers reported in the paper."""

import numpy as np
import pandas as pd
import pytest

from src import analysis as A
from src import data as D

# Table 1 (MAE, ME) as reported in reports/paper/sections/3Results.tex
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
    return A.prepare_scores(D.load_full_dataset())


@pytest.fixture(scope="module")
def results(scores):
    return A.run_analysis(scores)


def test_table1(results):
    df = results["agreement"].set_index(["outcome", "model"])
    for key, (mae, me) in TABLE_1.items():
        assert round(df.loc[key, "mae"], 3) == pytest.approx(mae), key
        assert round(df.loc[key, "me"], 3) == pytest.approx(me), key
    assert df["significant"].sum() == 14


def test_friedman(results):
    df = results["friedman"].set_index("outcome")
    for outcome, chi2 in FRIEDMAN.items():
        assert round(df.loc[outcome, "chi2"], 2) == pytest.approx(chi2)
    assert not df.loc["performance_domain", "significant"]


def test_spearman_significance(results):
    assert not results["spearman_llm_humans"]["significant"].any()
    assert results["spearman_llm_llm"]["significant"].sum() == 6


def test_holm():
    adj = A.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_to_percent_half_up():
    s = pd.Series([0.58, 0.125, 0.285, np.nan])
    assert A.to_percent(s).tolist()[:3] == [58, 13, 29]


# --- Original database values ---------------------------------------------


def test_spreadsheet_values(scores):
    def model_score(pid, ev):
        return scores.query(
            "paper_id == @pid and evaluator == @ev and outcome == 'model_description_domain'"
        )["score"].item()
    assert model_score(57, "Gemini") == 0
    assert model_score(56, "Claude") == 0
    assert scores["score"].notna().all()
