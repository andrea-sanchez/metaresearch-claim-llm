import numpy as np
import pandas as pd
import pytest

from src import data as D


@pytest.fixture(scope="module")
def raw():
    return D.load_papers(), D.load_llm_evaluations()


def test_validation_passes(raw):
    D.validate(*raw)


def test_ratings_are_canonical(raw):
    _, evals = raw
    values = set(pd.unique(evals[D.ITEM_COLS].values.ravel())) - {np.nan}
    assert {v for v in values if isinstance(v, str)} <= {D.YES, D.NO, D.NA}


def test_domains_partition_all_items():
    items = sorted(i for v in D.DOMAINS.values() for i in v)
    assert items == list(range(1, 43))


def test_score_formula():
    row = {c: D.NA for c in D.ITEM_COLS}
    row.update(q1=D.YES, q2=D.NO, q3=D.YES, q4=np.nan)   # study domain: 2/3
    scores = D.compute_domain_scores(pd.DataFrame([row]))
    assert scores.loc[0, "study_description_domain"] == pytest.approx(2 / 3)
    assert np.isnan(scores.loc[0, "model_description_domain"])  # all NA
    assert scores.loc[0, D.OVERALL_SCORE] == pytest.approx(2 / 3)


def test_full_dataset_shape(raw):
    full = D.build_full_dataset(*raw)
    n_papers = raw[1]["paper_id"].nunique()
    assert len(full) == n_papers * len(D.EVALUATOR_ORDER)
    assert not full.duplicated(["paper_id", "evaluator"]).any()
    assert full.loc[full.evaluator == D.HUMAN_EVALUATOR, D.ITEM_COLS].isna().all().all()
