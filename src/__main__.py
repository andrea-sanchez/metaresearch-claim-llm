"""
Run the full pipeline: data → scores → statistics → tables → figures.

    python -m src
"""

from __future__ import annotations

import logging

from src import analysis as A
from src import data as D
from src import figures as F

log = logging.getLogger("src")

TABLES_DIR = D.ROOT / "reports" / "tables"
IMAGES_DIR = D.ROOT / "reports" / "paper" / "images"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for o in A.SPREADSHEET_OVERRIDES:
        log.info("Spreadsheet value: paper %d / %s / %s = %s (%s)",
                 o.paper_id, o.evaluator, o.outcome, o.value, o.reason)

    # 1. Data
    full = D.load_full_dataset()
    D.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(D.PROCESSED_DIR / "full_dataset.csv", index=False)
    D.build_reduced_dataset(full).to_csv(D.PROCESSED_DIR / "reduced_dataset.csv", index=False)

    scores = A.prepare_scores(full)
    scores.to_csv(D.PROCESSED_DIR / "scores.csv", index=False)

    # 2. Statistics and tables
    results = A.run_analysis(scores)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.6g")
    (TABLES_DIR / "agreement.tex").write_text(A.agreement_to_latex(results["agreement"]))
    (TABLES_DIR / "friedman.tex").write_text(
        A.friedman_to_latex(results["friedman"], results["pairwise"]))
    log.info("Tables -> %s", TABLES_DIR)

    # 3. Figures
    F.make_all(scores, full, results, IMAGES_DIR)
    log.info("Figures -> %s", IMAGES_DIR)


if __name__ == "__main__":
    main()
