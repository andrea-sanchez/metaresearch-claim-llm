"""
Run the full pipeline: raw data -> processed tables -> statistics -> tables -> figures.

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

    # 1. Data
    full, scores = D.write_processed()

    # 2. Statistics and tables
    results = A.run_analysis(scores)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.6g")
    (TABLES_DIR / "agreement.tex").write_text(A.agreement_to_latex(results["agreement"]))
    (TABLES_DIR / "friedman.tex").write_text(
        A.friedman_to_latex(results["friedman"], results["pairwise"]))
    log.info("Tables -> %s", TABLES_DIR)
    log.info("Summary\n%s", A.summary(results))

    # 3. Figures
    F.make_all(scores, full, results, IMAGES_DIR)
    log.info("Figures -> %s", IMAGES_DIR)


if __name__ == "__main__":
    main()
