# Meta-research on LLM-assisted peer review: agreement and strictness in CLAIM audits

Code and data for the paper *Meta-Research on LLM-assisted peer review: Evaluating
agreement and strictness in CLAIM audits* (Research Methodology, MSc Computational
Biomedical Engineering, Universitat Pompeu Fabra, 2026).

ChatGPT (GPT-6), Claude (Sonnet 5.5) and Gemini (3.8 Flash) assessed 60 AI cardiac MRI
segmentation papers with the CLAIM 2020 checklist (42 items). Their domain and overall
compliance scores are compared with the published human assessment of
[Alabed et al. (2022)](https://doi.org/10.3389/fcvm.2022.956811) using MAE, ME,
Wilcoxon, Friedman and Spearman tests with Holm correction.

## Repository structure

```
.
├── data/
│   ├── raw/                     # versioned source data (see data/README.md)
│   │   ├── papers.csv           # metadata + human CLAIM scores (Alabed et al., 2022)
│   │   └── llm_evaluations.csv  # CLAIM item ratings of the three LLMs
│   └── processed/               # generated (not versioned)
├── reports/
│   ├── paper/                   # LaTeX source of the paper (LLNCS); images/ is generated
│   └── tables/                  # generated tables (.csv, .tex)
├── src/
│   ├── data.py                  # load, validate, join, compute CLAIM scores
│   ├── analysis.py              # statistics (Tables 1–2, Fig. 3)
│   ├── figures.py               # figures of the paper
│   └── __main__.py              # pipeline entry point
├── tests/                       # checks that the paper's numbers are reproduced
├── requirements.txt
└── README.md
```

## Reproducing the paper

Requires Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src        # raw data → processed data → tables → figures
pytest               # checks the numbers reported in the paper
```

The pipeline is deterministic (fixed seed for point jitter in the figures): running it
again produces the same tables and figures. Compile the paper from `reports/paper/`
(`main.tex`, LLNCS class, BibTeX style `splncs04`).

## Pipeline

| Step | Module | Output |
|---|---|---|
| 1. Load and validate raw data, compute CLAIM scores from item ratings | `src/data.py` | `data/processed/full_dataset.csv`, `reduced_dataset.csv`, `scores.csv` |
| 2. Statistics | `src/analysis.py` | `reports/tables/*.csv`, `*.tex` |
| 3. Figures | `src/figures.py` | `reports/paper/images/*.png` |

## Outputs

| Paper element | Generated file |
|---|---|
| Fig. 1 — overall score per paper | `reports/paper/images/fig1_overall_scores_per_paper.png` |
| Fig. 2a — distribution of overall scores | `reports/paper/images/fig2a_overall_score_distribution.png` |
| Fig. 2b — mean score per domain | `reports/paper/images/fig2b_mean_score_by_domain.png` |
| Table 1 — MAE, ME, Wilcoxon | `reports/tables/agreement.csv`, `agreement.tex` |
| Table 2 — Friedman + pairwise comparisons | `reports/tables/friedman.csv`, `pairwise.csv`, `friedman.tex` |
| Fig. 3 — Spearman ρ | `reports/paper/images/fig3_spearman_heatmaps.png`, `reports/tables/spearman_*.csv` |
| Appendix C — scores by study metadata | `reports/paper/images/figC1_scores_by_metadata.png` |
| Means and quartiles quoted in Section 3 | `reports/tables/descriptives.csv` |

## Methods implemented

- Compliance score = #Yes / (#Yes + #No) × 100 per domain and overall; *Not applicable*
  and missing ratings excluded; scores rounded to whole-number percentages (half up),
  as in the original spreadsheet.
- Error = LLM − human (percentage points); MAE (agreement) and ME (strictness).
- Two-sided Wilcoxon signed-rank test on the differences (zero differences excluded),
  Holm over the 15 tests.
- Friedman test on absolute errors (Holm over the 5 outcomes); pairwise Wilcoxon on
  absolute errors (Holm within each outcome).
- Spearman ρ (average ranks for ties); Holm separately for LLM–human and LLM–LLM (15 + 15).

## Values taken from the original database

Two domain scores are used as recorded in the original database rather than recomputed
from the items (`src/data.py::SPREADSHEET_OVERRIDES`):

| Paper | Evaluator | Domain | Value used | Reason |
|---|---|---|---|---|
| 56 | Claude | Model description | 0 | all items *Not applicable* (`#DIV/0!` in the spreadsheet) |
| 57 | Gemini | Model description | 0 | value recorded in the original database |

## Data and licence

The human CLAIM scores and paper metadata come from the supplementary material of
Alabed et al. (2022), *Frontiers in Cardiovascular Medicine* 9:956811. The LLM ratings
were collected by the authors on 6–9 November 2026; the prompt is in Appendix B of the
paper (`reports/paper/sections/6Appendix.tex`). Code released under the MIT licence
(`LICENSE`).

## Authors

Alicia Ramos, Andrea Sánchez, Mireia Brichs, Yolanda Romero — Universitat Pompeu Fabra
