# Meta-research on LLM-assisted peer review: agreement and strictness in CLAIM audits

ChatGPT, Claude and Gemini assess 60 cardiac MRI segmentation papers with the
CLAIM 2020 checklist (42 items). Their domain and overall compliance scores are
compared with the published human assessment (Alabed et al., 2022) using MAE,
ME, Wilcoxon, Friedman and Spearman tests with Holm correction.

## Repository structure

```
.
├── data/
│   ├── raw/                  # versioned source data (see data/README.md)
│   │   ├── papers.csv
│   │   └── llm_evaluations.csv
│   └── processed/            # generated (not versioned)
├── reports/
│   ├── paper/                # LaTeX source (LLNCS); figures in paper/images/ are generated
│   └── tables/               # generated tables (.csv, .tex)
├── src/
│   ├── data.py               # load, validate, join, compute CLAIM scores
│   ├── analysis.py           # statistics (Tables 1–2, Spearman)
│   ├── figures.py            # figures of the paper
│   └── __main__.py           # pipeline entry point
├── tests/
├── requirements.txt
└── README.md
```

## Reproducing the paper

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src        # data → statistics → reports/tables/ + reports/paper/images/
pytest               # checks that the paper's numbers are reproduced
```

Compile the paper from `reports/paper/` (`main.tex`, LLNCS class, BibTeX `splncs04`).

## Outputs

| Paper element | Generated file |
|---|---|
| Fig. 1 — overall score per paper | `reports/paper/images/35ScoresResults2.png` |
| Fig. 2a — score distribution | `reports/paper/images/4ScoresDistribution2.png` |
| Fig. 2b — mean score per domain | `reports/paper/images/5ScoresDomain2.png` |
| Table 1 — MAE, ME, Wilcoxon | `reports/tables/agreement.csv`, `agreement.tex` |
| Table 2 — Friedman + pairwise | `reports/tables/friedman.csv`, `pairwise.csv`, `friedman.tex` |
| Fig. 3 — Spearman ρ | `reports/paper/images/2Spearmanrho.png`, `reports/tables/spearman_*.csv` |
| Appendix B — scores by metadata | `reports/paper/images/6MetadataresultsAll.png` |
| Descriptive statistics | `reports/tables/descriptives.csv` |

## Methods implemented

- Compliance score = #Yes / (#Yes + #No) × 100 per domain and overall ("Results");
  *Not applicable* excluded. All scores rounded to whole-number percentages (half up).
- Error = LLM − human (percentage points). MAE, ME.
- Two-sided Wilcoxon signed-rank (zero differences excluded), Holm over 15 tests.
- Friedman on absolute errors (Holm over 5 outcomes); pairwise Wilcoxon on absolute
  errors (Holm within outcome).
- Spearman ρ (average ranks for ties); Holm separately for LLM–human and LLM–LLM (15 + 15).

## Spreadsheet values vs. item-level scores

The analysis uses the domain scores of the original database. Two of them differ
from the item-level computation (`src/analysis.py::SPREADSHEET_OVERRIDES`):

| Paper | Evaluator | Domain | Original database | Item-level |
|---|---|---|---|---|
| 57 | Gemini | Model description | 0 (hard-coded) | 66.7 |
| 56 | Claude | Model description | 0 (`#DIV/0!`) | undefined (all items N/A) |

## Authors

Alicia Ramos, Andrea Sánchez, Mireia Brichs, Yolanda Romero @ Universitat Pompeu Fabra
