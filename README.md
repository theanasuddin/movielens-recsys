# MovieLens 1M recommender project (DATA.ML.360, Group 2)

Course project for DATA.ML.360 Recommender Systems at Tampere University.
We build, evaluate and critique recommenders on MovieLens 1M over three parts.

Team: Anas Uddin, Melek Chelik, Menaka Wellane Hewage.

| Part | Content | Status |
| --- | --- | --- |
| 1 | Data, protocol, baselines | done |
| 2 | Models and honest comparison | not started |
| 3 | Beyond accuracy (track to be announced with Part 2) | not started |

## Part 1 submission

Everything handed in for Part 1 is in this repository:

| What | Where |
| --- | --- |
| Report (Part 1 version, 4 pages plus references) | [report/Group2_Recommender_Systems_Report.pdf](report/Group2_Recommender_Systems_Report.pdf), Word file: [report/Group2_Recommender_Systems_Report.docx](report/Group2_Recommender_Systems_Report.docx) |
| Results in the course results template | [results/Recommender_Systems_Results_Group2.xlsm](results/Recommender_Systems_Results_Group2.xlsm), sheet "Part 1 - Baselines" |
| The same results, machine-readable | [results/results.csv](results/results.csv) |
| Figures used in the report | [results/figures/](results/figures/) |
| Tables behind every number in the report | [results/tables/](results/tables/) |
| Code, and the command that reproduces all of the above | [run_all.py](run_all.py), see [Reproducing the results](#reproducing-the-results) |

The report is one document that grows with each part; the PDF is an export
of the .docx, and every number in it comes from the generated tables. The
workbook's "Part 1 - Baselines" sheet lists all 14 evaluated variants with
their test metrics, seed, and 95% interval in the Notes column.

Where each Part 1 requirement of the project description is answered:

| Requirement | Report | Code and outputs |
| --- | --- | --- |
| Data characterisation: counts, density, long tail, rating distribution, activity over time | Section 1.1, Figure 1 | `recsys/analysis/characterize.py`; `results/tables/data_summary.csv`, `long_tail.csv`, `burstiness.csv`, `monthly_activity.csv` |
| Temporal split, with where and why we cut | Section 1.3 | `recsys/data/split.py`, `configs/default.yaml`; `results/tables/split_summary.csv`, `held_out_profiles.csv` |
| Feedback treatment and filtering | Section 1.2 | `recsys/data/dataset.py`; `results/tables/part1_filter_sensitivity.csv` |
| Item-based kNN with tuned k and similarity | Section 1.5, Figure 2 | `recsys/models/item_knn.py`; `results/tables/part1_validation_itemknn.csv` |
| Most popular, eight variants | Section 1.5, Table 1 | `recsys/models/popularity.py`; `results/tables/part1_validation_popularity.csv`, `popularity_window_sizes.csv` |
| Precision@10, Recall@10, NDCG@10 and coverage, ranked against the full catalogue | Sections 1.4 and 1.6, Table 1 | `recsys/evaluation/`; `tests/test_metrics.py`; `results/tables/part1_paired_tests.csv` |
| Justification of split, feedback, filtering, candidate set and k | Sections 1.2 to 1.5 | |
| Effect of filtering thresholds and of the split choice | Section 1.6 | `results/tables/part1_filter_sensitivity.csv`, `part1_split_sensitivity.csv` |

## Setup

The experiments use Python 3.10 (we ran them with 3.10.11). Create a virtual
environment and install the pinned dependencies:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on macOS and Linux, then:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the packages we import directly.
`requirements-lock.txt` pins the full environment, indirect dependencies
included, if you want an exact copy:

```bash
python -m pip install -r requirements-lock.txt
```

## Data

MovieLens 1M cannot be redistributed, so it is not in the repository.
Download `ml-1m.zip` from <https://grouplens.org/datasets/movielens/1m/> and
unzip it into `data/` (details in [data/README.md](data/README.md)).
Alternatively, add `--download` to the command below and the script fetches
the zip and checks its MD5 sum.

## Reproducing the results

One command produces every number, table and figure in the report:

```bash
python run_all.py --data-dir data/ml-1m
```

The dataset path can point anywhere; nothing in the code depends on where the
files live. A full run takes about two and a half minutes on a laptop and
writes:

| Output | Content |
| --- | --- |
| `results/results.csv` | machine-readable results, one row per template row, with a `Sheet` column |
| `results/Recommender_Systems_Results_Group2.xlsm` | the course results template, filled in |
| `results/figures/` | `fig1_data_overview.png` and `fig2_itemknn_validation.png` (Figures 1 and 2 of the report), plus `rating_distribution.png` |
| `results/tables/` | everything behind the numbers quoted in the report (see below) |
| `results/run_info.json` | seed, resolved config, package versions, git commit, data checksums and timings |

The tables in `results/tables/` for Part 1:

| File | Content |
| --- | --- |
| `data_summary.csv`, `long_tail.csv`, `monthly_activity.csv`, `burstiness.csv` | data characterisation |
| `split_summary.csv`, `held_out_profiles.csv` | size of each part, and who is in the validation and test periods, for our split and for an 80/10/10 count split |
| `popularity_drift_top10.csv`, `popularity_drift_overlap.csv`, `popularity_window_sizes.csv` | how popularity changes between periods |
| `part1_validation_itemknn.csv`, `part1_validation_popularity.csv` | the full tuning grids on validation |
| `part1_paired_tests.csv`, `part1_test_per_user.csv` | paired bootstrap intervals and Wilcoxon tests, per-user test metrics |
| `part1_filter_sensitivity.csv`, `part1_split_sensitivity.csv` | the filtering and split checks |

Other options:

| Option | Meaning |
| --- | --- |
| `--config PATH` | another YAML config (default `configs/default.yaml`) |
| `--parts 1 2` | run only some parts; results.csv then holds only those rows |
| `--output-dir PATH` | write somewhere other than `results/` |
| `--download` | download MovieLens 1M first if it is missing |
| `--quick` | small search grids for a fast check; writes to `results-quick/`, not used in the report |

## Random seeds

All randomness comes from one seed, `seed: 42` in `configs/default.yaml`. It
drives the random baseline, the order used to break tied scores when ranking,
and the bootstrap resampling of confidence intervals. Models with a random
component (from Part 2 on) receive the same seed unless their config sets one.

## Evaluation protocol

These are the current defaults in `configs/default.yaml`. The report gives the
reasoning for each choice and the checks behind it.

- Split: one global temporal split. Ratings entered before 2001-01-01 train,
  ratings from 2001-01-01 to 2001-06-30 validate, ratings from 2001-07-01 test.
  Ratings with the same timestamp always fall in the same part.
- Tuning: hyperparameters are chosen on validation only. The chosen setting is
  then refitted on train plus validation and evaluated once on test.
- Relevance: a held-out rating of 4 or 5 counts as relevant.
- Candidates: every movie with at least one rating before the cut, minus the
  movies the user has already rated. There is no negative sampling.
- Evaluated users: users with at least one relevant held-out rating and at
  least one training rating.
- Metrics: Precision@10, Recall@10 and NDCG@10, computed per user and averaged,
  plus catalogue coverage. Intervals are 95% bootstrap intervals over users.

## Part 1 baselines

- Most popular: ranks movies by their number of training ratings. Eight
  variants: all training ratings or only the last 30, 90 or 180 days before
  the cut, counting every rating or only ratings of 4 and 5.
- Item-based kNN (`recsys/models/item_knn.py`): keeps the k most similar
  movies of each movie and scores a candidate by summing its similarities to
  the neighbours the user has rated. Tuned over similarity (cosine, Jaccard,
  adjusted cosine), training signal (all ratings or only 4 and 5) and
  k in {10, 20, 50, 100, 200, 500, 1000}.

Both are tuned by NDCG@10 on validation; the grids are in
`configs/default.yaml` under `part1`.

## Repository layout

```text
run_all.py                 single entry point
configs/default.yaml       every setting used for the report
recsys/
  config.py                YAML loading and validation
  data/movielens.py        parsing the .dat files, optional download
  data/split.py            global temporal split
  data/dataset.py          id maps, sparse matrices, TrainData
  analysis/characterize.py data statistics for report section 1.1
  models/                  recommenders behind one interface (base.py):
                           popularity.py (most popular, random), item_knn.py
  evaluation/metrics.py    P@k, R@k, NDCG@k, coverage, Gini
  evaluation/evaluator.py  full-catalogue top-k evaluation
  evaluation/stats.py      bootstrap intervals and paired tests
  reporting/               results.csv and workbook writer, template schema, figures
  experiments/             one runner per project part (part1.py)
templates/                 the course results template (.xlsm)
tests/                     unit tests
data/                      MovieLens goes here (not committed)
results/                   generated outputs (committed)
report/                    the course report (.docx and a PDF export)
```

## Tests

```bash
python -m pytest
```

The metric tests reproduce the worked example from Lecture 3.1, slide 8, and
check our NDCG@10 against `sklearn.metrics.ndcg_score` on 500 random rankings.
The course asks for this kind of check when a team writes its own metrics.

## Adding a model

Subclass `Recommender` in `recsys/models/`, implement `_fit` and `score`, and
register the class in `MODELS` in `recsys/models/__init__.py`. Set
`stochastic = True` if the model uses randomness so it receives the project
seed. The evaluator handles seen items, the candidate set and the metrics.

## Dataset citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article
19. <https://doi.org/10.1145/2827872>
