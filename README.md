# MovieLens 1M recommender project (DATA.ML.360, Group 2)

Course project for DATA.ML.360 Recommender Systems at Tampere University.
We build, evaluate and critique recommenders on MovieLens 1M over three parts.

Team: Anas Uddin, Melek Chelik, Menaka Wellane Hewage.

| Part | Content | Status |
| --- | --- | --- |
| 1 | Data, protocol, baselines | done |
| 2 | Models and honest comparison | done |
| 3 | Beyond accuracy, Track A (diversity and popularity bias) | not started |

## What to grade

Everything handed in is in this repository:

| What | Where |
| --- | --- |
| Report (Part 2 version, 8 pages plus references) | [report/Group2_Recommender_Systems_Report.pdf](report/Group2_Recommender_Systems_Report.pdf), Word file: [report/Group2_Recommender_Systems_Report.docx](report/Group2_Recommender_Systems_Report.docx) |
| Results in the course results template | [results/Recommender_Systems_Results_Group2.xlsm](results/Recommender_Systems_Results_Group2.xlsm), sheets "Part 1 - Baselines", "Part 2 - Models", "Part 2 - Disaggregated" and "Part 2 - Sparsity" |
| The same results, machine-readable | [results/results.csv](results/results.csv) |
| Figures used in the report | [results/figures/](results/figures/) |
| Tables behind every number in the report | [results/tables/](results/tables/) |
| Code, and the command that reproduces all of the above | [run_all.py](run_all.py), see [Reproducing the results](#reproducing-the-results) |

The report is one document that grows with each part; the PDF is an export
of the .docx, and every number in it comes from the generated tables. The one
exception is the result of our first sparsity protocol quoted in Section 2.5,
which is in the history at commit `ec678ea`
(`git show ec678ea:results/tables/part2_sparsity_ranking.csv`).
In the workbook, the Notes and "Variability / CI / test" columns hold the
intervals, seeds and paired tests behind each row.

Where each Part 2 requirement of the project description is answered:

| Requirement | Report | Code and outputs |
| --- | --- | --- |
| At least three new models from at least two paradigms: user-kNN, biased MF, BPR-MF, EASE, content profile | Section 2.1, Table 2 | `recsys/models/user_knn.py`, `matrix_factorization.py`, `bpr.py`, `ease.py`, `content_based.py` |
| A self-implemented algorithm, clearly identified: BPR-MF and biased MF | Section 2.1 | `recsys/models/bpr.py`, `recsys/models/matrix_factorization.py`; gradient and planted-structure checks in `tests/test_part2_models.py` |
| Tuning on validation only, with search space and budget, baselines tuned the same way | Section 2.2 | `configs/default.yaml` (`part2`), `recsys/tuning.py`; `results/tables/part2_validation_trials.csv`, `part2_selected.csv`, `part2_training_curves.csv` |
| Variability: seeds, intervals and paired tests | Sections 2.2 and 2.3, Figure 3 | `results/tables/part2_test_by_seed.csv`, `part2_paired_tests.csv` |
| Disaggregated results: user activity, item popularity, early vs late test period | Section 2.4, Table 3, Figure 4 | `results/tables/part2_breakdowns.csv`, `part2_breakdown_differences.csv`, `part2_recommendation_profile.csv`, `part2_item_groups.csv` |
| Sparsity ablation at 100%, 50% and 20% of the training interactions | Section 2.5, Figure 5 | `results/tables/part2_sparsity_summary.csv`, `part2_sparsity_detail.csv`, `part2_sparsity_top_two.csv`, `part2_sparsity_frozen_settings.csv` |
| Part 3 track announced | header table, Section 3.1 | `team.part3_track` in `configs/default.yaml` |
| The question Part 1 left open: a popularity window of a fixed number of ratings | Section 2.3 | `results/tables/part2_popularity_windows.csv` |

Where each Part 1 requirement is answered:

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
files live. A full run takes about 75 minutes on a laptop, almost all of it
Part 2 (`--parts 1` runs Part 1 alone in about two and a half minutes), and
writes:

| Output | Content |
| --- | --- |
| `results/results.csv` | machine-readable results, one row per template row, with a `Sheet` column |
| `results/Recommender_Systems_Results_Group2.xlsm` | the course results template, filled in |
| `results/figures/` | `fig1_data_overview.png`, `fig2_itemknn_validation.png`, `fig3_part2_models.png`, `fig4_part2_breakdowns.png` and `fig5_part2_sparsity.png` (Figures 1 to 5 of the report), plus `rating_distribution.png` |
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

And for Part 2:

| File | Content |
| --- | --- |
| `part2_validation_trials.csv`, `part2_training_curves.csv`, `part2_selected.csv` | every validation trial, the epoch curves of BiasedMF and BPR-MF, and the chosen settings |
| `part2_test_by_seed.csv`, `part2_test_per_user.csv` | test metrics per seed and per user |
| `part2_popularity_windows.csv` | every most-popular variant at both cuts: how many ratings and days its window holds, its validation NDCG@10 and, as a check only, its test NDCG@10 |
| `part2_paired_tests.csv` | the best validation model against every other model: bootstrap intervals, Wilcoxon and Holm-adjusted p-values |
| `part2_breakdowns.csv`, `part2_breakdown_differences.csv` | metrics per user-activity, item-popularity and test-period group, with intervals and paired differences |
| `part2_recommendation_profile.csv`, `part2_item_groups.csv` | how popular the recommended movies are, how much of each list goes to tail movies and new releases, and how the relevant test ratings split between head and tail |
| `part2_biasedmf_rating_error.csv` | BiasedMF's rating error against simple predictors |
| `part2_sparsity_*.csv` | the 50% and 20% runs: every sample, means and ranks, a paired test of the top two, and the settings re-tuning chose |

Other options:

| Option | Meaning |
| --- | --- |
| `--config PATH` | another YAML config (default `configs/default.yaml`) |
| `--parts 1 2` | run only some parts; results.csv then holds only those rows |
| `--output-dir PATH` | write somewhere other than `results/` |
| `--download` | download MovieLens 1M first if it is missing |
| `--quick` | small search grids for a fast check; writes to `results-quick/`, not used in the report |

## Random seeds

All randomness comes from seeds in `configs/default.yaml`. `seed: 42` drives
the random baseline, the order used to break tied scores when ranking, the
bootstrap resampling of confidence intervals, the random search of Part 2
and model initialisation during tuning. The two stochastic Part 2 models
(BiasedMF and BPR-MF) are refitted for the test run with each of
`part2.test_seeds` (42 to 46), and the sparsity ablation draws its training
samples with `part2.sparsity.sampling_seeds` (42 to 44).

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

## Part 2 models

All models are our own numpy code; no recommender library is used.

- User-based kNN (`user_knn.py`): each user's k most similar users vote for
  the movies they rated.
- Biased matrix factorisation (`matrix_factorization.py`, self-implemented):
  mu + b_u + b_i + p_u . q_i fitted by mini-batch SGD on the observed ratings.
- BPR-MF (`bpr.py`, self-implemented): the same kind of factor model trained
  with the pairwise BPR loss on sampled (user, rated movie, other movie) triples.
- EASE (`ease.py`): a linear item-item model with a closed-form solution.
- Content profile (`content_based.py`): matches a user's genre and release
  decade profile to each movie.

Every model, the Part 1 baselines included, is tuned on validation with the
search space in `configs/default.yaml` under `part2`: whole grids for the
models with a few discrete settings, 24 seeded random draws for the two
gradient-trained models, which also pick their number of epochs on
validation. Most popular gets eight more variants in Part 2: counting only
the 10,000 to 300,000 most recent ratings, a window that holds the same
amount of data at the validation and the test cut, which a window in days
does not (Part 1, Section 1.6). The sparsity ablation trains on 50% and 20%
of each user's ratings but keeps the full history out of the recommendation
lists, so the task stays the same as at 100%.

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
                           popularity.py (most popular, random), item_knn.py,
                           user_knn.py, matrix_factorization.py, bpr.py,
                           ease.py, content_based.py
  tuning.py                grid and random search, epoch selection on validation
  evaluation/metrics.py    P@k, R@k, NDCG@k, coverage, Gini
  evaluation/evaluator.py  full-catalogue top-k evaluation
  evaluation/stats.py      bootstrap intervals, paired tests, Holm correction
  reporting/               results.csv and workbook writer, template schema, figures
  experiments/             one runner per project part (part1.py, part2.py)
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
The Part 2 model tests compare the gradients of BiasedMF and BPR-MF with
finite differences, check that both learn planted structure, check EASE
against per-column ridge regression and user-kNN against brute-force voting,
and reproduce the content profile example of Lecture 2.2, slide 6.
`tests/test_part2.py` runs the whole Part 2 pipeline on a small synthetic
dataset.

## Adding a model

Subclass `Recommender` in `recsys/models/`, implement `_fit` and `score`, and
register the class in `MODELS` in `recsys/models/__init__.py`. Set
`stochastic = True` if the model uses randomness so it receives the project
seed. The evaluator handles seen items, the candidate set and the metrics.

## Dataset citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article
19. <https://doi.org/10.1145/2827872>
