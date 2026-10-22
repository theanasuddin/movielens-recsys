# MovieLens 1M recommender project (DATA.ML.360, Group 2)

Course project for DATA.ML.360 Recommender Systems at Tampere University.
We build, evaluate and critique recommenders on MovieLens 1M over three parts.

Team: Anas Uddin, Melek Chelik, Menaka Wellane Hewage.

| Part | Content | Status |
| --- | --- | --- |
| 1 | Data, protocol, baselines | in progress |
| 2 | Models and honest comparison | not started |
| 3 | Beyond accuracy (track to be announced with Part 2) | not started |

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

One command produces every number and figure in the report:

```bash
python run_all.py --data-dir data/ml-1m
```

The dataset path can point anywhere; nothing in the code depends on where the
files live. The command writes:

| Output | Content |
| --- | --- |
| `results/results.csv` | machine-readable results, one row per template row, with a `Sheet` column |
| `results/Recommender_Systems_Results_Group2.xlsm` | the course results template, filled in |
| `results/figures/` | figures used in the report |
| `results/tables/` | supporting tables (split sizes, tuning grids, sensitivity checks) |
| `results/run_info.json` | seed, resolved config, package versions, git commit, data checksums and timings |

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

## Repository layout

```text
run_all.py                 single entry point
configs/default.yaml       every setting used for the report
recsys/
  config.py                YAML loading and validation
  data/movielens.py        parsing the .dat files, optional download
  data/split.py            global temporal split
  data/dataset.py          id maps, sparse matrices, TrainData
  models/                  recommenders behind one interface (base.py)
  evaluation/metrics.py    P@k, R@k, NDCG@k, coverage, Gini
  evaluation/evaluator.py  full-catalogue top-k evaluation
  evaluation/stats.py      bootstrap intervals and paired tests
  reporting/               results.csv and workbook writer, template schema
  experiments/             one runner per project part
templates/                 the course results template (.xlsm)
tests/                     unit tests
data/                      MovieLens goes here (not committed)
results/                   generated outputs (committed)
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
