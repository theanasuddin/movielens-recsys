# Data

MovieLens 1M is not stored in this repository because the GroupLens licence
does not allow redistribution.

1. Download `ml-1m.zip` from https://grouplens.org/datasets/movielens/1m/
   (MD5 of the zip: `c4d9eecfca2ab87c1945afe126590906`).
2. Unzip it here, so that this folder contains `ml-1m/ratings.dat`,
   `ml-1m/movies.dat` and `ml-1m/users.dat`.

`run_all.py` reads `data/ml-1m` by default. If the files are somewhere else,
pass that folder with `--data-dir`. Adding `--download` fetches and checks the
zip automatically when the files are missing.
