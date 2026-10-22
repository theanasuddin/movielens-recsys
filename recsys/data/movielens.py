"""Reading the MovieLens 1M files.

The three files use "::" as the field separator and have no header row.
movies.dat is Latin-1 encoded (some titles contain accented characters).
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_MD5 = "c4d9eecfca2ab87c1945afe126590906"
FILES = ("ratings.dat", "movies.dat", "users.dat")
EXPECTED_ROWS = {"ratings.dat": 1_000_209, "movies.dat": 3_883, "users.dat": 6_040}

GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)

_YEAR = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class MovieLens:
    """The three MovieLens 1M tables.

    ratings: user_id, item_id, rating (1-5), timestamp (Unix seconds, UTC)
    movies:  item_id, title, year, genres ("Action|Comedy")
    users:   user_id, gender, age, occupation, zip
    """

    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame


def _read_dat(path: Path, names: list[str], encoding: str) -> pd.DataFrame:
    # Swapping "::" for a tab lets pandas use its fast C parser. Titles never
    # contain tabs, and QUOTE_NONE keeps quotes inside titles as plain text.
    text = path.read_text(encoding=encoding).replace("::", "\t")
    return pd.read_csv(
        io.StringIO(text), sep="\t", header=None, names=names,
        quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
    )


def load_ratings(data_dir: Path) -> pd.DataFrame:
    df = _read_dat(data_dir / "ratings.dat", ["user_id", "item_id", "rating", "timestamp"], "latin-1")
    return df.astype({"user_id": "int32", "item_id": "int32", "rating": "int8", "timestamp": "int64"})


def load_movies(data_dir: Path) -> pd.DataFrame:
    df = _read_dat(data_dir / "movies.dat", ["item_id", "title", "genres"], "latin-1")
    df["item_id"] = df["item_id"].astype("int32")
    df["year"] = df["title"].str.extract(_YEAR, expand=False).astype("Int16")
    return df[["item_id", "title", "year", "genres"]]


def load_users(data_dir: Path) -> pd.DataFrame:
    df = _read_dat(data_dir / "users.dat", ["user_id", "gender", "age", "occupation", "zip"], "latin-1")
    return df.astype({"user_id": "int32", "age": "int8", "occupation": "int8"})


def genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot genre indicators, one row per movie, indexed by item_id."""
    hot = movies["genres"].str.get_dummies(sep="|").reindex(columns=list(GENRES), fill_value=0)
    hot.index = movies["item_id"].to_numpy()
    return hot.astype("int8")


def check_files(data_dir: Path) -> None:
    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MovieLens 1M files not found in '{data_dir}': missing {missing}. "
            f"Download ml-1m.zip from https://grouplens.org/datasets/movielens/1m/, unzip it "
            f"and pass the ml-1m folder with --data-dir, or add --download."
        )


def validate(ml: MovieLens) -> None:
    """Fail on malformed data, warn when the row counts differ from the official release."""
    for name, df in (("ratings.dat", ml.ratings), ("movies.dat", ml.movies), ("users.dat", ml.users)):
        if len(df) != EXPECTED_ROWS[name]:
            log.warning("%s has %d rows, the official release has %d", name, len(df), EXPECTED_ROWS[name])
    r = ml.ratings
    if not r["rating"].between(1, 5).all():
        raise ValueError("ratings.dat contains ratings outside 1-5")
    if r.duplicated(["user_id", "item_id"]).any():
        raise ValueError("ratings.dat contains more than one rating for the same (user, movie) pair")
    if not r["item_id"].isin(ml.movies["item_id"]).all():
        raise ValueError("ratings.dat refers to movies that are not in movies.dat")
    if not r["user_id"].isin(ml.users["user_id"]).all():
        raise ValueError("ratings.dat refers to users that are not in users.dat")


def load_movielens(data_dir: str | Path) -> MovieLens:
    data_dir = Path(data_dir)
    check_files(data_dir)
    ml = MovieLens(load_ratings(data_dir), load_movies(data_dir), load_users(data_dir))
    validate(ml)
    log.info(
        "Loaded %d ratings, %d movies, %d users from %s",
        len(ml.ratings), len(ml.movies), len(ml.users), data_dir,
    )
    return ml


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: str | Path) -> Path:
    """Download ml-1m.zip from GroupLens, check its MD5 and unpack it so that
    the .dat files end up directly in data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ml-1m.zip"
        log.info("Downloading %s", GROUPLENS_URL)
        urllib.request.urlretrieve(GROUPLENS_URL, archive)
        checksum = _md5(archive)
        if checksum != ZIP_MD5:
            raise ValueError(f"MD5 mismatch for ml-1m.zip: got {checksum}, expected {ZIP_MD5}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        for name in FILES + ("README",):
            shutil.copy2(Path(tmp) / "ml-1m" / name, data_dir / name)
    check_files(data_dir)
    return data_dir
