from __future__ import annotations

import pytest

from recsys.config import load_config
from recsys.data import genre_matrix, load_movielens
from recsys.data.movielens import GENRES, check_files
from recsys.utils import REPO_ROOT


def _write_dat(folder, ratings, movies, users):
    (folder / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
    (folder / "movies.dat").write_text("\n".join(movies) + "\n", encoding="latin-1")
    (folder / "users.dat").write_text("\n".join(users) + "\n", encoding="latin-1")


def test_parses_the_double_colon_format(tmp_path):
    _write_dat(
        tmp_path,
        ratings=["1::10::5::978300760", "1::20::3::978302109", "2::10::4::978301968"],
        movies=[
            '10::Star Wars: Episode IV - A New Hope (1977)::Action|Adventure|Fantasy|Sci-Fi',
            '20::Misérables, Les (1995)::Drama|Musical',
            '30::Hellraiser: "Bloodline" (1996)::Horror',
        ],
        users=["1::F::1::10::48067", "2::M::56::16::70072"],
    )
    ml = load_movielens(tmp_path)

    assert ml.ratings["rating"].tolist() == [5, 3, 4]
    assert ml.movies.loc[0, "title"] == "Star Wars: Episode IV - A New Hope (1977)"
    assert ml.movies.loc[1, "title"] == "Misérables, Les (1995)"
    assert ml.movies.loc[2, "title"] == 'Hellraiser: "Bloodline" (1996)'
    assert ml.movies["year"].tolist() == [1977, 1995, 1996]
    assert ml.users["zip"].tolist() == ["48067", "70072"]


def test_rejects_duplicate_ratings(tmp_path):
    _write_dat(
        tmp_path,
        ratings=["1::10::5::978300760", "1::10::3::978302109"],
        movies=["10::A (1995)::Drama"],
        users=["1::F::1::10::48067"],
    )
    with pytest.raises(ValueError):
        load_movielens(tmp_path)


def test_missing_files_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ratings.dat"):
        check_files(tmp_path)


def test_genre_matrix_has_all_18_genres(tiny_ml):
    hot = genre_matrix(tiny_ml.movies)
    assert list(hot.columns) == list(GENRES) and len(GENRES) == 18
    assert hot.loc[30, ["Drama", "Action"]].tolist() == [1, 1]
    assert hot.loc[30].sum() == 2


def test_default_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.split.mode == "calendar"
    assert cfg.evaluation.k == 10


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  mode: calendar\n  test_starts: '2001-01-01'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test_starts"):
        load_config(path)
