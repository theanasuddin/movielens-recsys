
import numpy as np
import pytest

from recsys.reranking.mmr import mmr_rerank


def test_lambda_one_reproduces_relevance_order():
    items = np.array([10, 20, 30])
    scores = np.array([0.9, 0.8, 0.7])

    genres = np.array([
        [1, 0],  # Action
        [1, 0],  # Action
        [0, 1],  # Comedy
    ])

    result = mmr_rerank(
        items, scores, genres, k=3, lambda_=1.0
    )

    np.testing.assert_array_equal(result, [10, 20, 30])


def test_mmr_can_choose_a_different_genre():
    items = np.array([10, 20, 30])
    scores = np.array([0.9, 0.8, 0.7])

    genres = np.array([
        [1, 0],  # Action
        [1, 0],  # Action
        [0, 1],  # Comedy
    ])

    result = mmr_rerank(
        items, scores, genres, k=2, lambda_=0.5
    )

    np.testing.assert_array_equal(result, [10, 30])


def test_invalid_lambda_is_rejected():
    with pytest.raises(ValueError, match="lambda_"):
        mmr_rerank(
            [1], [0.9], [[1, 0]], lambda_=1.2
        )



from scipy.sparse import csr_matrix

from recsys.evaluation.diversity import (
    intra_list_diversity,
    genre_calibration_error,
)


def test_ild_manual_example():
    genres = np.array([
        [1, 0],  # Action
        [1, 0],  # Action
        [0, 1],  # Comedy
    ])

    recommendations = np.array([[0, 1, 2]])

    ild = intra_list_diversity(recommendations, genres)

    assert ild[0] == pytest.approx(2 / 3)


def test_ild_identical_genres():
    genres = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
    ])

    recommendations = np.array([[0, 1, 2]])

    ild = intra_list_diversity(recommendations, genres)

    assert ild[0] == pytest.approx(0.0)


def test_calibration_perfect_match():
    genres = np.array([
        [1, 0],  # Action
        [0, 1],  # Comedy
    ])

    recommendations = np.array([[0, 1]])

    history = csr_matrix([[1, 1]])

    error = genre_calibration_error(
        recommendations,
        genres,
        history,
        user_indices=np.array([0]),
    )

    assert error[0] == pytest.approx(0.0)


def test_calibration_completely_different():
    genres = np.array([
        [1, 0],  # Action
        [0, 1],  # Comedy
    ])

    recommendations = np.array([[1]])

    history = csr_matrix([[1, 0]])

    error = genre_calibration_error(
        recommendations,
        genres,
        history,
        user_indices=np.array([0]),
    )

    assert error[0] == pytest.approx(1.0)


def test_calibration_no_positive_history():
    genres = np.array([
        [1, 0],
        [0, 1],
    ])

    recommendations = np.array([[0, 1]])

    history = csr_matrix([[0, 0]])

    error = genre_calibration_error(
        recommendations,
        genres,
        history,
        user_indices=np.array([0]),
    )

    assert np.isnan(error[0])

