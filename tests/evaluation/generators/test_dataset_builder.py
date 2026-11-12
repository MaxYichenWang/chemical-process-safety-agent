"""
Tests for DatasetBuilder integration and mathematical pruning.
"""
import numpy as np

from src.evaluation.generators.dataset_builder import cosine_similarity


def test_cosine_similarity():
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert np.isclose(cosine_similarity(v1, v2), 1.0)

    v3 = [0.0, 1.0]
    assert np.isclose(cosine_similarity(v1, v3), 0.0)

    v4 = [0.5, 0.5]
    # dot is 0.5, norm is sqrt(0.5) * 1.0 -> 0.5 / 0.707 = 0.707
    assert cosine_similarity(v1, v4) > 0.7
