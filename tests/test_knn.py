"""Test K-nearest neighbors classifier."""

import os
import pytest
import numpy as np

from ml_algs.knn import KNearestNeighbors


DATASET = np.array([
    [2.77, 1.78, 5.7, 4.4, 0],
    [1.72, 1.16, 5.7, 4.4, 0],
    [3.67, 2.81, 5.7, 4.4, 0],
    [3.96, 2.61, 5.7, 4.4, 0],
    [2.99, 2.20, 5.7, 4.4, 0],
    [7.49, 3.16, 5.7, 4.4, 1],
    [9.00, 3.33, 5.7, 4.4, 1],
    [7.44, 0.47, 5.7, 4.4, 1],
    [10.12, 3.23, 5.7, 4.4, 1],
    [6.64, 3.319, 5.7, 4.4, 1]
])

project_root = os.path.dirname(os.path.dirname(__file__))
FLOWERS = np.loadtxt(
    os.path.join(project_root, 'data/flowers_data.csv'),
    delimiter=',',
    skiprows=1,
    usecols=(0, 1, 2, 3, 4)
)


@pytest.fixture
def knn():
    """Create a Knn to run tests on."""
    return KNearestNeighbors()


@pytest.mark.parametrize('k', range(1, len(FLOWERS)))
def test_knn_predict_returns_a_class_label(k, knn):
    """Predict should return a zero or one."""
    labels = knn.predict(FLOWERS, DATASET[:, range(4)], k=k)
    assert len(labels) == len(DATASET)
    assert (np.unique(labels) == (0, 1)).all()


@pytest.mark.parametrize('label_col', range(DATASET.shape[1]))
def test_label_col(label_col, knn):
    """Test any column can be label col."""
    cols = np.array(range(DATASET.shape[1]))
    not_label_cols = cols[cols != label_col]
    data = np.hstack((
        DATASET[:, range(label_col)],
        DATASET[:, [-1]],
        DATASET[:, range(label_col, DATASET.shape[1] - 1)]
    ))
    labels = knn.predict(data, data[:, not_label_cols], label_col=label_col, k=3)
    assert len(labels) == len(DATASET)
    assert (np.unique(labels) == (0, 1)).all()
    assert (labels == DATASET[:, -1]).all()


def test_knn_predicts_perfectly_k_is_1(knn):
    labels = knn.predict(FLOWERS, FLOWERS[:, range(4)], k=1)
    assert (labels == FLOWERS[:, -1]).all()


INVALID_PARAMS = [
    [DATASET[::2], DATASET[1::2, range(4)], 0, ValueError],
    [DATASET[::2], DATASET[1::2, range(4)], 11, ValueError],
    [DATASET[::2], DATASET[1::2, range(3)], 5, IndexError],
    [DATASET[::2], DATASET[1::2], 5, IndexError],

]


@pytest.mark.parametrize('train, pred, k, error', INVALID_PARAMS)
def test_knn_raises_error_if_k_invalid(train, pred, k, error, knn):
    """
    When initializing knn, error should raise if k is less than zero, greater
    than the length of the dataset, or not an integer.
    """
    with pytest.raises(error):
        knn.predict(train, pred, k=k)
