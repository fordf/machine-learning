
import os
import pytest
import numpy as np
from ml_algs.k_means import KMeansClassifier

DUPES = [21, 41, 45, 47, 107, 119, 139, 145, 170, 190, 195, 219, 226, 227, 248, 268, 284, 300, 315, 320, 345, 351, 352, 404, 427, 436, 476, 498, 604, 615, 617, 657, 691, 716, 727]
GROUPED_DUPES = [
    [45, 47],
    [41, 139, 615],
    [145, 320, 351, 716],
    [21, 300, 315, 352],
    [219, 284, 427],
    [227, 604, 727],
    [170, 226, 498, 691],
    [248, 404, 436, 476],
    [119, 190, 268],
    [617, 657],
    [107, 195, 345]
]

project_root = os.path.dirname(os.path.dirname(__file__))
LABELED_DATA = np.loadtxt(
    os.path.join(project_root, 'data/datas.csv'),
    delimiter=','
)
DATA = LABELED_DATA[:, range(4)]
NO_DUPE_DATA = np.delete(DATA.copy(), DUPES, axis=0)
DUPE_DATA = DATA[DUPES]


@pytest.fixture
def clf():
    """Return default KMeansClassifier."""
    return KMeansClassifier()


def test_centroid_for_every_datapoint_predicts_perfectly(clf):
    """
    Fit a centroid to every datapoint and assert each datapoint is predicted
    to be it's cluster.
    """
    data = NO_DUPE_DATA
    clf.fit(data, init_centroids=data)
    preds = clf.predict(data)
    assert (preds == np.array(range(len(data)))).all()


def test_dupes(clf):
    """Assert exact matches to centroids will be labeled to corresponding cluster."""
    data = DUPE_DATA
    centroids = DATA[[x[0] for x in GROUPED_DUPES]]
    clf.fit(data, init_centroids=centroids)
    preds = clf.predict(data)
    for i, pred in enumerate(preds):
        centroid_predicted = clf.centroids[pred]
        assert np.allclose(DATA[DUPES[i]], centroid_predicted)


def test_k_num_centroids(clf):
    """Test fit finishes with a max_iter but no min_step."""
    clf.fit(DATA[::4], k=2)
    assert len(clf.centroids) == 2


def test_max_iter():
    """Test fit finishes with a max_iter but no min_step."""
    clf = KMeansClassifier(max_iter=20, min_step=None)
    amt = len(DATA) // 2
    clf.fit(DATA[:amt:2], k=2)
    assert len(clf.predict(DATA[1:amt:2])) == amt // 2


INVALID_DATA = [
    [3, 2, None, TypeError],
    [[3], 2, None, TypeError],
    [[[3], [2]], 1, None, TypeError],
    [np.array([[]]), 2, None, TypeError],
    [np.array([[3]]), None, None, ValueError],
    [np.array([3]), 2, None, TypeError],
    [np.array([[3], [2]]), 0, None, ValueError],
    [np.array([[3], [2]]), 3, None, ValueError],
    [np.array([[3]]), 2, None, ValueError],
    [np.array([[3]]), 0, None, ValueError],
    [np.array([[3], [2]]), None, 3, TypeError],
    [np.array([[3], [2]]), None, [3], TypeError],
    [np.array([[3], [2]]), None, [[[3]]], TypeError],
    [np.array([[3]]), None, [[1], [3]], IndexError],
    [np.array([[3]]), None, [[1, 2]], IndexError],
    [np.array([[3], [2]]), None, [[], []], IndexError],
    [np.array([[3], [2]]), None, [[]], IndexError],
]


@pytest.mark.parametrize('data, k, init_centroids, error_type', INVALID_DATA)
def test_fit_invalid_data(data, k, init_centroids, error_type, clf):
    """Test my ugly error handling."""
    with pytest.raises(error_type):
        clf.fit(data, k=k, init_centroids=init_centroids)


def test_predict_raises_error_if_not_fit_yet():
    """Assert predict raises exception if clf hasn't been fitted yet."""
    clf = KMeansClassifier()
    with pytest.raises(Exception):
        clf.predict(DATA)
