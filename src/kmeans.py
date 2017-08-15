import random
import numpy as np


class KMeansClassifier(object):
    """
    Unsupervised clustering algorithm.

    clf = KMeansClassifier(max_iter=None, min_step='auto')
        -max_iter: the number of iterations before the algorithm stops
        -min_step: the smallest difference in distance between an old centroid
                   and a new centroid before the algorithm stops. If 'auto',
                   min_step is calculated to be 1/1000th of largest first step.

    clf.fit(data, k=2)
        -k: number of nodes to generate

    clf.predict(data)
    """

    def __init__(self, max_iter=None, min_step='auto'):
        """Initialize classifier instance."""
        assert max_iter or min_step
        if min_step is not None:
            assert (isinstance(min_step, (int, float)) and min_step > 0) or min_step == 'auto'
        if max_iter is not None:
            assert isinstance(max_iter, int) and max_iter > 0
        self.max_iter = max_iter
        self.min_step = min_step
        self.centroids = []

    def fit(self, data, k=None, init_centroids=None):
        """
        Find centroids of clusters.

        km_clf.fit(data, k=None, init_centroids=None)
            - data: 2d numpy array

            k or init_centroids required
            - k: number of centroids to initialize to random points in data
            - init_centroids: starting locations of centroids
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be non-empty 2d numpy array')
        if k is None and init_centroids is None:
            raise ValueError('parameter k or init_centroids required')
        if init_centroids is None:
            if k < 1 or k > len(data):
                raise ValueError('k must be in range 1 - len(data)')
            self.centroids = data[random.sample(range(len(data)), k)]
        else:
            try:
                self.centroids = np.array(init_centroids)
                assert len(self.centroids.shape) == 2
            except:
                raise TypeError('init_centroids must be 2d np array or list of lists')
            if len(self.centroids) > len(data):
                raise IndexError('init_centroids can\'t be longer than data')
            if self.centroids.shape[1] != data.shape[1]:
                raise IndexError(
                    '''init_centroids must be each be same shape
                    as datapoints in data: {}'''.format(data[0].shape))
        iters = 0
        while True:
            labels = self._predict(data)
            prev = self.centroids.copy()
            for i in range(len(self.centroids)):
                cluster = data[labels == i]
                if len(cluster):
                    self.centroids[i] = np.mean(cluster, axis=0)
                else:
                    self.centroids = self.centroids[:i] + self.centroids[i+1:]
            iters += 1
            if self.max_iter and iters > self.max_iter:
                break
            if self.min_step:
                biggest_step = np.max(
                    np.linalg.norm(prev - self.centroids, axis=1)
                )
                if self.min_step == 'auto':
                    self.min_step = biggest_step / 1000
                if biggest_step <= self.min_step:
                    break

    def predict(self, data):
        """Return predicted classes for given data."""
        if len(self.centroids) == 0:
            raise Exception("Classifier must be fit before using to predict.")
        return self._predict(data)

    def _predict(self, data):
        dists = np.array(
            [np.sum((data - centroid)**2, axis=1) for centroid in self.centroids]
        )
        return np.argmin(dists, axis=0)
