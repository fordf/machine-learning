"""Implement K Nearest Neighbour Classifier in Python."""

from types import FunctionType
from collections import Counter

import numpy as np


class KNearestNeighbors(object):
    """
    K Nearest Neigbour Classifier.

    KNearestNeighbors(distance_metric='euclidean')

    predefined distance metrics:
        - 'euclidean'(sqrt of sum of squared diffs)
        - 'manhattan'(sum of diffs)
    """

    def __init__(self, distance_metric='euclidean'):
        metrics = {
            'euclidean': self.euclidean_distances,
            'manhattan': self.manhattan_distances
        }
        if isinstance(distance_metric, str) and distance_metric in metrics:
            self.distance_metric = metrics[distance_metric]
        elif isinstance(distance_metric, FunctionType):
            self.distance_metric = distance_metric
        else:
            raise ValueError('distance_metric must be one of '
        'provided distance metrics: {"euclidean", "manhattan"} '
        'or function which takes in two np arrays as parameters.')


    def predict(self, predict_data, labeled_data, labels=None, k=5, label_col=-1):
        """
        Given a data point, predict the class of that data, based on dataset.

        knn.predict(predict_data, labeled_data, k=5, label_col=-1)
            - predict_data: data to be labeled
            - labeled_data: labeled data
            - labels: labels corresponding to labeled data. If not provided,
                labels will be extracted using label_col.
            - k: how many neighbors to group with
            - label_col: labeled_data, col index containing labels
        """
        labeled_data = np.array(labeled_data)
        predict_data = np.array(predict_data)
        if type(k) is not int or k > len(labeled_data) or k < 1:
            raise ValueError("k must be an integer value between 1 and the length of labeled_data")
        try:
            assert labeled_data.shape[1] == predict_data.shape[1] + 1
        except:
            raise IndexError('labeled_data must have the same width as predict_data with an extra col for labels')
        if labels is None:
            labels = labeled_data[:, label_col]
            labeled_data = np.delete(labeled_data, label_col, axis=1)
        return self._predict(labeled_data, labels, predict_data, k)

    def _predict(self, labeled_data, labels, predict_data, k, res=None):
        res = res if res else []
        for p in predict_data[len(res):]:
            dists = np.sqrt(np.sum((labeled_data - p)**2, axis=1))
            closest = labels[np.argpartition(dists, k - 1)[:k]]
            majority = Counter(closest).most_common()
            if len(majority) > 1:
                return self._predict(labeled_data, labels, predict_data, k - 1, res)
            res.append(majority[0][0])
        return np.array(res)

    @staticmethod
    def euclidean_distances(Xtrain, Xpred):
        return np.sqrt(np.sum((Xtrain - Xpred)**2, axis=1))

    @staticmethod
    def manhattan_distances(Xtrain, Xpred):
        return np.sum(Xtrain - Xpred, axis=1)