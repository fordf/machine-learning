"""Implement K Nearest Neighbour Classifier in Python."""

from collections import Counter
import numpy as np


class KNearestNeighbors(object):
    """K Nearest Neigbour Classifier."""

    def predict(self, predict_data, labeled_data, k=5, label_col=-1):
        """
        Given a data point, predict the class of that data, based on dataset.

        knn.predict(labeled_data, predict_data, k=5, label_col=-1)
            - labeled_data: labeled data
            - predict_data: data to be labeled
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
