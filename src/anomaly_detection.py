import numpy as np
from scipy.stats import multivariate_normal


class AnomalyDetector:

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.variance = np.var(X - self.mu, axis=0)

    def predict(self, X):
        try:
            return self.predict_probabilities(X) < self.threshold
        except NameError:
            raise NameError('Set threshold with select_threshold or manually.')

    def select_threshold(self, probabilities, y_true, steps=1000):
        """
        Find the ideal threshold (one that yields highest f1 score)
        to divide anomalies from normies.
        """
        best_threshold, best_f1, f1 = 0, 0, 0
        max_prob, min_prob = max(probabilities), min(probabilities)
        stepsize = (max_prob - min_prob) / steps

        for threshold in np.arange(min_prob, max_prob, stepsize):
            predictions = probabilities < threshold
            trues = predictions[predictions == y_true]
            falses = predictions[predictions != y_true]

            true_positives = sum(trues)
            false_positives = sum(falses)
            false_negatives = sum(falses == 0)

            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        print("Best: Threshold={} with f1_score={}".format(best_threshold, best_f1))

    def predict_probabilities(self, X):
        dist = multivariate_normal(
            mean=self.mu,
            cov=np.identity(len(self.mu)) * self.variance
        )
        return dist.pdf(X)