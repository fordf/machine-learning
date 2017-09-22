import numpy as np


class PCA:

    def __init__(self, X):
        """
        Singular value decomposition, returns (eigens, S).

        eigens is used to reduce (and recover) dimensionality of X
        with functions project_data and recover_data.

        S is used to compute the variance retained after X has been
        reduced, and then recovered, using the eigenvectors.
        """
        covariance_matrix = X.T.dot(X) / len(X)
        self.eigens, self.S, V = np.linalg.svd(covariance_matrix)

    def project_data(self, data, num_pcs):
        """Reduce data to number of principal components = num_pcs using eigenvectors."""
        return data.dot(self.eigens[:, :num_pcs])

    def recover_data(self, reduced_data):
        """Expand data's dimensionality to original size, values won't match exactly."""
        return reduced_data.dot(self.eigens[:reduced_data.shape[1]])

    def variance_retained(self, num_pcs=None):
        """
        Compute the variance retained after data is reduced to number of principal
        components equal to num_pcs.

        If num_pcs is None, returns variances retained for every possible num_pcs
        as a list.
        """
        total_sum = np.sum(self.S)
        if num_pcs is not None:
            return np.sum(self.S[:num_pcs]) / total_sum
        variances_retained = [np.sum(self.S[:i+1]) / total_sum for i in range(len(self.S))]
        return variances_retained
