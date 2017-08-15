import numpy as np


def pca(data):
    """
    Singular value decomposition, returns (eigens, S).

    eigens is used to reduce (and recover) dimensionality of data
    with functions project_data and recover_data.

    S is used to compute the variance retained after data has been
    reduced, and then recovered, using the eigenvectors.
    """
    covariance_matrix = data.T.dot(data) / len(data)
    eigens, S, V = np.linalg.svd(covariance_matrix)
    return eigens, S


def project_data(data, eigens, num_pcs):
    """Reduce data to number of principal components = num_pcs using eigenvectors."""
    return data.dot(eigens[:, :num_pcs])


def recover_data(reduced_data, eigens):
    """Expand data's dimensionality to original size, values won't match exactly."""
    return reduced_data.dot(eigens[:reduced_data.shape[1]])


def variance_retained(S, num_pcs=None):
    """
    Compute the variance retained after data is reduced to number of principal
    components equal to num_pcs.

    If num_pcs is None, returns variances retained for every possible num_pcs
    as a list.
    """
    total_sum = np.sum(S)
    if num_pcs is not None:
        return np.sum(S[:num_pcs]) / total_sum
    variances_retained = [np.sum(S[:i+1]) / total_sum for i in range(len(S))]
    return variances_retained
