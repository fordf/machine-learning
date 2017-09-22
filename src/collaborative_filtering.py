import numpy as np
from scipy.optimize import minimize
from helpers import rand_matrix, unroll


def fit(y, R, num_features, reg_param):
    """
    Find optimized weight matrixes U and W which can be used to predict values
    describing the relationship between two groups.

    y: matrix detailing a non-entirely known relationship between two groups.
       For example, if y is a matrix of movie ratings vs users, U will be shaped
       num movies vs num features and W will be num users vs num features.
       It's ideal to normalize y column-wise before fitting, so users (for example)
       with no known data will have their predictions be the mean known values.

    R: matrix shaped like y which identifies the known values of y
       (1 if known, 0 if not)

    Returns (scipy optimize result, U, W)
    """
    num_a, num_b = y.shape
    init_U = rand_matrix((num_a, num_features))
    init_W = rand_matrix((num_b, num_features))
    initial_weights = unroll((init_U, init_W))
    optimize_result = minimize(
        cost_and_gradients,
        initial_weights,
        args=(y, R, num_a, num_b, num_features, reg_param),
        method='CG',
        jac=True,
        options={'maxiter':100}
    )
    weights = optimize_result.x
    U = weights[:num_a * num_features].reshape((num_a, num_features), order='F')
    W = weights[num_a * num_features:].reshape((num_b, num_features), order='F')
    return optimize_result, U, W


def predict(U, W, y_means=None):
    """Return prediction matrix."""
    prediction_matrix = U.dot(W.T)
    if y_means is not None:
        prediction_matrix += y_means
    return prediction_matrix


def normalize(y, R):
    """
    Normalize each row of y to be centered at 0.

    Returns (y_normalized, row_means)
    """
    means = np.zeros((len(y), 1))
    for row_i, row in enumerate(y):
        means[row_i] = np.mean(row[R[row_i] == 1])
    return (y - means) * R, means


def cost_and_gradients(weights, y, R, num_a, num_b, num_features, reg_param):
    """
    Return cost and gradients to optimize.

    weights: unrolled (numpy) array of both weight matrixes to be optimized;
             first comes 'a', then 'b', corresponding to num_a and num_b
    y: matrix of a vs b where a and b are items whose relationship we are trying
       to learn (e.g. users and movies)
    R: matrix of shape y giving which data in y is known (1 if known, 0 if not)
    """
    U = weights[:num_a * num_features].reshape((num_a, num_features), order='F')
    W = weights[num_a * num_features:].reshape((num_b, num_features), order='F')
    U_grad = np.zeros_like(U)
    W_grad = np.zeros_like(W)

    J = np.sum((U.dot(W.T) - y)[R == 1] ** 2) / 2
    J += reg_param * np.sum(U ** 2) / 2 + reg_param * np.sum(W ** 2) / 2
    print(J)

    for i in range(num_a):
        idxs = R[i] == 1
        relvnt_W = W[idxs]
        relvnt_y = y[i, idxs]
        U_grad[i] = (U[i].dot(relvnt_W.T) - relvnt_y).dot(relvnt_W) + reg_param * U[i]

    for j in range(num_b):
        idxs = R[:, j] == 1
        relvnt_U = U[idxs]
        relvnt_y = y[idxs, j]
        W_grad[j] = (relvnt_U.dot(W[j].T) - relvnt_y).T.dot(relvnt_U) + reg_param * W[j]

    gradient = unroll((U_grad, W_grad))
    return J, gradient


test_R = np.array([
  [1, 1, 0, 0],
  [1, 0, 0, 0],
  [1, 0, 0, 0],
  [1, 0, 0, 0],
  [1, 0, 0, 0],
])

test_y = np.array([
  [5, 4, 0, 0],
  [3, 0, 0, 0],
  [4, 0, 0, 0],
  [3, 0, 0, 0],
  [3, 0, 0, 0]
])

test_U = np.array([
   [1.048686  ,-0.400232 , 1.194119],
   [0.780851  ,-0.385626 , 0.521198],
   [0.641509  ,-0.547854 ,-0.083796],
   [0.453618  ,-0.800218 , 0.680481],
   [0.937538  , 0.106090 , 0.361953],
])

test_W = np.array([
   [0.28544 , -1.68427 , 0.26294],
   [0.50501 , -0.45465 , 0.31746],
  [-0.43192 , -0.47880 , 0.84671],
   [0.72860 , -0.27189 , 0.32684],
])