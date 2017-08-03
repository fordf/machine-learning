import numpy as np
from sklearn.preprocessing import LabelBinarizer


def predict(x, thetas, unrolled=False):
    """Predict labels for dataset x."""
    actvs = _forward_prop(x, thetas)
    labels = actvs[-1].argmax(axis=1)
    return labels


def verbose_cost_gradients(nn_params, layer_sizes, x, y, reg_lambda):
    """Call cost_and_gradients, print cost."""
    cost, grads = cost_and_gradients(nn_params, layer_sizes, x, y, reg_lambda)
    print(cost)
    return cost, grads


def cost_and_gradients(nn_params, layer_sizes, x, y, reg_lambda):
    """Compute cost of dataset with nn_params."""
    m = x.shape[0]
    thetas = reroll(nn_params, layer_sizes)
    lb = LabelBinarizer()
    yvec = lb.fit_transform(y)
    actvs = _forward_prop(x, thetas)

    cost = 1 / m * np.sum(-yvec * np.log(actvs[-1]) - (1-yvec) * np.log(1-actvs[-1]))
    cost += reg_lambda / 2 / m * sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)

    partial_ds = [actvs[-1] - yvec]
    grads = []
    for i in range(1, len(layer_sizes)):
        activation = actvs[-i - 1]
        theta = thetas[-i]
        big_d = partial_ds[i - 1].T.dot(with_ones(activation))
        grad = big_d / m
        grad += reg_lambda / m * np.hstack((np.zeros((len(theta), 1)), theta[:, 1:]))
        grads.append(grad)
        if i < len(layer_sizes) - 1:
            partial_d = partial_ds[i - 1].dot(theta[:, 1:])
            partial_d *= activation * (1 - activation)
            partial_ds.append(partial_d)
    return cost, unroll(grads[::-1])


def accuracy(predicted, y):
    """Cross validate kinda."""
    assert all(np.unique(predicted) == np.unique(y))
    return np.sum(predicted == y) / len(y)


def sigmoid(z):
    """Return sigmoid of z."""
    try:
        return 1.0 / (1.0 + np.e ** -z)
    except FloatingPointError:
        return 0


def sigmoid_grad(z):
    """Return sigmoid of z."""
    sig = sigmoid(z)
    return sig * (1 - sig)


def with_ones(x):
    """Return mtx with additional first column of ones."""
    # import ipdb; ipdb.set_trace()
    return np.hstack((np.ones((len(x), 1)), x))


def visualize(image_array, labels, y=None, label_dict=None):
    """Imshow each image_mtx alongside predicted label."""
    assert image_array.ndim in (3, 4)
    from matplotlib import pyplot as plt
    from random import sample
    if label_dict is None:
        label_dict = {l: l for l in np.unique(labels)}
    try:
        for i in sample(list(range(len(image_array))), len(image_array)):
            print('NN predicted: {}'.format(label_dict[labels[i]]))
            if y is not None:
                print('actual: ' + str(y[i]))
            plt.imshow(image_array[i])
            plt.show()
            inp = input('q to quit, anything else to continue\n')
            if inp in ['q', 'quit', 'exit']:
                break
    except KeyboardInterrupt:
        plt.close()


def _forward_prop(x, thetas, return_activations=False):
    """Return activations at each layer during forward propagation."""
    actvs = [x]
    for theta in thetas:
        a = sigmoid(with_ones(actvs[-1]).dot(theta.T))
        actvs.append(a)
    return actvs


def unroll(arrays):
    """Unroll list of matrixes into vector."""
    return np.concatenate([arr.flatten(order='F') for arr in arrays])


def reroll(nn_params, layer_sizes):
    """Reshape vector nn_params into 2 dimensions according to layer_sizes."""
    thetas = []
    num_thetas = len(layer_sizes) - 1
    splits = [0] + [
        (1 + layer_sizes[i]) * layer_sizes[i + 1] for i in range(num_thetas)
    ]
    for i in range(num_thetas):
        theta = nn_params[splits[i]:splits[i] + splits[i + 1]]
        thetas.append(theta.reshape(layer_sizes[i] + 1, layer_sizes[i + 1]).T)
    return thetas


def init_rand_params(layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        params.append(_init_rand_weights(layer_sizes[i], layer_sizes[i + 1]))
    return unroll(params)


def _init_rand_weights(num_incoming, num_outgoing, eps=.12):
    randoms = np.random.rand(num_outgoing, 1 + num_incoming)
    return randoms * 2 * eps - eps
