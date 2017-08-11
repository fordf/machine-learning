import os
import pickle
import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:
    """
    NeuralNetwork(weights=None)

    weights:
        prefitted weights to predict with
    """

    def __init__(self, weights=None):
        self.weights = self._handle_file_or_arr(weights, 'weights')
        self.layer_sizes = None
        self.reg_lambda = 0
        self.X = None
        self.y = None

    def predict(self, x):
        """Predict labels for dataset x."""
        if self.weights is None:
            raise ValueError('Load weights into NN or fit before predicting.')
        actvs = self.forward_prop(x, self.weights)
        return self.label_handler.inverse_transform(actvs[-1].argmax(axis=1))

    def fit(self,
            X=None,
            y=None,
            layer_sizes=None,
            cost_grad_func='default',
            args=(),
            reg_lambda=0,
            options=None,
            method='CG'):
        """
        fit(
            self,
            X=None,
            y=None,
            layer_sizes=None,
            cost_grad_func='default',
            args=(),
            reg_lambda=0,
            options=None,
            method='CG'
        )

        X, y, layer_sizes, cost_grad_func required here or already
        attached to instance. Will attach to instance if provided here.

        X and y:
            - can be paths to data files, or numpy arrays

        reg_lambda:
            - 0 or positive number
            - make lower if underfitting, higher if overfitting

        cost_grad_func:
            - must be a jacobian func (return cost and weight gradients)
            - first arg needs to be the unrolled weights
            - provided options:
                - 'default', 'verbose'

        args:
            - if own cost_grad_func provided, these must be the function's
              arguments excluding the first (weights)

        valid methods:
            'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg'

        options:
            ex:
            {
                'disp': True,
                'maxiter': 400
            }
        """
        valid_methods = 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg'
        cost_grad_funcs = {
            'default': self.cost_and_gradients,
            'verbose': self.verbose_cost_gradients,
        }

        if X is not None:
            X = self._handle_file_or_arr(X, 'X')
        else:
            X = self.X
        if y is not None:
            self.label_handler = LabelHandler()
            y = self._handle_file_or_arr(y, 'y')
            y = self.label_handler.fit_transform(y)
        else:
            y = self.y
        if isinstance(cost_grad_func, str):
            cost_grad_func = cost_grad_funcs[cost_grad_func]
        if method not in valid_methods:
            raise ValueError("valid methods: {}".format(valid_methods))

        self.reg_lambda = reg_lambda or self.reg_lambda
        self.layer_sizes = layer_sizes or self.layer_sizes

        required = (('X', X),
                    ('y', y),
                    ('layer_sizes', self.layer_sizes),
                    ('cost_grad_func', cost_grad_func))
        self.assert_have_required(required)

        num_features = X.shape[1]
        self.check_layer_sizes(num_features)

        if args == () and cost_grad_func in cost_grad_funcs.values():
            args = (X, y, self.layer_sizes, self.reg_lambda)

        initial_weights = self.init_rand_weights(layer_sizes)

        optimize_result = minimize(
            cost_grad_func,
            initial_weights,
            args=args,
            method=method,
            jac=True,
            options=options
        )
        self.weights = self.reroll(optimize_result.x, layer_sizes)
        return optimize_result

    def accuracy(self, predicted, actual):
        return np.sum(predicted == actual) / len(actual)

    def save_weights(self, dirname, prefix='WLayer', weights=None):
        """
        Save weights or self.weights into new/existing directory.

        Each weight layer will be saved into its own file with corresponding
        layer index (from 1).
        """
        if weights is None:
            weights = self.weights
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for i, mtx in enumerate(weights):
            np.savetxt(
                os.path.join(dirname, '{}_{}'.format(prefix, i + 1)),
                mtx
            )

    def load_weights(self, dirname):
        """Load weights into instance sorted by number in filename."""
        def findnum(string):
            return int(''.join([x for x in string if x.isdigit()]))

        to_load = sorted(os.listdir(dirname), key=findnum)
        weights = []
        for file in to_load:
            weights.append(np.loadtxt('/'.join((dirname,file))))
        self.weights = np.array(weights)

    def pickleme(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as f:
            nn = pickle.load(f)
        return nn

    def cost_and_gradients(self, unrolled_weights, x, y, layer_sizes, reg_lambda):
        """Compute cost of dataset with unrolled_weights."""
        m = x.shape[0]
        thetas = self.reroll(unrolled_weights, layer_sizes)
        actvs = self.forward_prop(x, thetas)

        cost = 1 / m * np.sum(-y * np.log(actvs[-1]) - (1-y) * np.log(1 - actvs[-1]))
        cost += reg_lambda / 2 / m * sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)

        partial_ds = [actvs[-1] - y]
        grads = []
        for i in range(1, len(layer_sizes)):
            activation = actvs[-i - 1]
            theta = thetas[-i]
            big_d = partial_ds[i - 1].T.dot(self.with_ones(activation))
            grad = big_d / m
            grad += reg_lambda / m * np.hstack((np.zeros((len(theta), 1)), theta[:, 1:]))
            grads.append(grad)
            if i < len(layer_sizes) - 1:
                partial_d = partial_ds[i - 1].dot(theta[:, 1:])
                partial_d *= activation * (1 - activation)
                partial_ds.append(partial_d)
        return cost, self.unroll(grads[::-1])

    def forward_prop(self, x, weights):
        """Return activations at each layer during forward propagation."""
        actvs = [x]
        for theta in weights:
            a = self.sigmoid(self.with_ones(actvs[-1]).dot(theta.T))
            actvs.append(a)
        return actvs

    @staticmethod
    def unroll(arrays):
        """Unroll list of matrixes into vector."""
        return np.concatenate([arr.flatten(order='F') for arr in arrays])

    @staticmethod
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

    def init_rand_weights(self, layer_sizes):
        params = []
        for i in range(len(layer_sizes) - 1):
            params.append(self._init_rand_weights(layer_sizes[i], layer_sizes[i + 1]))
        return self.unroll(params)

    @staticmethod
    def _init_rand_weights(num_incoming, num_outgoing, eps=.12):
        randoms = np.random.rand(num_outgoing, 1 + num_incoming)
        return randoms * 2 * eps - eps

    @staticmethod
    def sigmoid(z):
        """Return sigmoid of z."""
        try:
            return 1.0 / (1.0 + np.e ** -z)
        except FloatingPointError:
            return 0

    def sigmoid_grad(self, z):
        """Return sigmoid of z."""
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def with_ones(x):
        """Return mtx with additional first column of ones."""
        return np.hstack((np.ones((len(x), 1)), x))

    def verbose_cost_gradients(self, unrolled_weights, x, y, layer_sizes, reg_lambda):
        """Call cost_and_gradients, print cost."""
        cost, grads = self.cost_and_gradients(unrolled_weights, x, y, layer_sizes, reg_lambda)
        print(cost)
        return cost, grads

    def check_layer_sizes(self, num_features):
        num_classes = len(self.label_handler.labelmap)
        if num_features != self.layer_sizes[0]:
            raise ValueError('Input layer must be sized equal to number '
                'of features: {} != {}'.format(self.layer_sizes[0], num_features))
        if len(self.layer_sizes) < 2:
            raise ValueError('Must be at least 2 layers.')
        if self.layer_sizes[-1] != num_classes:
            raise ValueError('Output layer must be sized equal to num classes: '
                             '{} != {}'.format(self.layer_sizes[-1], num_classes))

    def assert_have_required(self, required):
        missing = set(item[0] for item in required if item[1] is None)
        if len(missing):
            raise ValueError('Required: {}'.format(missing))

    @staticmethod
    def _handle_file_or_arr(given, param_name, required=()):
        if given is None:
            if given in required:
                raise ValueError('Required: ' + param_name)
            return None
        if isinstance(given, str):
            data = np.loadtxt(given)
        elif isinstance(given, list):
            data = np.array(given)
        elif isinstance(given, np.ndarray):
            data = given
        else:
            raise TypeError(param_name + ' must be filename or array/list')
        return data


class LabelHandler:

    def fit_transform(self, labels):
        self.labelmap, mapped = np.unique(labels, return_inverse=True)
        return self.binarize(mapped)

    def binarize(self, vector):
        identity = np.identity(len(self.labelmap))
        return identity[vector]

    def inverse_transform(self, labels):
        return self.labelmap[labels.astype(int)]


def visualize(image_array, shape, predicted=None, actual=None, label_dict=None, order='F'):
    """If your data just so happened to be ."""
    from matplotlib import pyplot as plt
    from random import sample
    image_array = image_array.copy()
    try:
        for i in sample(list(range(len(image_array))), len(image_array)):
            if predicted is not None:
                print('NN predicted: {}'.format(predicted[i]))
            if actual is not None:
                print('actual: ' + str(actual[i]))
            plt.imshow(image_array[i].reshape(shape, order=order))
            plt.show()
            inp = input('q to quit, anything else to continue\n')
            if inp in ['q', 'quit', 'exit']:
                break
    except KeyboardInterrupt:
        plt.close()
