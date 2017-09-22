import os
import logging
import numpy as np
from scipy.optimize import minimize

from .helpers import rand_matrix, unroll


class NeuralNetwork:
    """
    NeuralNetwork(weights=None, label_names=None)

    weights:
        prefitted weights to predict with

    label_names:
        if labels learned from training data aren't desirable, provide this to
        map to other values.
        Ex: w/o label_names -> [1,0,1,3,2]
            w/ labels_names = ['dog', 'cat', 'frog', 'bird'] ->
                ['cat', 'dog', 'cat', 'bird', 'frog']
    """

    def __init__(self, weights=None, label_names=None):
        self.weights = self._handle_file_or_arr(weights, 'weights')
        self.layer_sizes = None
        self.reg_param = 0
        self.X_train = None
        self.y_train = None
        self.label_handler = None
        self.label_names = label_names

    def predict(self, x, label_names=None):
        """Predict labels for dataset x."""
        label_names = label_names or self.label_names
        if self.weights is None:
            raise ValueError('Load weights into NN or fit before predicting.')
        output = self.forward_prop(x, self.weights)[-1]
        labels = self.label_handler.inverse_transform(output.argmax(axis=1)).astype(int)
        if label_names is not None:
            labels = np.array(self.label_names)[labels]
        return labels

    def fit(self,
            X=None,
            y=None,
            layer_sizes=None,
            cost_grad_func='default',
            args=(),
            reg_param=0,
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
            reg_param=0,
            options=None,
            method='CG'
        )

        X, y, layer_sizes, cost_grad_func required here or already
        attached to instance. Will attach to instance if provided here.

        X:
            - shape = (num input, num features)

        X and y:
            - can be paths to data files, or numpy arrays

        layer_sizes: list of layer ... sizes not including bias units

        reg_param:
            - 0 or positive number
            - make lower if underfitting, higher if overfitting

        cost_grad_func:
            - must be a jacobian func (return cost and unrolled weight gradients)
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
        self.prefit(X, y, layer_sizes, reg_param, method)

        cost_grad_funcs = {
            'default': self.cost_and_gradients,
            'verbose': self.verbose_cost_gradients,
        }
        if isinstance(cost_grad_func, str):
            args = (self.X_train, self.y_train, self.layer_sizes, self.reg_param)
            if cost_grad_func == 'verbose':
                logger = logging.getLogger('cost_grads')
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler('train.log')
                fh.setLevel(logging.INFO)
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                logger.addHandler(fh)
                logger.addHandler(ch)
                args += (logger,)
            cost_grad_func = cost_grad_funcs[cost_grad_func]

        required = (('X', self.X_train),
                    ('y', self.y_train),
                    ('layer_sizes', self.layer_sizes),
                    ('cost_grad_func', cost_grad_func))
        self.assert_have_required(required)
        self.check_layer_sizes(self.X_train)

        initial_weights = self.init_rand_weights(self.layer_sizes)

        optimize_result = minimize(
            cost_grad_func,
            initial_weights,
            args=args,
            method=method,
            jac=True,
            options=options
        )
        self.weights = self.reroll(optimize_result.x, self.layer_sizes)
        return optimize_result

    def prefit(self,
               X=None,
               y=None,
               layer_sizes=None,
               reg_param=0,
               method='CG'):
        """Set up any of the nitty gritty hyperparams prior to finding minimum weights."""

        valid_methods = 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg'

        if y is not None:
            self.label_handler = LabelHandler()
            y = self._handle_file_or_arr(y, 'y')
            self.y_train = self.label_handler.fit_transform(y)
        self.layer_sizes = layer_sizes or self.layer_sizes
        self.reg_param = reg_param or self.reg_param
        if X is not None:
            self.X_train = self._handle_file_or_arr(X, 'X')
            if self.layer_sizes is not None:
                self.check_layer_sizes(self.X_train)
        if method not in valid_methods:
            raise ValueError("valid methods: {}".format(valid_methods))

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
            weights.append(np.loadtxt('/'.join((dirname, file))))
        self.weights = np.array(weights)

    def cost_and_gradients(self, unrolled_weights, x, y, layer_sizes, reg_param):
        """
        Compute regularized cost of dataset with unrolled_weights.

        cost_and_gradients(self, unrolled_weights, x, y, layer_sizes, reg_param)

            - unrolled_weights: vector weights unrolled in 'F' (fortran?/matlab) order.
            - x, y: train_on, labels
            - layer_sizes: not including bias units
            - reg_param: punish large weights with larger reg
        """
        m = x.shape[0]
        weights = self.reroll(unrolled_weights, layer_sizes)
        actvs = self.forward_prop(x, weights)

        cost = np.sum(-y  * np.log(actvs[-1]) - (1-y) * np.log(1 - actvs[-1])) / m
        cost += reg_param / 2 / m * sum(np.sum(theta[:, 1:] ** 2) for theta in weights)

        partial_d = actvs[-1] - y
        grads = [None] * len(weights)
        for i in range(1, len(layer_sizes)):
            prev_activation = actvs[-i - 1]
            wlayer = weights[-i]
            grad = partial_d.T.dot(self.with_ones(prev_activation)) / m
            grad += reg_param / m * np.hstack((np.zeros((len(wlayer), 1)), wlayer[:, 1:]))
            grads[i-1] = grad
            if i < len(weights):
                partial_d = partial_d.dot(wlayer[:, 1:])
                partial_d *= prev_activation * (1 - prev_activation)
        return cost, unroll(grads[::-1])

    def verbose_cost_gradients(self, unrolled_weights, x, y, layer_sizes, reg_param, logger):
        """Call cost_and_gradients, print cost."""
        cost, grads = self.cost_and_gradients(unrolled_weights, x, y, layer_sizes, reg_param)
        logger.info('{:.6f} {:.6f}'.format(cost, np.linalg.norm(grads)))
        return cost, grads

    def forward_prop(self, x, weights):
        """Return activations at each layer during forward propagation."""
        actvs = [x]
        for theta in weights:
            a = self.sigmoid(self.with_ones(actvs[-1]).dot(theta.T))
            actvs.append(a)
        return actvs

    @staticmethod
    def reroll(unrolled_weights, layer_sizes):
        weights = []
        prev_split = 0
        for i in range(len(layer_sizes) - 1):
            m, n = layer_sizes[i + 1], layer_sizes[i] + 1
            next_split = prev_split + m * n
            wlayer = unrolled_weights[prev_split:next_split]
            weights.append(wlayer.reshape(m, n, order='F'))
            prev_split = next_split
        return weights

    def init_rand_weights(self, layer_sizes):
        params = []
        for i in range(len(layer_sizes) - 1):
            params.append(rand_matrix((layer_sizes[i], layer_sizes[i + 1] + 1)))
        return unroll(params)

    @staticmethod
    def sigmoid(z):
        """Return sigmoid of z."""
        return 1.0 / (1.0 + np.e ** -z)

    @staticmethod
    def with_ones(x):
        """Return mtx with additional first column of ones."""
        return np.hstack((np.ones((len(x), 1)), x))

    def check_layer_sizes(self, input_data):
        num_features = input_data.shape[1]
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
    def _handle_file_or_arr(given, param_name):
        if given is None:
            return None
        if isinstance(given, str):
            data = np.loadtxt(given)
        elif isinstance(given, (list, tuple)):
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
