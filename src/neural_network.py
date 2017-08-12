import os
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
        self.X_train = None
        self.y_train = None
        self.label_handler = None

    def predict(self, x):
        """Predict labels for dataset x."""
        if self.weights is None:
            raise ValueError('Load weights into NN or fit before predicting.')
        output = self.forward_prop(x, self.weights)[-1]
        return self.label_handler.inverse_transform(output.argmax(axis=1))

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

        X:
            - shape = (num input, num features)

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
        self.prefit(X, y, layer_sizes, reg_lambda, options, method)

        cost_grad_funcs = {
            'default': self.cost_and_gradients,
            'verbose': self.verbose_cost_gradients,
        }
        if isinstance(cost_grad_func, str):
            cost_grad_func = cost_grad_funcs[cost_grad_func]
            args = (self.X_train, self.y_train, self.layer_sizes, self.reg_lambda)

        required = (('X', self.X_train),
                    ('y', self.y_train),
                    ('layer_sizes', self.layer_sizes),
                    ('cost_grad_func', cost_grad_func))
        self.assert_have_required(required)
        self.check_layer_sizes(self.X_train)

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

    def prefit(self,
               X=None,
               y=None,
               layer_sizes=None,
               reg_lambda=0,
               options=None,
               method='CG'):
        """Set up any of the nitty gritty hyperparams prior to finding minimum weights."""

        valid_methods = 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg'
        cost_grad_funcs = {
            'default': self.cost_and_gradients,
            'verbose': self.verbose_cost_gradients,
        }

        if X is not None:
            self.X_train = self._handle_file_or_arr(X, 'X')
        if y is not None:
            self.label_handler = LabelHandler()
            y = self._handle_file_or_arr(y, 'y')
            self.y_train = self.label_handler.fit_transform(y)
        if method not in valid_methods:
            raise ValueError("valid methods: {}".format(valid_methods))
        self.reg_lambda = reg_lambda or self.reg_lambda
        self.layer_sizes = layer_sizes or self.layer_sizes

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

    def cost_and_gradients(self, unrolled_weights, x, y, layer_sizes, reg_lambda):
        """Compute cost of dataset with unrolled_weights."""
        m = x.shape[0]
        weights = self.reroll(unrolled_weights, layer_sizes)
        actvs = self.forward_prop(x, weights)

        cost = 1 / m * np.sum(-y * np.log(actvs[-1]) - (1-y) * np.log(1 - actvs[-1]))
        cost += reg_lambda / 2 / m * sum(np.sum(theta[:, 1:] ** 2) for theta in weights)

        partial_d = actvs[-1] - y
        grads = []
        for i in range(1, len(layer_sizes)):
            prev_activation = actvs[-i - 1]
            wlayer = weights[-i]
            grad = partial_d.T.dot(self.with_ones(prev_activation)) / m
            grad += reg_lambda / m * np.hstack((np.zeros((len(wlayer), 1)), wlayer[:, 1:]))
            grads.append(grad)
            if i < len(layer_sizes) - 1:
                partial_d = partial_d.dot(wlayer[:, 1:])
                partial_d *= prev_activation * (1 - prev_activation)
        return cost, self.unroll(grads[::-1])

    def verbose_cost_gradients(self, unrolled_weights, x, y, layer_sizes, reg_lambda):
        """Call cost_and_gradients, print cost."""
        cost, grads = self.cost_and_gradients(unrolled_weights, x, y, layer_sizes, reg_lambda)
        print(cost)
        return cost, grads

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
        num_wlayers = len(layer_sizes) - 1
        splits = [0] + [
            (1 + layer_sizes[i]) * layer_sizes[i + 1] for i in range(num_wlayers)
        ]
        for i in range(num_wlayers):
            wlayer = nn_params[splits[i]:splits[i] + splits[i + 1]]
            thetas.append(wlayer.reshape(layer_sizes[i] + 1, layer_sizes[i + 1]).T)
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
