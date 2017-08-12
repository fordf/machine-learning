import os
import random
import pytest
import numpy as np
from faker import Faker
from collections import namedtuple

from ml_algs.neural_network import NeuralNetwork, LabelHandler

fake = Faker()

project_root = os.path.dirname(os.path.dirname(__file__))
FLOWERS = np.loadtxt(
    os.path.join(project_root, 'data/flowers_data.csv'),
    delimiter=',',
    skiprows=1,
    usecols=(0, 1, 2, 3, 4)
)
X = FLOWERS[:, range(4)]
y_int = FLOWERS[:, -1]
y_string = np.loadtxt(
    os.path.join(project_root, 'data/flowers_data.csv'),
    delimiter=',',
    skiprows=1,
    usecols=(5),
    dtype=str)

NN = namedtuple(
    'NN', (
        'instance',
        'X',
        'y',
        'classes',
        'layer_sizes',
        'num_features',
        'num_input',
    )
)


@pytest.fixture
def nn():
    nn = NeuralNetwork()
    return nn


@pytest.fixture
def simplenn(nn):
    nn.fit(X=[[0],[1]], y=[1, 0], layer_sizes=[1,2])
    return nn


@pytest.fixture
def fakenn(nn):

    num_input, num_features = random.sample(range(100), 2)

    fakex = np.random.rand(num_input, num_features) * 5
    fakex[num_input // 2:] *= -1

    fakey = (np.random.rand(num_input) * random.randint(1,4)).astype(int)
    classes = fake.words(fakey.max() + 1)

    layer_sizes = [num_features] + random.sample(range(10), random.randint(0,3)) + [len(classes)]
    nn.fit(fakex, fakey, layer_sizes, options={'maxiter': 50})

    return NN(
        nn,
        fakex,
        fakey,
        classes,
        layer_sizes,
        num_features,
        num_input,
    )


@pytest.fixture
def weighted_nn(nn):
    nn.load_weights('/'.join((project_root, 'data/flower_weights')))
    return nn


"""
methods:

['__init__',
 'accuracy',
 'assert_have_required',
 'check_layer_sizes',
 'cost_and_gradients',
 'fit',
 'forward_prop',
 'from_pickle',
 'init_rand_weights',
 'load_weights',
 'pickleme',
 'predict',
 'save_weights',
 'sigmoid_grad',
 'verbose_cost_gradients']


X:
- 4 features
- 100 datapoints

y:
- 2 classes
- [0, 'setosa'], [1, 'versicolor']
"""

LAYERSIZES = (
    ((4,3,2), None),
    ((4,2), None),
    ((4,30,12,2), None),
    ((4), ValueError),
    ((4,1), ValueError),
    ((3,2), ValueError),
)


@pytest.mark.parametrize('layer_sizes, error', LAYERSIZES)
def test_check_layer_sizes(layer_sizes, error, nn):
    nn.layer_sizes = layer_sizes
    nn.label_handler = LabelHandler()
    nn.label_handler.fit
    if error:
        with pytest.raises(error):
            nn.check_layer_sizes(X)
    else:
        assert nn.check_layer_sizes(X) is None


def test_assert_have_required(nn):
    assert nn.assert_have_required((('x','x'), ('yes', 'ok'))) is None
    with pytest.raises(ValueError):
        nn.assert_have_required((('x', None), 'yes', 'ok'))


def test_accuracy(nn):
    assert nn.accuracy(np.array([0,1]), np.array([0,1])) == 1.0
    assert nn.accuracy(np.array([0,1]), np.array([1,0])) == 0.0
