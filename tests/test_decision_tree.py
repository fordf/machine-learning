"""Test decision tree classifier implementation."""

import os
import pytest
import numpy as np

from ml_algs.decision_tree import DecisionTree

DATASET = [
    [2.771244718, 1.784783929, 0],
    [1.728571309, 1.169761413, 0],
    [3.678319846, 2.81281357, 0],
    [3.961043357, 2.61995032, 0],
    [2.999208922, 2.209014212, 0],
    [7.497545867, 3.162953546, 1],
    [9.00220326, 3.339047188, 1],
    [7.444542326, 0.476683375, 1],
    [10.12493903, 3.234550982, 1],
    [6.642287351, 3.319983761, 1]
]

BEST_SPLIT = (
    0,
    6.642287351,
    [[2.771244718, 1.784783929, 0],
     [1.728571309, 1.169761413, 0],
     [3.678319846, 2.81281357, 0],
     [3.961043357, 2.61995032, 0],
     [2.999208922, 2.209014212, 0]],
    [[7.497545867, 3.162953546, 1],
     [9.00220326, 3.339047188, 1],
     [7.444542326, 0.476683375, 1],
     [10.12493903, 3.234550982, 1],
     [6.642287351, 3.319983761, 1]])


project_root = os.path.dirname(os.path.dirname(__file__))
FLOWERS = np.loadtxt(
    os.path.join(project_root, 'data/flowers_data.csv'),
    delimiter=',',
    skiprows=1,
    usecols=(0, 1, 2, 3, 4),
)


@pytest.fixture
def tree_sepal(dtree):
    """Create a fixture for the last column which contains overlap."""
    sepal = [row[3:] for row in FLOWERS]
    dtree.fit(sepal, (0, 1))
    return dtree


@pytest.fixture
def dtree():
    """Create a dtree to run tests on."""
    return DecisionTree()


@pytest.fixture
def fitted_dtree(dtree):
    """Create a tree from flowers data."""
    dtree.fit(FLOWERS, (0, 1))
    return dtree


def test_find_best_split(dtree):
    """Test find_best_split on small dataset."""
    assert dtree._find_best_split(DATASET) == BEST_SPLIT


def test_find_best_split_value(dtree):
    """Test find_best_split_gets_best_value."""
    assert dtree._find_best_split(FLOWERS)[1] == 3.0


def test_dtree_fit_has_root(fitted_dtree):
    """Test fit method creates a tree."""
    assert fitted_dtree.root


def test_clf_fit(fitted_dtree):
    """Test fit method creates a split."""
    assert fitted_dtree.root.split_value == 3.0


def test_clf_fit_chilrend(fitted_dtree):
    """Test class on the left is 1."""
    assert fitted_dtree.root.left == 0.0
    assert fitted_dtree.root.right == 1.0
