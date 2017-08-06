# Machine Learning

## Algorithms

### Neural Network

Python implementation of neural network as specified by Andrew Ng's ML Coursera course.
All arrays should be numpy arrays.

- Module: neural_network

- Functions:

```python
predict(X, weights)
```
Classify X based on trained weights.

```python
cost_and_gradients(weights, X, y, layer_sizes, reg_lambda)
```
Calculate cost and gradients (partial derivs) of weights.

weights: unrolled weights
X: labeled data (without labels)
y: labels
layer_sizes: list of layer sizes NOT INCLUDING bias units
reg_lambda: regularization param (0 - inf)

```python
init_rand_weights(layer_sizes)
```
Returns unrolled small and random weights

```python
reroll(weights, layer_sizes)
```
Convert vectorized weights back into matrixes.

```python
accuracy(predicted_labels, actual_labels)
```
Percentage of data correctly classified.

```python
visualize(image_array, shape, labels=None, y=None, label_dict=None, order='F')
```
Show images within image_array one at a time, display predicted and actual labels
if provided.

Training using scipy's [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html):

    ```python
    options = {
        'disp': True,
        'maxiter': 424
    }
    args = (X, y, layer_sizes, reg_lambda)
    initial_weights = init_rand_weights(layer_sizes)

    min_weights = minimize(
        cost_and_gradients,
        initial_weights,
        args=args
        method=method,
        jac=True,
        options=options
    )


### K-Nearest Neighbors (Knn) Classifier:

- Module: knn
"""
K-Nearest Neigbour algorithm seeks to categorize new data based on the labels of the K closest data points. The distance between two points (p and q) is calculated as:
```python
d = sqrt(sum(p - q) ** 2)
```
"""
- Methods:
    - predict(dataset): Predict class values for unclassified dataset.

- Initialize:
    - `Knn(dataset, k=5)`

The advantages of using  the Knn classifier are:
- A low cost of learning
- Often Successful when data is well mixed.

The disadvantages of using the Knn classifier are:
- Very inefficient for large data sets
- There’s no real model to interpret
- Performance depends on the number of dimensions
- Inconsistent results when there’s ties in votes


### Decision Tree Classifier

- Module: decision_tree

- Initialize:
    - `DecisionTree(min_leaf_size=1, max_depth=3)`

- Methods:
    - ```python
    fit(dataset, classes)
    ```
    Build a decision tree off of data. Dataset should be a list of rows, with the final element of each row being the class value.

    - ```python
    predict(dataset)
    ```
    Predict class values for unclassified dataset, using fitted tree.

    - ```python
    cross_validate(self, dataset, train_split=.7, label_col=-1)
    ```
    Splits a classified dataset in two, one to build the decision tree, the other to predict with. Returns the percentage of predicted labels that match actual labels.


### K-Means Clustering

- Module: k_means

- Initialize:
    - ```python
    clf = KMeansClassifier(max_iter=None, min_step='auto')
    ```
        - max_iter: The number of iterations before the algorithm stops
        - min_step: The smallest difference in distance between an old centroid
                  and a new centroid before the algorithm stops. If 'auto',
                  min_step is calculated to be 1/1000th of largest first step.

- Methods:
    - ```python
    clf.fit(data, k=None, init_centroids=None)
    ```
    Find centroids of clusters.
        - data: 2d numpy array

        k or init_centroids required
        - k: number of centroids to randomly initialize
        - init_centroids: starting locations of centroids

    - ```python
    clf.predict(data)
    ```
    Return predicted classes for given data.


## Notebooks

### Analyzing a Dataset

[The dataset](https://archive.ics.uci.edu/ml/datasets/Housing)

#### What's in the data

It concerns housing values in the suburbs of Boston, back in 1978.
For each tract of land, there is data on:

* Crime rate
* Proportion of non-retail business (by acreage)
* Whether tract bounds the Charles River
* Nitric Oxide Concetrations
* Age of housing
* Distance to employment centers
* Pupil-teacher ratio
* Proportion of black citizens
* Median value of occupied homes

and more!!

I aimed to visualize relationships in the data and find where correlations might exist.

### Working With Imperfect Data

[The dataset](https://data.cityofnewyork.us/Education/School-Demographics-and-Accountability-Snapshot-20/ihfw-zy9j)

Annual school accounts of NYC public school student populations served by grade, special programs, ethnicity, gender and Title I funded programs. It has 10000 rows and 38 columns, many of which I don't know the meaning of. For each school, there is data for:

* Total enrollment
* Enrollment for each grade
* Number/Percentage for Asian, Black, Hispanic, and White racial groups.
* Number/Percentage for english language learners
* Number/Percentage for male/female

and more!!!

No cleaning of the csv file itself was necessary, it could all be done with pandas.