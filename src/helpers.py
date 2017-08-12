import pickle
import numpy as np
from math import ceil

def train_val_test_split(data, y=None, train=.7, val=.15, test=.15):
    assert(train + val + test == 1)
    if y is not None:
        data = np.hstack((data, y.reshape(len(y), 1)))
    data = data.copy()
    m = len(data)
    np.random.shuffle(data)
    splits = [0]
    for i, split in enumerate((train, val, test)):
        splits.append(splits[i] + ceil(split * m))
    split_data = [data[splits[i]: splits[i + 1]] for i in range(3)]
    return tuple(split_data)


def accuracy(predicted, actual):
    return np.sum(predicted == actual) / len(actual)

def pickleme(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def from_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def visualize(image_array, shape, predicted=None, actual=None, labelmap=None, order='F'):
    """
    If your data just so happens to be iamges, show them one at a time,
    optionally printing predicted and/or actual label for each.

    visualize(image_array, shape, predicted=None, actual=None, order='F')
    """
    from matplotlib import pyplot as plt
    from random import sample
    image_array = image_array.copy()
    try:
        for i in sample(list(range(len(image_array))), len(image_array)):
            if predicted is not None:
                pred = predicted[i]
                if labelmap is not None:
                    pred = labelmap[pred]
                print('NN predicted: {}'.format(pred))
            if actual is not None:
                act = actual[i]
                if labelmap is not None:
                    act = labelmap[act]
                print('actual: {}'.format(act))
            plt.imshow(image_array[i].reshape(shape, order=order))
            plt.show()
            inp = input('q to quit, anything else to continue\n')
            if inp in ['q', 'quit', 'exit']:
                break
    except KeyboardInterrupt:
        plt.close()