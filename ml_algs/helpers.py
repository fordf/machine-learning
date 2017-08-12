import pickle
import numpy as np

def accuracy(predicted, actual):
    return np.sum(predicted == actual) / len(actual)

def pickleme(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def from_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def visualize(image_array, shape, predicted=None, actual=None, order='F'):
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