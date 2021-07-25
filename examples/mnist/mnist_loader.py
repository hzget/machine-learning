"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

from tensorflow import keras
import numpy as np
import joblib
    
def save_local_mnist(mnist):
    joblib.dump(mnist, "my_mnist.pkl")
    return mnist

def load_local_mnist():
    try:
        return joblib.load("my_mnist.pkl")
    except FileNotFoundError:
        return save_local_mnist(load_keras_mnist())
    
def load_keras_mnist():
    """Return the MNIST data as a tuple containing the training data,
    load data from keras build-in dataset and then convert 28x28 
    """
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
