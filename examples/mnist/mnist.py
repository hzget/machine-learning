"""
mnist.py
~~~~~~~~~~

It is used to calculate param in the neural networks.
"""

import numpy as np
import mnist_loader

from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from keras_tuner import HyperModel

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers", 2, 20)):
        model.add(
            layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

class MyHyperModel(HyperModel):
    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(self.classes, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

class mnist_nn(object):

    def __init__(self):
        self.x_train, self.y_train = [], []
        self.x_val, self.y_val = [], []
        self.x_test, self.y_test = [], []
        self.tuner = []
        self.best_hps = []
        self.model = []
        self.history = []
    
    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_val, self.y_val), \
        (self.x_test, self.y_test) = mnist_loader.load_local_mnist()
    
    def train_model(self):
        self.prepare_data()
        self.tuner = create_tuner()
        self.best_hps = get_best_hyperparams(self)
        self.model = self.tuner.hypermodel.build(self.best_hps)
        
        #retrain the model
        self.history = self.model.fit(self.x_train, self.y_train, epochs=3,
                       validation_data=(self.x_val, self.y_val),
                       callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])
    
    def save_model(self):
        self.model.save("my_keras_model.h5")

def load_model():
    try:
        return keras.models.load_model("my_keras_model.h5")
    except (FileNotFoundError, IOError):
        mnn = mnist_nn()
        mnn.train_model()
        mnn.save_model()
        return mnn.model

def create_tuner():
    hypermodel = MyHyperModel(classes=10)
    tuner = RandomSearch(hypermodel, objective="val_accuracy",
                         max_trials=3, overwrite=True, directory="my_dir",
                         project_name="helloworld")
    return tuner
        
def get_best_hyperparams(mnist):
    mnist.tuner.search(mnist.x_train, mnist.y_train, epochs=3,
                       validation_data=(mnist.x_val, mnist.y_val))
    
    return mnist.tuner.get_best_hyperparameters(num_trials=1)[0]
