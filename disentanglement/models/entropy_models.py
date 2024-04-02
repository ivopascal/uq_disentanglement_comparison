from keras.models import Sequential
from keras_uncertainty.models import StochasticClassifier
from keras.layers import Dense, Dropout


import numpy as np


def predictive_entropy(probs, axis=-1, eps=1e-6):
    probs = np.mean(probs, axis=0)
    return -np.sum(probs * np.log(probs + eps), axis=axis)


def expected_entropy(probs, eps=1e-6):
    return -np.mean((probs * np.log(probs + eps)).sum(axis=-1), axis=0)


def mutual_information(probs):
    return predictive_entropy(probs) - expected_entropy(probs)


def train_entropy_dropout_model(x_train, y_train, prob=0.5):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(prob))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    mc_model = StochasticClassifier(model)

    return mc_model
