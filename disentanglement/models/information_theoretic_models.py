from typing import Union

from keras_uncertainty.models import StochasticClassifier
from keras.layers import Dense


import numpy as np

from disentanglement.models.architectures import CustomDeepEnsembleClassifier
from disentanglement.settings import BATCH_SIZE, TEST_MODE


def predictive_entropy(probs, axis=-1, eps=1e-6) -> np.ndarray:
    probs = np.mean(probs, axis=0)
    return -np.sum(probs * np.log(probs + eps), axis=axis)


def expected_entropy(probs, eps=1e-6)-> np.ndarray:
    return -np.mean((probs * np.log(probs + eps)).sum(axis=-1), axis=0)


def mutual_information(probs) -> np.ndarray:
    return predictive_entropy(probs) - expected_entropy(probs)


def train_it_model(model_creator, x_train, y_train, n_classes, epochs) \
        -> Union[CustomDeepEnsembleClassifier, StochasticClassifier]:
    model = model_creator()

    if isinstance(model, CustomDeepEnsembleClassifier):
        for estimator in model.train_estimators:
            estimator.add(Dense(n_classes, activation="softmax"))
            estimator.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)
        return model

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    if TEST_MODE:
        epochs = 1

    model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)
    mc_model = StochasticClassifier(model)

    return mc_model

