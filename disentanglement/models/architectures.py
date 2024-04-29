import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras_uncertainty.layers import StochasticDropout, DropConnectDense
from keras_uncertainty.models import DeepEnsembleClassifier

from disentanglement.datatypes import UqModel
from disentanglement.settings import NUM_DEEP_ENSEMBLE_ESTIMATORS


class CustomDeepEnsembleClassifier(DeepEnsembleClassifier):
    def __init__(self, model_fn, num_estimators):
        super(CustomDeepEnsembleClassifier, self).__init__(model_fn, num_estimators)

    def predict_samples(self, x, num_samples=-1, batch_size=-1):
        predictions = []
        for estimator in self.test_estimators:
            predictions.append(estimator.predict(x, batch_size=batch_size))
        return np.array(predictions)


def get_blobs_dropout_architecture(prob=0.5):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))

    return model


def get_blobs_ensemble_architecture(prob=0.5):
    def model_fn():
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(2,)))
        model.add(Dropout(prob))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(prob))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    ensemble_model = CustomDeepEnsembleClassifier(model_fn, num_estimators=NUM_DEEP_ENSEMBLE_ESTIMATORS)
    return ensemble_model


def get_cifar10_ensemble_architecture(prob=0.3):
    def model_fn():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(prob))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    ensemble_model = CustomDeepEnsembleClassifier(model_fn, num_estimators=NUM_DEEP_ENSEMBLE_ESTIMATORS)
    return ensemble_model


def get_blobs_dropconnect_architecture(prob=0.5):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))

    return model


def get_cifar10_dropout_architecture(prob=0.3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(StochasticDropout(prob))

    return model


def get_cifar10_dropconnect_architecture(prob=0.3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(DropConnectDense(64, activation='relu', prob=prob))

    return model
