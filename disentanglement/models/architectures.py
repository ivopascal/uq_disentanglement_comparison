import numpy as np
from keras import Sequential, Input, Model
from keras.src.constraints import max_norm
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, \
    AveragePooling2D
from keras_uncertainty.losses import regression_gaussian_nll_loss
from tensorflow.keras import backend as K

from keras_uncertainty.layers import StochasticDropout, DropConnectDense, FlipoutDense
from keras_uncertainty.models import DeepEnsembleClassifier, DeepEnsembleRegressor

from disentanglement.data.blobs import N_BLOBS_TRAINING_SAMPLES
from disentanglement.settings import NUM_DEEP_ENSEMBLE_ESTIMATORS


class CustomDeepEnsembleClassifier(DeepEnsembleClassifier):
    def __init__(self, model_fn, num_estimators):
        super(CustomDeepEnsembleClassifier, self).__init__(model_fn, num_estimators)
        self.estimator_to_use = 0

    def predict_samples(self, x, num_samples=-1, batch_size=-1):
        predictions = []
        for estimator in self.test_estimators:
            predictions.append(estimator.predict(x, batch_size=batch_size, verbose=0))
        return np.array(predictions)

    def predict(self, X, batch_size=32, num_ensembles=None, return_std=False, **kwargs):
        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        prediction = self.test_estimators[self.estimator_to_use % self.num_estimators].predict(X, batch_size=batch_size,
                                                                                               **kwargs)
        self.estimator_to_use += 1

        return prediction


def get_blobs_dropout_architecture(prob=0.2, **_):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))
    return model


def get_blobs_flipout_architecture(n_training_samples=N_BLOBS_TRAINING_SAMPLES):
    num_batches = n_training_samples / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0,
        'prior_sigma_2': 2.0,
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, **prior_params, activation="relu", input_shape=(2,)))
    model.add(FlipoutDense(32, kl_weight, **prior_params, activation="relu"))

    return model


def get_blobs_ensemble_architecture(prob=0.2, **_):
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


def get_blobs_dropconnect_architecture(prob=0.2, **_):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))

    return model


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def get_eeg_convolutional_blocks(channels=22, samples=513):
    # This is based off ShallowConvNet, but we can add a (Bayesian) FC layer after the conv block
    model = Sequential()
    model.add(Conv2D(40, (1, 13),
                     input_shape=(channels, samples, 1),
                     kernel_constraint=max_norm(2., axis=(0, 1, 2))))
    model.add(Conv2D(40, (channels, 1), use_bias=False,
                     kernel_constraint=max_norm(2., axis=(0, 1, 2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
    model.add(Activation(square))
    model.add(AveragePooling2D(pool_size=(1, 35), strides=(1, 7)))
    model.add(Activation(log))
    model.add(Flatten())

    return model


def get_cifar10_convolutional_blocks(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())

    return model


def get_auto_mpg_backbone(input_shape=9):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Dense(16, activation='relu'))

    return model


def get_auto_mpg_dropout_architecture(**_):
    return get_dropout_from_backbone(backbone_func=get_auto_mpg_backbone, hidden_size=16)


def get_auto_mpg_ensemble_architecture(**_):
    return get_regression_ensemble_from_backbone(backbone_func=get_auto_mpg_backbone, hidden_size=16)


def get_auto_mpg_flipout_architecture(n_training_samples):
    return get_flipout_from_backbone(backbone_func=get_auto_mpg_backbone, hidden_size=16,
                                     n_training_samples=n_training_samples)


def get_auto_mpg_dropconnect_architecture(**_):
    return get_dropconnect_from_backbone(backbone_func=get_auto_mpg_backbone, hidden_size=16)


def get_fashion_mnist_convolutional_blocks():
    return get_cifar10_convolutional_blocks(input_shape=(28, 28, 1))


def get_wine_backbone(input_shape=13):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Dense(32))
    model.add(Dense(32))

    return model


def get_dropout_from_backbone(backbone_func, prob=0.3, hidden_size=64, **_):
    model = backbone_func()
    model.add(Dense(hidden_size, activation='relu'))
    model.add(StochasticDropout(prob))

    return model


def get_ensemble_from_backbone(backbone_func, prob=0.3, hidden_size=64, **_):
    def model_fn():
        model = backbone_func()
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dropout(prob))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    ensemble_model = CustomDeepEnsembleClassifier(model_fn, num_estimators=NUM_DEEP_ENSEMBLE_ESTIMATORS)
    return ensemble_model


def get_regression_ensemble_from_backbone(backbone_func, prob=0.3, hidden_size=64, **_):
    def model_fn():
        backbone = backbone_func()
        hidden_representation = backbone(backbone.inputs)
        mean = Dense(1, activation="linear")(hidden_representation)
        var = Dense(1, activation="softplus")(hidden_representation)

        train_model = Model(backbone.inputs, mean)
        pred_model = Model(backbone.inputs, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

        return train_model, pred_model

    ensemble_model = DeepEnsembleRegressor(model_fn, num_estimators=NUM_DEEP_ENSEMBLE_ESTIMATORS)
    return ensemble_model


def get_flipout_from_backbone(backbone_func, n_training_samples, hidden_size=64, **_):
    num_batches = n_training_samples / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0,
        'prior_sigma_2': 2.0,
        'prior_pi': 0.5
    }

    model = backbone_func()
    model.add(FlipoutDense(hidden_size, kl_weight, **prior_params, activation="relu", ))

    return model


def get_dropconnect_from_backbone(backbone_func, hidden_size=64, prob=0.3, **_):
    model = backbone_func()
    model.add(DropConnectDense(hidden_size, activation='relu', prob=prob))

    return model


def get_cifar10_flipout_architecture(n_training_samples):
    return get_flipout_from_backbone(get_cifar10_convolutional_blocks, hidden_size=64,
                                     n_training_samples=n_training_samples)


def get_cifar10_dropout_architecture(**_):
    return get_dropout_from_backbone(get_cifar10_convolutional_blocks, hidden_size=64)


def get_cifar10_dropconnect_architecture(**_):
    return get_dropconnect_from_backbone(get_cifar10_convolutional_blocks, hidden_size=64)


def get_cifar10_ensemble_architecture(**_):
    return get_ensemble_from_backbone(get_cifar10_convolutional_blocks, hidden_size=64)


def get_fashion_mnist_flipout_architecture(n_training_samples):
    return get_flipout_from_backbone(get_fashion_mnist_convolutional_blocks, hidden_size=64,
                                     n_training_samples=n_training_samples)


def get_fashion_mnist_dropout_architecture(**_):
    return get_dropout_from_backbone(get_fashion_mnist_convolutional_blocks, hidden_size=64)


def get_fashion_mnist_dropconnect_architecture(**_):
    return get_dropconnect_from_backbone(get_fashion_mnist_convolutional_blocks, hidden_size=64)


def get_fashion_mnist_ensemble_architecture(**_):
    return get_ensemble_from_backbone(get_fashion_mnist_convolutional_blocks, hidden_size=64)


def get_wine_flipout_architecture(n_training_samples):
    return get_flipout_from_backbone(get_wine_backbone, hidden_size=16,
                                     n_training_samples=n_training_samples)


def get_wine_dropout_architecture(**_):
    return get_dropout_from_backbone(get_wine_backbone, hidden_size=16)


def get_wine_dropconnect_architecture(**_):
    return get_dropconnect_from_backbone(get_wine_backbone, hidden_size=16)


def get_wine_ensemble_architecture(**_):
    return get_ensemble_from_backbone(get_wine_backbone, hidden_size=16)


def get_eeg_flipout_architecture(n_training_samples, **_):
    return get_flipout_from_backbone(get_eeg_convolutional_blocks, hidden_size=32,
                                     n_training_samples=n_training_samples)


def get_eeg_dropout_architecture(**_):
    return get_dropout_from_backbone(get_eeg_convolutional_blocks, hidden_size=32)


def get_eeg_dropconnect_architecture(**_):
    return get_dropconnect_from_backbone(get_eeg_convolutional_blocks, hidden_size=32)


def get_eeg_ensemble_architecture(**_):
    return get_ensemble_from_backbone(get_eeg_convolutional_blocks, hidden_size=32)
