import numpy as np
from keras.models import Model
from keras_uncertainty.layers import SamplingSoftmax
from keras_uncertainty.models import DisentangledStochasticClassifier, DeepEnsembleClassifier
from keras.layers import Dense, Input
from keras_uncertainty.utils import numpy_entropy
from sklearn.metrics import accuracy_score

from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE


def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)


def two_head_model(trunk_model, num_classes=2, num_samples=100):
    input_shape = trunk_model.layers[0].input.shape[1:]

    inp = Input(shape=input_shape)
    x = trunk_model(inp)
    logit_mean = Dense(num_classes, activation="linear")(x)
    logit_var = Dense(num_classes, activation="softplus")(x)
    probs = SamplingSoftmax(num_samples=num_samples, variance_type="linear_std")([logit_mean, logit_var])

    train_model = Model(inp, probs, name="train_model")
    pred_model = Model(inp, [logit_mean, logit_var], name="pred_model")

    train_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return train_model, pred_model


def train_gaussian_logits_model(trunk_model_creator, x_train, y_train, n_classes, epochs) -> DisentangledStochasticClassifier:
    trunk_model = trunk_model_creator()

    if isinstance(trunk_model, DeepEnsembleClassifier):
        for i, estimator in enumerate(trunk_model.train_estimators):
            train_model, pred_model = two_head_model(estimator, n_classes)
            train_model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)
            trunk_model.test_estimators[i] = pred_model
        trunk_model.outputs = [0, 1]  # This tells Stochastic Model that there's two outputs
        return DisentangledStochasticClassifier(trunk_model, epi_num_samples=trunk_model.num_estimators)

    train_model, pred_model = two_head_model(trunk_model, n_classes)

    if TEST_MODE:
        epochs = 1

    train_model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)

    fin_model = DisentangledStochasticClassifier(pred_model, epi_num_samples=NUM_SAMPLES)

    return fin_model


def get_average_uncertainty_gaussian_logits(dataset, architecture_func, epochs):
    X_train, y_train, X_test, y_test = dataset
    n_classes = len(np.unique(y_train))
    gaussian_logits_model = train_gaussian_logits_model(architecture_func, X_train, y_train, n_classes,
                                                        epochs=epochs)
    pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(X_test, batch_size=BATCH_SIZE)

    return accuracy_score(y_test, pred_mean.argmax(axis=1)), uncertainty(pred_ale_std).mean(), uncertainty(pred_epi_std).mean()
