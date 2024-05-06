import numpy as np
from keras.models import Model
from keras.src.callbacks import CSVLogger
from keras_uncertainty.layers import SamplingSoftmax
from keras_uncertainty.models import DisentangledStochasticClassifier, DeepEnsembleClassifier
from keras.layers import Dense, Input
from keras_uncertainty.utils import numpy_entropy
from sklearn.metrics import accuracy_score

from disentanglement.datatypes import Dataset
from disentanglement.logging import TQDM
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE, MODEL_TRAIN_VERBOSE


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
    trunk_model = trunk_model_creator(n_training_samples=x_train.shape[0])

    if isinstance(trunk_model, DeepEnsembleClassifier):
        for i, estimator in enumerate(trunk_model.train_estimators):
            train_model, pred_model = two_head_model(estimator, n_classes)
            csv_logger = CSVLogger('./training_logs.csv', append=True, separator=';')
            train_model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=MODEL_TRAIN_VERBOSE, callbacks=[csv_logger])
            trunk_model.test_estimators[i] = pred_model
            TQDM.update(1)
        trunk_model.outputs = [0, 1]  # This tells Stochastic Model that there's two outputs
        return DisentangledStochasticClassifier(trunk_model, epi_num_samples=trunk_model.num_estimators)

    train_model, pred_model = two_head_model(trunk_model, n_classes)

    if TEST_MODE:
        epochs = 1

    csv_logger = CSVLogger('./training_logs.csv', append=True, separator=';')
    train_model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=MODEL_TRAIN_VERBOSE, callbacks=[csv_logger])
    fin_model = DisentangledStochasticClassifier(pred_model, epi_num_samples=NUM_SAMPLES)
    TQDM.update(1)

    return fin_model


def get_average_uncertainty_gaussian_logits(dataset: Dataset, architecture_func, epochs):
    n_classes = len(np.unique(dataset.y_train))
    gaussian_logits_model = train_gaussian_logits_model(architecture_func, dataset.X_train, dataset.y_train, n_classes,
                                                        epochs=epochs)

    if isinstance(gaussian_logits_model.model, DeepEnsembleClassifier):
        num_samples = gaussian_logits_model.model.num_estimators
    else:
        num_samples = NUM_SAMPLES
    pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(dataset.X_test, batch_size=BATCH_SIZE,
                                                                          num_samples=num_samples)

    return (accuracy_score(dataset.y_test, pred_mean.argmax(axis=1)),
            uncertainty(pred_ale_std).mean(),
            uncertainty(pred_epi_std).mean())
