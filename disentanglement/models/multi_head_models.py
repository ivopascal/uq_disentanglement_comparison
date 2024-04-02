from keras.models import Model
from keras_uncertainty.layers import SamplingSoftmax
from keras_uncertainty.models import DisentangledStochasticClassifier
from keras.layers import Dense, Input
from keras_uncertainty.utils import numpy_entropy

from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES


def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)


def two_head_model(trunk_model, num_classes=2, num_samples=100):
    inp = Input(shape=(2,))
    x = trunk_model(inp)
    logit_mean = Dense(num_classes, activation="linear")(x)
    logit_var = Dense(num_classes, activation="softplus")(x)
    probs = SamplingSoftmax(num_samples=num_samples, variance_type="linear_std")([logit_mean, logit_var])

    train_model = Model(inp, probs, name="train_model")
    pred_model = Model(inp, [logit_mean, logit_var], name="pred_model")

    train_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return train_model, pred_model


def train_disentangle_model(trunk_model_creator, x_train, y_train, epochs):
    train_model, pred_model = two_head_model(trunk_model_creator())
    train_model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)

    fin_model = DisentangledStochasticClassifier(pred_model, epi_num_samples=NUM_SAMPLES)

    return fin_model


def eval_disentangled_model(disentangled_model, samples):
    pred_mean, pred_ale_std, pred_epi_std = disentangled_model.predict(samples, batch_size=BATCH_SIZE)
    ale_entropy = uncertainty(pred_ale_std)
    epi_entropy = uncertainty(pred_epi_std)

    return ale_entropy, epi_entropy
