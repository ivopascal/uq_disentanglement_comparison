from keras import Sequential
from keras.src.layers import Dense
from keras_uncertainty.layers import StochasticDropout


def get_blobs_dropout_architecture(prob=0.5):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))

    return model
