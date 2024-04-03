from keras import Sequential
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras_uncertainty.layers import StochasticDropout, DropConnectDense


def get_blobs_dropout_architecture(prob=0.5):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))

    return model, "MC-Dropout"


def get_blobs_dropconnect_architecture(prob=0.5):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))

    return model, "MC-DropConnect"


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

    return model, "MC-DropConnect"


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
