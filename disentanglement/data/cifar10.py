import keras

from disentanglement.datatypes import Dataset


def get_train_test_cifar_10() -> Dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return Dataset(x_train, y_train, x_test[:1000], y_test[:1000])
