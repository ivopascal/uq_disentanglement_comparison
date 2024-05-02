import keras

from disentanglement.datatypes import Dataset
from disentanglement.settings import TEST_MODE

N_CIFAR10_TRAINING_SAMPLES = 50_000
N_CIFAR10_TEST_SAMPLES = 1_000


def get_train_test_cifar_10() -> Dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if TEST_MODE:
        N_CIFAR10_TEST_SAMPLES = 100

    x_test = x_test[:N_CIFAR10_TEST_SAMPLES]
    y_test = y_test[:N_CIFAR10_TEST_SAMPLES]

    return Dataset(x_train, y_train, x_test, y_test)
