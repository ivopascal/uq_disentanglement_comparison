import keras
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from disentanglement.datatypes import Dataset
from disentanglement.settings import TEST_MODE, N_CIFAR10_TEST_SAMPLES

N_CIFAR10_TRAINING_SAMPLES = 50_000


def get_train_test_cifar_10() -> Dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if TEST_MODE:
        n_test_samples = 100
    else:
        n_test_samples = N_CIFAR10_TEST_SAMPLES

    x_test = x_test[:n_test_samples]
    y_test = y_test[:n_test_samples]

    return Dataset(x_train, y_train, x_test, y_test)


def get_train_test_fashion_mnist() -> Dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if TEST_MODE:
        n_test_samples = 100
    else:
        n_test_samples = N_CIFAR10_TEST_SAMPLES

    x_test = x_test[:n_test_samples]
    y_test = y_test[:n_test_samples]

    return Dataset(x_train, y_train, x_test, y_test)


def get_train_test_wine() -> Dataset:
    X, y = load_wine(return_X_y=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return Dataset(X_train, y_train, X_test, y_test)

