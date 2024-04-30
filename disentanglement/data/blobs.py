from sklearn.datasets import make_blobs

from disentanglement.datatypes import Dataset


def get_train_test_blobs() -> Dataset:
    X_train, y_train = make_blobs(n_samples=500, n_features=2, centers=[[-1.5, 1.5], [0, -1.5]],
                                  random_state=0)
    X_test, y_test = make_blobs(n_samples=200, n_features=2, centers=[[-1.5, 1.5], [0, -1.5]], random_state=1)
    return Dataset(X_train, y_train, X_test, y_test)
