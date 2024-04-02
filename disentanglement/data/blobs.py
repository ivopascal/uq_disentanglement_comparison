from sklearn.datasets import make_blobs


def get_train_test_blobs():
    X_train, y_train = make_blobs(n_samples=200, n_features=2, centers=[[-1.5, 1.5], [0, -1.5]],
                                  random_state=0)
    X_test, y_test = make_blobs(n_samples=200, n_features=2, centers=[[-1.5, 1.5], [0, -1.5]], random_state=1)
    return X_train, y_train, X_test, y_test
