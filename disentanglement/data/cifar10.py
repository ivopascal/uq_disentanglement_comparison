import keras


def get_train_test_cifar_10():
    return keras.datasets.cifar10.load_data()
