from disentanglement.data.blobs import get_train_test_blobs
from disentanglement.data.cifar10 import get_train_test_cifar_10
from disentanglement.models.architectures import get_blobs_dropout_architecture


def get_datasets():
    return {# get_train_test_cifar_10(): "CIFAR10",
            "blobs": (get_train_test_blobs(), get_blobs_dropout_architecture, 50),
            }