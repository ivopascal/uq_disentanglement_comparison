from disentanglement.data.blobs import get_train_test_blobs
from disentanglement.data.cifar10 import get_train_test_cifar_10
from disentanglement.models.architectures import get_blobs_dropout_architecture, get_cifar10_dropout_architecture, get_cifar10_dropconnect_architecture, get_blobs_dropconnect_architecture


def get_datasets():
    return {"CIFAR10": (get_train_test_cifar_10(),
                        [get_cifar10_dropout_architecture, get_cifar10_dropconnect_architecture],
                        100),
            "blobs": (get_train_test_blobs(),
                      [get_blobs_dropout_architecture, get_blobs_dropconnect_architecture],
                      50),
            }
