from disentanglement.data.blobs import get_train_test_blobs
from disentanglement.data.cifar10 import get_train_test_cifar_10
from disentanglement.datatypes import UqModel
from disentanglement.models.architectures import get_blobs_dropout_architecture, get_cifar10_dropout_architecture, \
    get_cifar10_dropconnect_architecture, get_blobs_dropconnect_architecture, get_blobs_ensemble_architecture, \
    get_cifar10_ensemble_architecture


def get_datasets():
    return {"CIFAR10": (get_train_test_cifar_10(),
                        [UqModel(get_cifar10_ensemble_architecture, "Deep Ensemble"),
                         UqModel(get_cifar10_dropout_architecture, "MC-DropConnect"),
                         UqModel(get_cifar10_dropconnect_architecture, "MC-DropConnect")],
                        100),
            "blobs": (get_train_test_blobs(),
                      [UqModel(get_blobs_ensemble_architecture, "Deep Ensemble"),
                       UqModel(get_blobs_dropout_architecture, "MC-DropConnect"),
                       UqModel(get_blobs_dropconnect_architecture, "MC-DropConnect")],
                      50),
            }
