from typing import List

from disentanglement.data.blobs import get_train_test_blobs
from disentanglement.data.cifar10 import get_train_test_cifar_10
from disentanglement.datatypes import UqModel, ExperimentConfig
from disentanglement.models.architectures import get_blobs_dropout_architecture, get_cifar10_dropout_architecture, \
    get_cifar10_dropconnect_architecture, get_blobs_dropconnect_architecture, get_blobs_ensemble_architecture, \
    get_cifar10_ensemble_architecture


def get_experiment_configs() -> List[ExperimentConfig]:
    return [ExperimentConfig(
        dataset_name="CIFAR10",
        dataset=get_train_test_cifar_10(),
        models=[UqModel(get_cifar10_ensemble_architecture, "Deep Ensemble"),
                UqModel(get_cifar10_dropout_architecture, "MC-DropConnect"),
                UqModel(get_cifar10_dropconnect_architecture, "MC-DropConnect")],
        epochs=100,
    ),
        ExperimentConfig(
            dataset_name="blobs",
            dataset=get_train_test_blobs(),
            models=[UqModel(get_blobs_ensemble_architecture, "Deep Ensemble"),
                    UqModel(get_blobs_dropout_architecture, "MC-DropConnect"),
                    UqModel(get_blobs_dropconnect_architecture, "MC-DropConnect")],
            epochs=50
        )
    ]

