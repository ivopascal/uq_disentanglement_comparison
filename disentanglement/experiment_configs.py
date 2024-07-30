from typing import List

from disentanglement.data.blobs import get_train_test_blobs
from disentanglement.data.cifar10 import get_train_test_cifar_10, get_train_test_fashion_mnist
from disentanglement.data.eeg import get_eeg_data, N_EEG_SUBJECTS
from disentanglement.datatypes import UqModel, ExperimentConfig, Dataset
from disentanglement.models.architectures import get_blobs_dropout_architecture, \
    get_blobs_dropconnect_architecture, get_blobs_ensemble_architecture, \
    get_blobs_flipout_architecture, get_cifar10_flipout_architecture, get_cifar10_dropout_architecture, \
    get_cifar10_dropconnect_architecture, \
    get_cifar10_ensemble_architecture, get_eeg_dropout_architecture, get_eeg_dropconnect_architecture, \
    get_eeg_flipout_architecture, get_eeg_ensemble_architecture, get_fashion_mnist_dropconnect_architecture, \
    get_fashion_mnist_dropout_architecture, get_fashion_mnist_flipout_architecture, \
    get_fashion_mnist_ensemble_architecture
from disentanglement.settings import TEST_MODE, N_CIFAR_REPETITIONS


def get_test_mode_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            dataset_name="CIFAR10",
            dataset=get_train_test_cifar_10(),
            models=[UqModel(get_cifar10_dropout_architecture, "MC-Dropout", epochs=100),
                    UqModel(get_cifar10_dropconnect_architecture, "MC-DropConnect", epochs=100),
                    UqModel(get_cifar10_flipout_architecture, "Flipout", epochs=500),
                    UqModel(get_cifar10_ensemble_architecture, "Deep Ensemble", epochs=100),
                    ],
            meta_experiments=["ood_class"
                              ],
        ),
        ExperimentConfig(
            dataset_name="blobs",
            dataset=get_train_test_blobs(),
            models=[
                UqModel(get_blobs_dropout_architecture, "MC-Dropout", epochs=50),
                UqModel(get_blobs_ensemble_architecture, "Deep Ensemble", epochs=50),
                UqModel(get_blobs_dropconnect_architecture, "MC-DropConnect", epochs=50),
                UqModel(get_blobs_flipout_architecture, "Flipout", epochs=500)
            ],
            meta_experiments=[
                "decreasing_dataset",
                "label_noise"
            ]
        ),
        ExperimentConfig(
            dataset_name="Motor Imagery BCI Test",
            dataset=get_eeg_data(subject_id=0),
            models=[
                UqModel(get_eeg_dropout_architecture, "MC-Dropout", epochs=100),
                UqModel(get_eeg_dropconnect_architecture, "MC-DropConnect",
                        epochs=100),
                UqModel(get_eeg_flipout_architecture, "Flipout",
                        epochs=500),
                UqModel(get_eeg_ensemble_architecture, "Deep Ensemble", epochs=100)
            ],
            meta_experiments=["decreasing_dataset", ]
        ),

        get_fashion_mnist_configs(meta_experiments=["decreasing_dataset",
                                                    "label_noise",
                                                    "ood_class"
                                                    ]),
    ]


def get_fashion_mnist_configs(meta_experiments=[], run_index=1) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=f"Fashion MNIST {run_index}",
        dataset=get_train_test_fashion_mnist(),
        models=[UqModel(get_fashion_mnist_dropout_architecture, "MC-Dropout", epochs=100),
                UqModel(get_fashion_mnist_dropconnect_architecture, "MC-DropConnect", epochs=100),
                UqModel(get_fashion_mnist_flipout_architecture, "Flipout", epochs=500),
                UqModel(get_fashion_mnist_ensemble_architecture, "Deep Ensemble", epochs=100),
                ],
        meta_experiments=meta_experiments,
    )


def get_cifar10_config(meta_experiments=[], run_index=1) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=f"CIFAR10 {run_index}",
        dataset=get_train_test_cifar_10(),
        models=[UqModel(get_cifar10_dropout_architecture, "MC-Dropout", epochs=100),
                UqModel(get_cifar10_dropconnect_architecture, "MC-DropConnect", epochs=100),
                UqModel(get_cifar10_flipout_architecture, "Flipout", epochs=500),
                UqModel(get_cifar10_ensemble_architecture, "Deep Ensemble", epochs=100),
                ],
        meta_experiments=meta_experiments,
    )


def get_cifar10_plotting_config(meta_experiments=[]) -> ExperimentConfig:
    models = get_cifar10_config(run_index=0, meta_experiments=meta_experiments).models

    return ExperimentConfig(
        dataset_name="CIFAR10",
        dataset=None,
        models=models,
        meta_experiments=meta_experiments
    )


def get_fashion_mnist_plotting_config(meta_experiments=[]) -> ExperimentConfig:
    models = get_fashion_mnist_configs(run_index=0, meta_experiments=meta_experiments).models

    return ExperimentConfig(
        dataset_name="Fashion MNIST",
        dataset=None,
        models=models,
        meta_experiments=meta_experiments
    )


def get_blobs_config(meta_experiments=[]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name="blobs",
        dataset=get_train_test_blobs(),
        models=[UqModel(get_blobs_dropout_architecture, "MC-Dropout", epochs=50),
                UqModel(get_blobs_ensemble_architecture, "Deep Ensemble", epochs=50),
                UqModel(get_blobs_dropconnect_architecture, "MC-DropConnect", epochs=50),
                UqModel(get_blobs_flipout_architecture, "Flipout", epochs=50)
                ],
        meta_experiments=meta_experiments,
    )


def get_eeg_config_single_subject(subject_id, meta_experiments=[]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=f"Motor Imagery BCI {subject_id}",
        dataset=get_eeg_data(subject_id),
        models=[
            UqModel(get_eeg_dropout_architecture, "MC-Dropout", epochs=100),
            UqModel(get_eeg_dropconnect_architecture, "MC-DropConnect",
                    epochs=100),
            UqModel(get_eeg_flipout_architecture, "Flipout", epochs=500),
            UqModel(get_eeg_ensemble_architecture, "Deep Ensemble", epochs=100)
        ],
        meta_experiments=meta_experiments,
    )


def get_eeg_plotting_config(meta_experiments=[]) -> ExperimentConfig:
    models = get_eeg_config_single_subject(subject_id=0, meta_experiments=meta_experiments).models

    return ExperimentConfig(
        dataset_name="Motor Imagery BCI",
        dataset=None,
        models=models,
        meta_experiments=meta_experiments
    )


def get_eeg_configs(meta_experiments=[]) -> List[ExperimentConfig]:
    return [get_eeg_config_single_subject(subject_id, meta_experiments=meta_experiments) for subject_id in
            range(9)]


def get_experiment_configs() -> List[ExperimentConfig]:
    if TEST_MODE:
        return get_test_mode_configs()

    return [
        # *[get_cifar10_config(meta_experiments=["decreasing_dataset",
        #                                        "label_noise",
        #                                        "ood_class"
        #                                        ], run_index=i) for i in range(N_CIFAR_REPETITIONS)]
        # ,
        # get_cifar10_plotting_config(["decreasing_dataset",
        #                              "label_noise",
        #                              "ood_class"]),
        # get_blobs_config(meta_experiments=["decreasing_dataset",
        #                                    "label_noise",
        #                                    ]),
        # *get_eeg_configs(meta_experiments=["decreasing_dataset",
        #                                    "label_noise",
        #                                    "ood_class"
        #                                    ]),
        # get_eeg_plotting_config(meta_experiments=["decreasing_dataset",
        #                                           "label_noise",
        #                                           "ood_class"
        #                                           ])
        *[get_fashion_mnist_configs(meta_experiments=["decreasing_dataset",
                                                      "label_noise",
                                                      "ood_class"
                                                      ], run_index=i) for i in range(N_CIFAR_REPETITIONS)][::1],
        get_fashion_mnist_plotting_config(["decreasing_dataset",
                                           "label_noise",
                                           "ood_class"])
    ]
