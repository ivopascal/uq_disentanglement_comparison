import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from disentanglement.data.datasets import get_datasets
from disentanglement.models.entropy_models import mutual_information, expected_entropy, \
    train_entropy_model
from disentanglement.models.multi_head_models import uncertainty, \
    train_disentangle_model
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE
from disentanglement.util import normalise


def run_decreasing_dataset(dataset, architecture_func, epochs):
    disentangling_accuracies = []
    disentangling_aleatorics = []
    disentangling_epistemics = []

    entropy_accuracies = []
    entropy_aleatorics = []
    entropy_epistemics = []
    X_train, y_train, X_test, y_test = dataset
    max_train_samples = y_train.shape[0]
    n_classes = len(np.unique(y_train))

    num_dataset_sizes = 20
    if TEST_MODE:
        num_dataset_sizes = 3

    dataset_sizes = np.logspace(start=1, stop=np.log2(max_train_samples), base=2, num=num_dataset_sizes)

    for dataset_size in tqdm(dataset_sizes):
        X_train_sub, y_train_sub = X_train[:int(dataset_size)], y_train[:int(dataset_size)]

        disentangle_model = train_disentangle_model(architecture_func, X_train_sub, y_train_sub, n_classes,
                                                    epochs=epochs)
        entropy_model = train_entropy_model(architecture_func, X_train_sub, y_train_sub, n_classes, epochs=epochs)

        pred_mean, pred_ale_std, pred_epi_std = disentangle_model.predict(X_test, batch_size=BATCH_SIZE)
        entropy_preds = entropy_model.predict_samples(X_test, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        disentangling_accuracies.append(accuracy_score(y_test, pred_mean.argmax(axis=1)))
        disentangling_aleatorics.append(uncertainty(pred_ale_std).mean())
        disentangling_epistemics.append(uncertainty(pred_epi_std).mean())

        entropy_accuracies.append(accuracy_score(y_test, entropy_preds.mean(axis=0).argmax(axis=1)))
        entropy_aleatorics.append(expected_entropy(entropy_preds).mean())
        entropy_epistemics.append(mutual_information(entropy_preds).mean())

    return disentangling_accuracies, disentangling_aleatorics, disentangling_epistemics, entropy_accuracies, entropy_aleatorics, entropy_epistemics, dataset_sizes


def plot_decreasing_dataset(dataset_name, config):
    dataset, architectures, epochs = config

    fig, axes = plt.subplots(2, len(architectures), figsize=(10, 6), sharey=True, sharex=True)
    for arch_idx, architecture_func in enumerate(architectures):
        _, uq_name = architecture_func()
        disentangling_accuracies, disentangling_aleatorics, disentangling_epistemics, entropy_accuracies, entropy_aleatorics, entropy_epistemics, dataset_sizes = run_decreasing_dataset(
            dataset, architecture_func, epochs)

        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(dataset_sizes, disentangling_accuracies, label="Gaussian Logits")
        ax_acc.plot(dataset_sizes, entropy_accuracies, label="Information Theoretic")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Dataset size")
        ax_acc.legend()

        if not os.path.exists("../../figures/decreasing_dataset/"):
            os.mkdir("../../figures/decreasing_dataset/")

        if TEST_MODE:
            fig_acc.savefig(f"../../figures/decreasing_dataset/accuracies_{uq_name}_{dataset_name}_TEST.pdf")
        else:
            fig_acc.savefig(f"../../figures/decreasing_dataset/accuracies_{uq_name}_{dataset_name}.pdf")

        axes[0][arch_idx].plot(dataset_sizes, normalise(disentangling_epistemics), label="Epistemic")
        axes[0][arch_idx].plot(dataset_sizes, normalise(disentangling_aleatorics), label="Aleatoric")
        axes[0][arch_idx].set_title(uq_name)

        axes[1][arch_idx].plot(dataset_sizes,
                               normalise(entropy_epistemics),
                               label="Epistemic")
        axes[1][arch_idx].plot(dataset_sizes,
                               normalise(entropy_aleatorics),
                               label="Aleatoric")

        if arch_idx == 0:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty (normalised)")
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty (normalised)")

        if arch_idx == len(architectures) - 1:
            axes[0][arch_idx].legend()
            axes[0][arch_idx].set_xlabel("Dataset size")
            axes[1][arch_idx].set_xlabel("Dataset size")

    fig.suptitle(f"Disentangled uncertainty over decreasing dataset sizes for {dataset_name}", fontsize=20)
    fig.tight_layout()

    if TEST_MODE:
        fig.savefig(f"../../figures/decreasing_dataset/disentangled_uncertainties_{dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"../../figures/decreasing_dataset/disentangled_uncertainties_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()

    for dataset_name, config in get_datasets().items():
        if dataset_name == "blobs":
            plot_decreasing_dataset(dataset_name, config)

    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
