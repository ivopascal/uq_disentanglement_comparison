import os
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

from disentanglement.data.datasets import get_datasets
from disentanglement.models.entropy_models import train_entropy_model, expected_entropy, mutual_information
from disentanglement.models.multi_head_models import train_disentangle_model, uncertainty
from disentanglement.settings import NUM_SAMPLES, BATCH_SIZE, TEST_MODE
from disentanglement.util import normalise


def partial_shuffle_dataset(X, y, percentage):
    X_noisy, y_noisy = shuffle(X, y)
    np.random.shuffle(y_noisy[:int(len(y_noisy) * percentage)])
    X_noisy, y_noisy = shuffle(X_noisy, y_noisy)
    return X_noisy, y_noisy


def run_label_noise(dataset, architecture_func, epochs):
    X_train, y_train, X_test, y_test = dataset

    noises = np.arange(0, 1, 0.1)
    if TEST_MODE:
        noises = np.arange(0, 1, 0.4)
    n_classes = len(np.unique(y_train))

    disentangling_accuracies = []
    disentangling_aleatorics = []
    disentangling_epistemics = []

    entropy_accuracies = []
    entropy_aleatorics = []
    entropy_epistemics = []

    for noise in tqdm(noises):
        X_noisy, y_noisy = partial_shuffle_dataset(X_train, y_train, percentage=noise)
        X_test_noisy, y_test_noisy = partial_shuffle_dataset(X_test, y_test, percentage=noise)

        disentangle_model = train_disentangle_model(architecture_func, X_noisy, y_noisy, n_classes,
                                                    epochs=epochs)
        entropy_model = train_entropy_model(architecture_func, X_noisy, y_noisy, n_classes, epochs=epochs)

        pred_mean, pred_ale_std, pred_epi_std = disentangle_model.predict(X_test_noisy, batch_size=BATCH_SIZE)
        entropy_preds = entropy_model.predict_samples(X_test_noisy, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        disentangling_accuracies.append(accuracy_score(y_test_noisy, pred_mean.argmax(axis=1)))
        disentangling_aleatorics.append(uncertainty(pred_ale_std).mean())
        disentangling_epistemics.append(uncertainty(pred_epi_std).mean())

        entropy_accuracies.append(accuracy_score(y_test_noisy, entropy_preds.mean(axis=0).argmax(axis=1)))
        entropy_aleatorics.append(expected_entropy(entropy_preds).mean())
        entropy_epistemics.append(mutual_information(entropy_preds).mean())

    return disentangling_accuracies, disentangling_aleatorics, disentangling_epistemics, entropy_accuracies, entropy_aleatorics, entropy_epistemics, noises


def label_noise(dataset_name, config):
    dataset, architectures, epochs = config
    fig, axes = plt.subplots(2, len(architectures), figsize=(10, 6), sharey=True, sharex=True)

    for arch_idx, architecture_func in enumerate(architectures):
        _, uq_name = architecture_func()
        disentangling_accuracies, disentangling_aleatorics, disentangling_epistemics, entropy_accuracies, entropy_aleatorics, entropy_epistemics, noises = run_label_noise(dataset, architecture_func, epochs)

        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(noises, disentangling_accuracies, label="Gaussian Logits")
        ax_acc.plot(noises, entropy_accuracies, label="Information Theoretic")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Dataset size")
        ax_acc.legend()

        if not os.path.exists("../../figures/noise_dataset/"):
            os.mkdir("../../figures/noise_dataset")

        if TEST_MODE:
            fig_acc.savefig(f"../../figures/noise_datasets/accuracies_{uq_name}_{dataset_name}_TEST.pdf")
        else:
            fig_acc.savefig(f"../../figures/noise_datasets/accuracies_{uq_name}_{dataset_name}.pdf")

        axes[0][arch_idx].plot(noises, normalise(disentangling_epistemics), label="Epistemic")
        axes[0][arch_idx].plot(noises, normalise(disentangling_aleatorics), label="Aleatoric")
        axes[0][arch_idx].set_title(uq_name)

        axes[1][arch_idx].plot(noises,
                               normalise(entropy_epistemics),
                               label="Epistemic")
        axes[1][arch_idx].plot(noises,
                               normalise(entropy_aleatorics),
                               label="Aleatoric")
        axes[1][arch_idx].set_xlabel("Labels shuffled")

        if arch_idx == 0:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty (normalised)")
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty (normalised)")

        if arch_idx == len(architectures) - 1:
            axes[0][arch_idx].legend()

    fig.suptitle(f"Disentangled uncertainty over shuffled labels for {dataset_name}", fontsize=20)
    fig.tight_layout()

    if TEST_MODE:
        fig.savefig(f"../../figures/noise_datasets/disentangled_uncertainties_{dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"../../figures/noise_datasets/disentangled_uncertainties_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()
    label_noise("blobs", get_datasets()["blobs"])
    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")