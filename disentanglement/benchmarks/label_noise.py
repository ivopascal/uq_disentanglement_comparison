import os
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

from disentanglement.data.datasets import get_datasets
from disentanglement.datatypes import UncertaintyResults
from disentanglement.models.information_theoretic_models import train_it_model, expected_entropy, mutual_information
from disentanglement.models.gaussian_logits_models import train_gaussian_logits_model, uncertainty
from disentanglement.settings import NUM_SAMPLES, BATCH_SIZE, TEST_MODE, FIGURE_FOLDER
from disentanglement.util import normalise


def partial_shuffle_dataset(X, y, percentage):
    X_noisy, y_noisy = shuffle(X, y)
    np.random.shuffle(y_noisy[:int(len(y_noisy) * percentage)])
    X_noisy, y_noisy = shuffle(X_noisy, y_noisy)
    return X_noisy, y_noisy


def run_label_noise(dataset, architecture_func, epochs):
    X_train, y_train, X_test, y_test = dataset

    noises = np.arange(0, 1.05, 0.05)
    if TEST_MODE:
        noises = np.arange(0, 1, 0.4)
        epochs = 2
    n_classes = len(np.unique(y_train))

    gaussian_logits_results = UncertaintyResults()
    it_results = UncertaintyResults()

    for noise in tqdm(noises):
        X_noisy, y_noisy = partial_shuffle_dataset(X_train, y_train, percentage=noise)
        X_test_noisy, y_test_noisy = partial_shuffle_dataset(X_test, y_test, percentage=noise)

        gaussian_logits_model = train_gaussian_logits_model(architecture_func, X_noisy, y_noisy, n_classes,
                                                        epochs=epochs)
        pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(X_test_noisy, batch_size=BATCH_SIZE)
        gaussian_logits_results.append_values(accuracy_score(y_test, pred_mean.argmax(axis=1)),
                                              uncertainty(pred_ale_std).mean(), uncertainty(pred_epi_std).mean(),
                                              noise)

        it_model = train_it_model(architecture_func, X_noisy, y_noisy, n_classes, epochs=epochs)
        it_preds = it_model.predict_samples(X_test_noisy, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        it_results.append_values(accuracy_score(y_test, it_preds.mean(axis=0).argmax(axis=1)),
                                 expected_entropy(it_preds).mean(), mutual_information(it_preds).mean(), noise)

    return gaussian_logits_results, it_results


def label_noise(dataset_name, config):
    dataset, architectures, epochs = config
    fig, axes = plt.subplots(2, len(architectures), figsize=(10, 6), sharey=True, sharex=True)

    for arch_idx, architecture in enumerate(architectures):
        uq_name = architecture.uq_name
        gaussian_logits_results, it_results = run_label_noise(dataset, architecture.model_function, epochs)
        noises = gaussian_logits_results.changed_parameter_values
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(noises, gaussian_logits_results.accuracies, label="Gaussian Logits")
        ax_acc.plot(noises, it_results.accuracies, label="Information Theoretic")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Labels shuffled")
        ax_acc.legend()

        if not os.path.exists(f"{FIGURE_FOLDER}/noise_dataset/"):
            os.mkdir(f"{FIGURE_FOLDER}/noise_dataset")

        if TEST_MODE:
            fig_acc.savefig(f"{FIGURE_FOLDER}/noise_datasets/accuracies_{uq_name}_{dataset_name}_TEST.pdf")
        else:
            fig_acc.savefig(f"{FIGURE_FOLDER}/noise_datasets/accuracies_{uq_name}_{dataset_name}.pdf")

        axes[0][arch_idx].plot(noises, normalise(gaussian_logits_results.epistemic_uncertainties), label="Epistemic")
        axes[0][arch_idx].plot(noises, normalise(gaussian_logits_results.aleatoric_uncertainties), label="Aleatoric")
        axes[0][arch_idx].set_title(uq_name)

        axes[1][arch_idx].plot(noises,
                               normalise(it_results.epistemic_uncertainties),
                               label="Epistemic")
        axes[1][arch_idx].plot(noises,
                               normalise(it_results.aleatoric_uncertainties),
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
        fig.savefig(f"{FIGURE_FOLDER}/noise_datasets/disentangled_uncertainties_{dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"{FIGURE_FOLDER}/noise_datasets/disentangled_uncertainties_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()
    label_noise("CIFAR10", get_datasets()["CIFAR10"])
    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")