import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm

from disentanglement.data.datasets import get_datasets
from disentanglement.datatypes import UncertaintyResults
from disentanglement.models.gaussian_logits_models import get_average_uncertainty_gaussian_logits
from disentanglement.models.information_theoretic_models import get_average_uncertainty_it
from disentanglement.settings import TEST_MODE, FIGURE_FOLDER
from disentanglement.util import normalise


def run_decreasing_dataset(dataset, model_function, epochs):
    gl_results = UncertaintyResults()
    it_results = UncertaintyResults()

    X_train, y_train, X_test, y_test = dataset

    num_dataset_sizes = 20
    if TEST_MODE:
        num_dataset_sizes = 3
        epochs = 2

    dataset_sizes = np.logspace(start=0.01, stop=1, base=2, num=num_dataset_sizes) - 1

    X_train, y_train = shuffle(X_train, y_train)
    for dataset_size in tqdm(dataset_sizes):
        X_train_subs = []
        y_train_subs = []
        for y_value in np.unique(y_train):
            n_samples_per_class = int((y_train == y_value).sum() * dataset_size)
            if n_samples_per_class == 0:
                n_samples_per_class = 1
            X_train_subs.append(X_train[y_train == y_value][:n_samples_per_class])
            y_train_subs.append(y_train[y_train == y_value][:n_samples_per_class])

        X_train_sub = np.concatenate(X_train_subs)
        y_train_sub = np.concatenate(y_train_subs)
        X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)
        small_dataset = X_train_sub, y_train_sub, X_test, y_test

        gl_results.append_values(*get_average_uncertainty_gaussian_logits(small_dataset, model_function, epochs),
                                 dataset_size)

        it_results.append_values(*get_average_uncertainty_it(small_dataset, model_function, epochs), dataset_size)

    return gl_results, it_results


def plot_ale_epi_acc_on_axes(ax, results: UncertaintyResults, accuracy_y_ax_to_share=None, is_final_column=False):
    ax.plot(results.changed_parameter_values, normalise(results.epistemic_uncertainties), label="Epistemic")
    ax.plot(results.changed_parameter_values, normalise(results.aleatoric_uncertainties), label="Aleatoric")

    accuracy_axes = ax.twinx()
    accuracy_axes.plot(results.changed_parameter_values, results.accuracies,
                       label="Accuracy", color='green')

    if is_final_column:
        accuracy_axes.set_ylabel("Accuracy", color='green')
    else:
        plt.setp(accuracy_axes.get_yticklabels(), visible=False)

    if accuracy_y_ax_to_share:
        accuracy_axes.sharey(accuracy_y_ax_to_share)
    return accuracy_axes


def plot_decreasing_dataset(dataset_name, config):
    dataset, architectures, epochs = config

    fig, axes = plt.subplots(2, len(architectures), figsize=(10, 6), sharey=True, sharex=True)
    accuracy_y_ax_to_share = None
    for arch_idx, architecture in enumerate(architectures):
        gaussian_logits_results, it_results = run_decreasing_dataset(
            dataset, architecture.model_function, epochs)

        if not os.path.exists(f"{FIGURE_FOLDER}/decreasing_dataset/"):
            os.mkdir(f"{FIGURE_FOLDER}/decreasing_dataset/")

        is_first_column = arch_idx == 0
        is_final_column = arch_idx == len(architectures) - 1

        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[0][arch_idx], gaussian_logits_results,
                                                          accuracy_y_ax_to_share, is_final_column)
        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[1][arch_idx], it_results,
                                                          accuracy_y_ax_to_share, is_final_column)

        axes[0][arch_idx].set_title(architecture.uq_name)
        axes[1][arch_idx].set_xlabel("Dataset size")

        if is_first_column:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty (normalised)")
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty (normalised)")

        if is_final_column:
            axes[0][arch_idx].legend()

    fig.suptitle(f"Disentangled uncertainty over decreasing dataset sizes for {dataset_name}", fontsize=20)
    fig.tight_layout()

    if TEST_MODE:
        fig.savefig(f"{FIGURE_FOLDER}/decreasing_dataset/disentangled_uncertainties_{dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"{FIGURE_FOLDER}/decreasing_dataset/disentangled_uncertainties_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()

    for name, conf in get_datasets().items():
        if name == "blobs":
            plot_decreasing_dataset(name, conf)

    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
