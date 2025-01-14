import gc
import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils import shuffle

from disentanglement.benchmarks.plotting import plot_ale_epi_acc_on_axes
from disentanglement.datatypes import UncertaintyResults, Dataset
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.logging import TQDM
from disentanglement.models.gaussian_logits_models import get_average_uncertainty_gaussian_logits
from disentanglement.models.information_theoretic_models import get_average_uncertainty_it
from disentanglement.models.logit_variance import get_average_uncertainty_logit_variance
from disentanglement.settings import TEST_MODE, FIGURE_FOLDER
from disentanglement.util import load_results_from_file, save_results_to_file, print_correlations

META_EXPERIMENT_NAME = "decreasing_dataset"
DATASET_SIZES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def create_subsampled_dataset(X_train, y_train, dataset, dataset_size) -> Dataset:
    X_train_subs = []
    y_train_subs = []

    for y_value in np.unique(y_train):
        n_samples_per_class = int(np.sum((y_train == y_value)) * dataset_size)
        if n_samples_per_class == 0:
            n_samples_per_class = 1
        X_train_subs.append(X_train[y_train == y_value][:n_samples_per_class])
        y_train_subs.append(y_train[y_train == y_value][:n_samples_per_class])

    X_train_sub = np.concatenate(X_train_subs)
    y_train_sub = np.concatenate(y_train_subs)
    X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)
    small_dataset = Dataset(X_train_sub, y_train_sub, dataset.X_test, dataset.y_test,
                            is_regression=dataset.is_regression)

    return small_dataset


def run_decreasing_dataset(dataset: Dataset, model_function, epochs):
    gl_results = UncertaintyResults()
    it_results = UncertaintyResults()

    dataset_sizes = DATASET_SIZES
    if TEST_MODE:
        epochs = 2
        dataset_sizes = [0.1, 0.5]

    X_train, y_train = shuffle(dataset.X_train, dataset.y_train)
    y_train = y_train.reshape(-1)

    for dataset_size in dataset_sizes:
        small_dataset = create_subsampled_dataset(X_train, y_train, dataset, dataset_size)
        adjusted_epochs = int(epochs / dataset_size)

        print(get_average_uncertainty_logit_variance(small_dataset, model_function, adjusted_epochs))

        gl_results.append_values(
            *get_average_uncertainty_gaussian_logits(small_dataset, model_function, adjusted_epochs),
            dataset_size)

        it_results.append_values(*get_average_uncertainty_it(small_dataset, model_function, adjusted_epochs),
                                 dataset_size)
        gc.collect()

    return gl_results, it_results


def request_results(experiment_config, architecture, from_folder):
    if from_folder:
        try:
            gaussian_logits_results, it_results, gaussian_logits_results_std, it_results_std = load_results_from_file(
                experiment_config, architecture,
                meta_experiment_name=META_EXPERIMENT_NAME)
            print(f"Found results for {META_EXPERIMENT_NAME}, "
                  f"on {experiment_config.dataset_name}, "
                  f"with {architecture.uq_name}")
            print(f"Correlation on changing dataset size - {architecture.uq_name}")
            print_correlations(gaussian_logits_results, it_results)

            return gaussian_logits_results, it_results, gaussian_logits_results_std, it_results_std

        except FileNotFoundError:
            print(
                f"Failed to find results for {META_EXPERIMENT_NAME}, on {experiment_config.dataset_name}, with {architecture.uq_name}")

    gaussian_logits_results, it_results = run_decreasing_dataset(
        experiment_config.dataset, architecture.model_function, architecture.epochs)
    save_results_to_file(experiment_config, architecture, gaussian_logits_results, it_results,
                         meta_experiment_name=META_EXPERIMENT_NAME)

    return gaussian_logits_results, it_results, None, None


def plot_decreasing_dataset(experiment_config, from_folder=False):
    fig, axes = plt.subplots(2, len(experiment_config.models), figsize=(10, 6), sharey=True, sharex=True)
    accuracy_y_ax_to_share = None
    font_size = 14
    plt.rcParams['font.size'] = font_size

    for arch_idx, architecture in enumerate(experiment_config.models):
        TQDM.set_description(
            f"Running experiment  {META_EXPERIMENT_NAME} on {experiment_config.dataset_name} with {architecture.uq_name}")

        gaussian_logits_results, it_results, gaussian_logits_results_std, it_results_std = request_results(
            experiment_config, architecture, from_folder)

        ## PLOTTING
        is_first_column = arch_idx == 0
        is_final_column = arch_idx == len(experiment_config.models) - 1

        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[0][arch_idx], gaussian_logits_results,
                                                          accuracy_y_ax_to_share, is_final_column,
                                                          std=gaussian_logits_results_std)
        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[1][arch_idx], it_results,
                                                          accuracy_y_ax_to_share, is_final_column,
                                                          std=it_results_std)

        axes[0][arch_idx].set_title(architecture.uq_name, fontsize=font_size)
        axes[1][arch_idx].set_xlabel("Dataset size", fontsize=font_size)


        ## ADD HEADERS & LEGEND TO FIRST COLUMN
        if is_first_column:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty", fontsize=font_size)
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty", fontsize=font_size)

            handles, labels = axes[0][arch_idx].get_legend_handles_labels()

            labels.append("Acc")
            line = Line2D([0], [0], label='Acc', color='green')
            handles.append(line)

            axes[0][arch_idx].legend(handles=handles, labels=labels, loc='upper left', fontsize=10)

    fig.tight_layout()
    if not os.path.exists(f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/"):
        os.mkdir(f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/")

    if TEST_MODE:
        fig.savefig(
            f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/{META_EXPERIMENT_NAME}_{experiment_config.dataset_name}_TEST.pdf")
    else:
        fig.savefig(
            f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/{META_EXPERIMENT_NAME}_{experiment_config.dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()
    experiment_configs = get_experiment_configs()
    for experiment_conf in experiment_configs:
        if experiment_conf.dataset_name == "Motor Imagery BCI":
            plot_decreasing_dataset(experiment_conf, from_folder=False)

    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
