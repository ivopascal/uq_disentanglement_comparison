import gc
import os.path
from datetime import datetime
from typing import Dict, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils import shuffle

from disentanglement.benchmarks.plotting import plot_ale_epi_acc_on_axes, plot_results_on_idx
from disentanglement.datatypes import UncertaintyResults, Dataset
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.logging import TQDM
from disentanglement.models.disentanglement import DISENTANGLEMENT_FUNCS
from disentanglement.settings import TEST_MODE, FIGURE_FOLDER
from disentanglement.util import print_correlations
from disentanglement.results_storing import save_results_to_file, load_results_from_file

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


def run_decreasing_dataset(dataset: Dataset, model_function, epochs) -> Dict[str, UncertaintyResults]:
    results = {disentanglement_name: UncertaintyResults() for disentanglement_name, func in
               DISENTANGLEMENT_FUNCS.items()}

    dataset_sizes = DATASET_SIZES
    if TEST_MODE:
        epochs = 2
        dataset_sizes = [0.1, 0.5]

    X_train, y_train = shuffle(dataset.X_train, dataset.y_train)
    y_train = y_train.reshape(-1)

    for dataset_size in dataset_sizes:
        small_dataset = create_subsampled_dataset(X_train, y_train, dataset, dataset_size)
        adjusted_epochs = int(epochs / dataset_size)

        for disentanglement_name, disentanglement_func in DISENTANGLEMENT_FUNCS.items():
            results[disentanglement_name].append_values(*disentanglement_func(small_dataset, model_function,
                                                                              adjusted_epochs), dataset_size)
            gc.collect()

    return results


def request_results_or_run(experiment_config, architecture, from_folder,
                           run_function: Callable, meta_experiment_name: str):
    if from_folder:
        try:
            results, results_std = load_results_from_file(
                experiment_config, architecture,
                meta_experiment_name=meta_experiment_name)
            print(f"Found results for {meta_experiment_name}, "
                  f"on {experiment_config.dataset_name}, "
                  f"with {architecture.uq_name}")
            print(f"Correlation on {meta_experiment_name} - {architecture.uq_name}")
            print_correlations(results)

            return results, results_std

        except FileNotFoundError:
            print(
                f"Failed to find results for {meta_experiment_name}, "
                f"on {experiment_config.dataset_name}, "
                f"with {architecture.uq_name}")

    results = run_function(experiment_config.dataset, architecture.model_function, architecture.epochs)
    save_results_to_file(experiment_config, architecture, results, meta_experiment_name=meta_experiment_name)

    return results, None


def plot_decreasing_dataset(experiment_config, from_folder=False):
    fig, axes = plt.subplots(len(DISENTANGLEMENT_FUNCS), len(experiment_config.models), figsize=(10, 6), sharey=True,
                             sharex=True)
    fontsize = 14
    plt.rcParams['font.size'] = fontsize

    for arch_idx, architecture in enumerate(experiment_config.models):
        TQDM.set_description(
            f"Running experiment  {META_EXPERIMENT_NAME} on {experiment_config.dataset_name} with {architecture.uq_name}")

        results, results_std = request_results_or_run(
            experiment_config, architecture, from_folder, run_decreasing_dataset, "decreasing_dataset")

        plot_results_on_idx(results, results_std, arch_idx, axes, experiment_config, architecture, META_EXPERIMENT_NAME)

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
