import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from disentanglement.benchmarks.decreasing_dataset import plot_ale_epi_acc_on_axes
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.datatypes import UncertaintyResults, ExperimentConfig, Dataset
from disentanglement.models.gaussian_logits_models import get_average_uncertainty_gaussian_logits
from disentanglement.models.information_theoretic_models import get_average_uncertainty_it
from disentanglement.settings import TEST_MODE, FIGURE_FOLDER, NUM_DECREASING_DATASET_STEPS, NUM_LABEL_NOISE_STEPS
from disentanglement.logging import TQDM
from disentanglement.util import load_results_from_file, save_results_to_file

META_EXPERIMENT_NAME = 'label_noise'


def partial_shuffle_dataset(x, y, percentage):
    x_noisy, y_noisy = shuffle(x, y)
    np.random.shuffle(y_noisy[:int(len(y_noisy) * percentage)])
    x_noisy, y_noisy = shuffle(x_noisy, y_noisy)
    return x_noisy, y_noisy


def run_label_noise(dataset: Dataset, architecture_func, epochs):
    gl_results = UncertaintyResults()
    it_results = UncertaintyResults()

    noises = np.linspace(0, 1.0, NUM_LABEL_NOISE_STEPS)

    if TEST_MODE:
        epochs = 2

    for noise in noises:
        X_train_noisy, y_train_noisy = partial_shuffle_dataset(dataset.X_train, dataset.y_train, percentage=noise)
        X_test_noisy, y_test_noisy = partial_shuffle_dataset(dataset.X_test, dataset.y_test, percentage=noise)
        noisy_dataset = Dataset(X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy)
        gl_results.append_values(*get_average_uncertainty_gaussian_logits(noisy_dataset, architecture_func, epochs),
                                 noise)

        it_results.append_values(*get_average_uncertainty_it(noisy_dataset, architecture_func, epochs), noise)

    return gl_results, it_results


def label_noise(experiment_config: ExperimentConfig, from_folder=False):
    fig, axes = plt.subplots(2, len(experiment_config.models), figsize=(10, 6), sharey=True, sharex=True)
    accuracy_y_ax_to_share = None
    for arch_idx, architecture in enumerate(experiment_config.models):
        TQDM.set_description(f"Running experiment {META_EXPERIMENT_NAME} on {experiment_config.dataset_name} with {architecture.uq_name}")

        gaussian_logits_results, it_results = None, None
        if from_folder:
            try:
                gaussian_logits_results, it_results = load_results_from_file(experiment_config, architecture,
                                                                             meta_experiment_name=META_EXPERIMENT_NAME)
                print(f"Found results for {META_EXPERIMENT_NAME}, on {experiment_config.dataset_name}, with {architecture.uq_name}")

            except FileNotFoundError:
                print(f"failed to find results for {META_EXPERIMENT_NAME}, on {experiment_config.dataset_name}, with {architecture.uq_name}")
        if not gaussian_logits_results or not it_results:
            gaussian_logits_results, it_results = run_label_noise(experiment_config.dataset,
                                                                  architecture.model_function, architecture.epochs)
            save_results_to_file(experiment_config, architecture, gaussian_logits_results, it_results,
                                 meta_experiment_name=META_EXPERIMENT_NAME)
        if not os.path.exists(f"{FIGURE_FOLDER}/noise_dataset/"):
            os.mkdir(f"{FIGURE_FOLDER}/noise_dataset")

        is_first_column = arch_idx == 0
        is_final_column = arch_idx == len(experiment_config.models) - 1

        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[0][arch_idx], gaussian_logits_results,
                                                          accuracy_y_ax_to_share, is_final_column,
                                                          normalise_uncertainties=False)
        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[1][arch_idx], it_results,
                                                          accuracy_y_ax_to_share, is_final_column,
                                                          normalise_uncertainties=False)

        axes[0][arch_idx].set_title(architecture.uq_name)
        axes[1][arch_idx].set_xlabel("Labels shuffled")
        if is_first_column:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty")
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty")

        if is_final_column:
            axes[0][arch_idx].legend()

    fig.suptitle(f"Disentangled uncertainty over shuffled labels for {experiment_config.dataset_name}", fontsize=20)
    fig.tight_layout()

    if TEST_MODE:
        fig.savefig(
            f"{FIGURE_FOLDER}/noise_dataset/disentangled_uncertainties_{experiment_config.dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"{FIGURE_FOLDER}/noise_dataset/disentangled_uncertainties_{experiment_config.dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()
    experiment_configs = get_experiment_configs()
    for experiment_conf in experiment_configs:
        if experiment_conf.dataset_name == "CIFAR10":
            label_noise(experiment_conf)
    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
