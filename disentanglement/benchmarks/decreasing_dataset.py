import gc
import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils import shuffle

from disentanglement.datatypes import UncertaintyResults, Dataset
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.logging import TQDM
from disentanglement.models.gaussian_logits_models import get_average_uncertainty_gaussian_logits
from disentanglement.models.information_theoretic_models import get_average_uncertainty_it
from disentanglement.settings import TEST_MODE, FIGURE_FOLDER, NUM_DECREASING_DATASET_STEPS
from disentanglement.util import normalise, load_results_from_file, save_results_to_file

META_EXPERIMENT_NAME = "decreasing_dataset"


def run_decreasing_dataset(dataset: Dataset, model_function, epochs):
    gl_results = UncertaintyResults()
    it_results = UncertaintyResults()

    dataset_sizes = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    if TEST_MODE:
        epochs = 2
        dataset_sizes = [0.1, 0.5]

    dataset_sizes = dataset_sizes[::-1]

    X_train, y_train = shuffle(dataset.X_train, dataset.y_train)
    y_train = y_train.reshape(-1)
    for dataset_size in dataset_sizes:
        X_train_subs = []
        y_train_subs = []

        adjusted_epochs = int(epochs / dataset_size)

        for y_value in np.unique(y_train):
            n_samples_per_class = int(np.sum((y_train == y_value)) * dataset_size)
            if n_samples_per_class == 0:
                n_samples_per_class = 1
            X_train_subs.append(X_train[y_train == y_value][:n_samples_per_class])
            y_train_subs.append(y_train[y_train == y_value][:n_samples_per_class])

        X_train_sub = np.concatenate(X_train_subs)
        y_train_sub = np.concatenate(y_train_subs)
        X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)
        small_dataset = Dataset(X_train_sub, y_train_sub, dataset.X_test, dataset.y_test)

        gl_results.append_values(*get_average_uncertainty_gaussian_logits(small_dataset, model_function, adjusted_epochs),
                                 dataset_size)

        it_results.append_values(*get_average_uncertainty_it(small_dataset, model_function, adjusted_epochs), dataset_size)
        gc.collect()

    return gl_results, it_results


def plot_ale_epi_acc_on_axes(ax, results: UncertaintyResults, accuracy_y_ax_to_share=None, is_final_column=False, std=None,
                             normalise_uncertainties=False):
    # results.changed_parameter_values = results.changed_parameter_values[1:]
    # results.epistemic_uncertainties = results.epistemic_uncertainties[1:]
    # results.aleatoric_uncertainties = results.aleatoric_uncertainties[1:]
    # results.accuracies = results.accuracies[1:]


    if normalise_uncertainties:
        scatter_alpha = 0.15
    else:
        scatter_alpha = 1.0

    if normalise_uncertainties:
        ax.plot(results.changed_parameter_values, normalise(results.epistemic_uncertainties), label="Epistemic", alpha=scatter_alpha)
        ax.plot(results.changed_parameter_values, normalise(results.aleatoric_uncertainties), label="Aleatoric", alpha=scatter_alpha)

        z = np.polyfit(np.log(results.changed_parameter_values), normalise(results.epistemic_uncertainties), 1)
        ax.plot(results.changed_parameter_values, np.poly1d(z)(np.log(results.changed_parameter_values)))

        z = np.polyfit(np.log(results.changed_parameter_values), normalise(results.aleatoric_uncertainties), 1)
        ax.plot(results.changed_parameter_values, np.poly1d(z)(np.log(results.changed_parameter_values)))

    else:
        if std:
            ax.fill_between(results.changed_parameter_values, np.array(results.epistemic_uncertainties) - 1.96 * np.array(std.epistemic_uncertainties),
                            np.array(results.epistemic_uncertainties) + 1.96 * np.array(std.epistemic_uncertainties), alpha=0.3, label='_Epi')
            ax.fill_between(results.changed_parameter_values,
                            np.array(results.aleatoric_uncertainties) - 1.96 * np.array(std.aleatoric_uncertainties),
                            np.array(results.aleatoric_uncertainties) + 1.96 * np.array(std.aleatoric_uncertainties), alpha=0.3, label='_Ale')

        ax.plot(results.changed_parameter_values, results.epistemic_uncertainties, label="Epi")
        ax.plot(results.changed_parameter_values, results.aleatoric_uncertainties, label="Ale")




    # ax.set_xscale('log')

    if normalise_uncertainties:
        scatter_alpha = 0.15
    else:
        scatter_alpha = 1.0
    accuracy_axes = ax.twinx()
    accuracy_axes.plot(results.changed_parameter_values, results.accuracies,
                          label="Accuracy", color='green', alpha=scatter_alpha)

    if std:
        accuracy_axes.fill_between(results.changed_parameter_values,
                        np.array(results.accuracies) - 1.96 * np.array(std.accuracies),
                        np.array(results.accuracies) + 1.96 * np.array(std.accuracies), alpha=0.3, color='green')

    if normalise_uncertainties:
        z = np.polyfit(np.log(results.changed_parameter_values), results.accuracies, 1)
        accuracy_axes.plot(results.changed_parameter_values, np.poly1d(z)(np.log(results.changed_parameter_values)), color='green')

    if is_final_column:
        accuracy_axes.set_ylabel("Accuracy", color='green')
    else:
        plt.setp(accuracy_axes.get_yticklabels(), visible=False)

    if accuracy_y_ax_to_share:
        accuracy_axes.sharey(accuracy_y_ax_to_share)
    return accuracy_axes


def plot_decreasing_dataset(experiment_config, from_folder=False):
    fig, axes = plt.subplots(2, len(experiment_config.models), figsize=(10, 6), sharey=True, sharex=True)
    accuracy_y_ax_to_share = None
    font_size = 14
    plt.rcParams['font.size'] = font_size

    for arch_idx, architecture in enumerate(experiment_config.models):
        TQDM.set_description(
            f"Running experiment  {META_EXPERIMENT_NAME} on {experiment_config.dataset_name} with {architecture.uq_name}")
        gaussian_logits_results, it_results = None, None
        if from_folder:
            try:
                gaussian_logits_results, it_results, gaussian_logits_results_std, it_results_std = load_results_from_file(experiment_config, architecture,
                                                                             meta_experiment_name=META_EXPERIMENT_NAME)
                print(
                    f"Found results for {META_EXPERIMENT_NAME}, on {experiment_config.dataset_name}, with {architecture.uq_name}")
            except FileNotFoundError:
                print(
                    f"failed to find results for {META_EXPERIMENT_NAME}, on {experiment_config.dataset_name}, with {architecture.uq_name}")
        if not gaussian_logits_results or not it_results:
            gaussian_logits_results, it_results = run_decreasing_dataset(
                experiment_config.dataset, architecture.model_function, architecture.epochs)
            save_results_to_file(experiment_config, architecture, gaussian_logits_results, it_results,
                                 meta_experiment_name=META_EXPERIMENT_NAME)

        if TEST_MODE:  # Check if it's possible to load data from disk
            gaussian_logits_results, it_results = load_results_from_file(experiment_config, architecture,
                                                                         meta_experiment_name=META_EXPERIMENT_NAME)

        if not os.path.exists(f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/"):
            os.mkdir(f"{FIGURE_FOLDER}/{META_EXPERIMENT_NAME}/")

        is_first_column = arch_idx == 0
        is_final_column = arch_idx == len(experiment_config.models) - 1

        # entry_to_ignore = np.argmin(gaussian_logits_results.changed_parameter_values)
        # del gaussian_logits_results.accuracies[entry_to_ignore]
        # del gaussian_logits_results.epistemic_uncertainties[entry_to_ignore]
        # del gaussian_logits_results.aleatoric_uncertainties[entry_to_ignore]
        # del gaussian_logits_results.changed_parameter_values[entry_to_ignore]
        #
        # del it_results.accuracies[entry_to_ignore]
        # del it_results.epistemic_uncertainties[entry_to_ignore]
        # del it_results.aleatoric_uncertainties[entry_to_ignore]
        # del it_results.changed_parameter_values[entry_to_ignore]

        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[0][arch_idx], gaussian_logits_results,
                                                          accuracy_y_ax_to_share, is_final_column, std=gaussian_logits_results_std,
                                                          normalise_uncertainties=False)
        accuracy_y_ax_to_share = plot_ale_epi_acc_on_axes(axes[1][arch_idx], it_results,
                                                          accuracy_y_ax_to_share, is_final_column, std=it_results_std,
                                                          normalise_uncertainties=False)

        axes[0][arch_idx].set_title(architecture.uq_name, fontsize=font_size)
        axes[1][arch_idx].set_xlabel("Dataset size", fontsize=font_size)


        if is_first_column:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nUncertainty", fontsize=font_size)
            axes[1][arch_idx].set_ylabel("Information Theoretic\nUncertainty", fontsize=font_size)

            handles, labels = axes[0][arch_idx].get_legend_handles_labels()

            labels.append("Acc")
            line = Line2D([0], [0], label='Acc', color='green')
            handles.append(line)

            axes[0][arch_idx].legend(handles=handles, labels=labels, loc='upper left', fontsize=10)



    # fig.suptitle(f"Disentangled uncertainty over decreasing dataset sizes for {experiment_config.dataset_name}",
    #              fontsize=20)
    fig.tight_layout()

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
