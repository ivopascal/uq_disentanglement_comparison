import os
from datetime import datetime
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score

from disentanglement.datatypes import UncertaintyResults
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.models.information_theoretic_models import train_it_model, expected_entropy, mutual_information
from disentanglement.models.gaussian_logits_models import train_gaussian_logits_model, uncertainty
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE, FIGURE_FOLDER
from disentanglement.logging import TQDM
from disentanglement.util import load_results_from_file, save_results_to_file

META_EXPERIMENT_NAME = "ood_class"


def determine_tprs_for_roc(base_fpr, y_ood_true, y_ood_score):
    fpr, tpr, _ = roc_curve(y_ood_true, y_ood_score)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return tpr


def run_ood_class_detection(dataset, architecture_func, epochs) -> Tuple[UncertaintyResults, UncertaintyResults]:
    ood_classes = np.unique(dataset.y_train)
    n_classes = len(np.unique(dataset.y_train))
    y_test = dataset.y_test.reshape(-1)
    y_train = dataset.y_train.reshape(-1)

    base_fpr = np.linspace(0, 1, 101)

    if TEST_MODE:
        epochs = 1
        ood_classes = ood_classes[:2]

    ale_gaussian_logit_tprs = []
    epi_gaussian_logit_tprs = []
    ale_it_tprs = []
    epi_it_tprs = []
    for ood_class in ood_classes:
        X_train_id = dataset.X_train[y_train != ood_class]
        y_train_id = dataset.y_train[y_train != ood_class]
        y_test_ood = y_test == ood_class

        it_model = train_it_model(architecture_func, X_train_id, y_train_id, n_classes, epochs=epochs)
        it_preds = it_model.predict_samples(dataset.X_test, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        ale_it_tprs.append(determine_tprs_for_roc(base_fpr, y_test_ood, expected_entropy(it_preds)))
        epi_it_tprs.append(determine_tprs_for_roc(base_fpr, y_test_ood, mutual_information(it_preds)))

        accuracy = accuracy_score(y_test, it_preds.mean(axis=0).argmax(axis=1))
        if accuracy < 0.4:
            print(f"Warning, low accuracy: {accuracy} with Information Theoretic model")

        gaussian_logits_model = train_gaussian_logits_model(architecture_func, X_train_id, y_train_id,
                                                            n_classes, epochs=epochs)
        pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(dataset.X_test, batch_size=BATCH_SIZE)

        accuracy = accuracy_score(y_test, it_preds.mean(axis=0).argmax(axis=1))
        if accuracy < 0.4:
            print(f"Warning, low accuracy: {accuracy} with Information Theoretic model")

        ale_gaussian_logits = uncertainty(pred_ale_std)
        epi_gaussian_logits = uncertainty(pred_epi_std)
        ale_gaussian_logit_tprs.append(determine_tprs_for_roc(base_fpr, y_test_ood, ale_gaussian_logits))
        epi_gaussian_logit_tprs.append(determine_tprs_for_roc(base_fpr, y_test_ood, epi_gaussian_logits))

    gl_results = UncertaintyResults(accuracies=np.empty_like(base_fpr),
                                    aleatoric_uncertainties=np.array(ale_gaussian_logit_tprs).mean(axis=0),
                                    epistemic_uncertainties=np.array(epi_gaussian_logit_tprs).mean(axis=0),
                                    changed_parameter_values=base_fpr
                                    )

    it_results = UncertaintyResults(accuracies=np.empty_like(base_fpr),
                                    aleatoric_uncertainties=np.array(ale_it_tprs).mean(axis=0),
                                    epistemic_uncertainties=np.array(epi_it_tprs).mean(axis=0),
                                    changed_parameter_values=base_fpr
                                    )

    return gl_results, it_results


def plot_roc_on_ax(ax, aleatoric_tpr, epistemic_tpr, base_fpr):
    ax.plot(base_fpr, aleatoric_tpr, label="Aleatoric")
    ax.plot(base_fpr, epistemic_tpr, label="Epistemic")
    ax.plot(base_fpr, base_fpr, color='black', linestyle='dashed')


def plot_ood_class_detection(experiment_config, from_folder=None):
    if not os.path.exists(f"{FIGURE_FOLDER}/ood_class/"):
        os.mkdir(f"{FIGURE_FOLDER}/ood_class/")

    fig, axes = plt.subplots(2, len(experiment_config.models), figsize=(10, 6), sharey=True, sharex=True)

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
            gaussian_logits_results, it_results = run_ood_class_detection(experiment_config.dataset,
                                                                          architecture.model_function,
                                                                          architecture.epochs)
            save_results_to_file(experiment_config, architecture, gaussian_logits_results, it_results,
                                 meta_experiment_name=META_EXPERIMENT_NAME)

        plot_roc_on_ax(axes[0][arch_idx], gaussian_logits_results.aleatoric_uncertainties,
                       gaussian_logits_results.epistemic_uncertainties,
                       gaussian_logits_results.changed_parameter_values)
        plot_roc_on_ax(axes[1][arch_idx], it_results.aleatoric_uncertainties, it_results.epistemic_uncertainties,
                       it_results.changed_parameter_values)

        axes[0][arch_idx].set_title(architecture.uq_name)
        axes[1][arch_idx].set_xlabel("False Positive Rate")

        if arch_idx == 0:
            axes[0][arch_idx].set_ylabel("Gaussian Logits\nTrue Positive Rate")
            axes[1][arch_idx].set_ylabel("Information Theoretic\nTrue Positive Rate")

        if arch_idx == len(experiment_config.models) - 1:
            axes[0][arch_idx].legend(loc="lower right")

    fig.suptitle(f"ROC curves for OOD detection for {experiment_config.dataset_name}", fontsize=20)
    fig.tight_layout()

    if TEST_MODE:
        fig.savefig(f"{FIGURE_FOLDER}/ood_class/ood_roc_curve_{experiment_config.dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"{FIGURE_FOLDER}/ood_class/ood_roc_curve_{experiment_config.dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()

    experiment_configs = get_experiment_configs()
    for experiment_conf in experiment_configs:
        if experiment_conf.dataset_name == "CIFAR10":
            plot_ood_class_detection(experiment_conf)
    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
