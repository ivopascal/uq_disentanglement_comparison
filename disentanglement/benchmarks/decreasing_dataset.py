import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from disentanglement.data.datasets import get_datasets
from disentanglement.datatypes import UncertaintyResults
from disentanglement.models.information_theoretic_models import mutual_information, expected_entropy, train_it_model
from disentanglement.models.gaussian_logits_models import uncertainty, \
    train_gaussian_logits_model
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE, FIGURE_FOLDER
from disentanglement.util import normalise


def run_decreasing_dataset(dataset, model_function, epochs):
    gaussian_logits_results = UncertaintyResults()
    it_results = UncertaintyResults()

    X_train, y_train, X_test, y_test = dataset
    max_train_samples = y_train.shape[0]
    n_classes = len(np.unique(y_train))

    num_dataset_sizes = 20
    if TEST_MODE:
        num_dataset_sizes = 3
        epochs = 2

    dataset_sizes = np.logspace(start=1, stop=np.log2(max_train_samples), base=2, num=num_dataset_sizes)

    for dataset_size in tqdm(dataset_sizes):
        X_train_sub, y_train_sub = X_train[:int(dataset_size)], y_train[:int(dataset_size)]

        gaussian_logits_model = train_gaussian_logits_model(model_function, X_train_sub, y_train_sub, n_classes,
                                                            epochs=epochs)
        pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(X_test, batch_size=BATCH_SIZE)
        gaussian_logits_results.append_values(accuracy_score(y_test, pred_mean.argmax(axis=1)),
                                              uncertainty(pred_ale_std).mean(), uncertainty(pred_epi_std).mean(),
                                              dataset_size)

        it_model = train_it_model(model_function, X_train_sub, y_train_sub, n_classes, epochs=epochs)
        it_preds = it_model.predict_samples(X_test, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        it_results.append_values(accuracy_score(y_test, it_preds.mean(axis=0).argmax(axis=1)),
                                 expected_entropy(it_preds).mean(), mutual_information(it_preds).mean(), dataset_size)

    return gaussian_logits_results, it_results


def plot_decreasing_dataset(dataset_name, config):
    dataset, architectures, epochs = config

    fig, axes = plt.subplots(2, len(architectures), figsize=(10, 6), sharey=True, sharex=True)
    for arch_idx, architecture in enumerate(architectures):
        gaussian_logits_results, it_results = run_decreasing_dataset(
            dataset, architecture.model_function, epochs)
        dataset_sizes = gaussian_logits_results.changed_parameter_values
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(dataset_sizes, gaussian_logits_results.accuracies, label="Gaussian Logits")
        ax_acc.plot(dataset_sizes, it_results.accuracies, label="Information Theoretic")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Dataset size")
        ax_acc.legend()

        if not os.path.exists(f"{FIGURE_FOLDER}/decreasing_dataset/"):
            os.mkdir(f"{FIGURE_FOLDER}/decreasing_dataset/")

        if TEST_MODE:
            fig_acc.savefig(
                f"{FIGURE_FOLDER}/decreasing_dataset/accuracies_{architecture.uq_name}_{dataset_name}_TEST.pdf")
        else:
            fig_acc.savefig(f"{FIGURE_FOLDER}/decreasing_dataset/accuracies_{architecture.uq_name}_{dataset_name}.pdf")

        axes[0][arch_idx].plot(dataset_sizes, normalise(gaussian_logits_results.epistemic_uncertainties),
                               label="Epistemic")
        axes[0][arch_idx].plot(dataset_sizes, normalise(gaussian_logits_results.aleatoric_uncertainties),
                               label="Aleatoric")
        axes[0][arch_idx].set_title(architecture.uq_name)

        axes[1][arch_idx].plot(dataset_sizes,
                               normalise(it_results.epistemic_uncertainties),
                               label="Epistemic")
        axes[1][arch_idx].plot(dataset_sizes,
                               normalise(it_results.aleatoric_uncertainties),
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
        fig.savefig(f"{FIGURE_FOLDER}/decreasing_dataset/disentangled_uncertainties_{dataset_name}_TEST.pdf")
    else:
        fig.savefig(f"{FIGURE_FOLDER}/decreasing_dataset/disentangled_uncertainties_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()

    for name, conf in get_datasets().items():
        if name == "blobs":
            plot_decreasing_dataset(name, conf)

    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
