import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from disentanglement.data.datasets import get_datasets
from disentanglement.models.information_theoretic_models import train_it_model, expected_entropy, mutual_information
from disentanglement.models.gaussian_logits_models import train_gaussian_logits_model, uncertainty
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES, TEST_MODE, FIGURE_FOLDER


def ood_class_detection(dataset_name, config):
    dataset, architectures, epochs = config

    if TEST_MODE:
        epochs = 1

    X_train, y_train, X_test, y_test = dataset
    ood_class = y_train.max()
    X_train_id = X_train[y_train[:, 0] != ood_class]  # Effectively removes the last class
    y_train_id = y_train[y_train != ood_class]
    n_classes = len(np.unique(y_train_id))

    for architecture in architectures:
        uq_name = architecture.uq_name
        gaussian_logits_model = train_gaussian_logits_model(architecture.model_function, X_train_id, y_train_id, n_classes,
                                                        epochs=epochs)
        it_model = train_it_model(architecture.model_function, X_train_id, y_train_id, n_classes, epochs=epochs)

        pred_mean, pred_ale_std, pred_epi_std = gaussian_logits_model.predict(X_test, batch_size=BATCH_SIZE)
        it_preds = it_model.predict_samples(X_test, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)

        ale_gaussian_logits = uncertainty(pred_ale_std)
        epi_gaussian_logits = uncertainty(pred_epi_std)

        y_test = y_test.reshape(-1)
        axes[0][0].set_ylabel("Gaussian Logits")
        axes[0][0].hist(ale_gaussian_logits[y_test != ood_class], label="ID")
        axes[0][0].hist(ale_gaussian_logits[y_test == ood_class], label="OOD")
        axes[0][0].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, ale_gaussian_logits):.3f}")
        axes[0][0].set_xlabel("Aleatoric uncertainty")

        axes[0][1].hist(epi_gaussian_logits[y_test != ood_class], label="ID")
        axes[0][1].hist(epi_gaussian_logits[y_test == ood_class], label="OOD")
        axes[0][1].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, epi_gaussian_logits):.3f}")
        axes[0][1].set_xlabel("Epistemic uncertainty")

        axes[0][1].legend()

        ale_it = expected_entropy(it_preds)
        epi_it = mutual_information(it_preds)

        axes[1][0].set_ylabel("Information Theoretic")
        axes[1][0].hist(ale_it[y_test != ood_class], label="ID")
        axes[1][0].hist(ale_it[y_test == ood_class], label="OOD")
        axes[1][0].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, ale_it):.3f}")
        axes[1][0].set_xlabel("Aleatoric uncertainty")

        axes[1][1].hist(epi_it[y_test != ood_class], label="ID")
        axes[1][1].hist(epi_it[y_test == ood_class], label="OOD")
        axes[1][1].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, epi_it):.3f}")
        axes[1][1].set_xlabel("Epistemic uncertainty")

        fig.suptitle(f"Disentangled uncertainty for OOD class detection with {uq_name} on {dataset_name}", fontsize=20)
        fig.tight_layout()

        if not os.path.exists(f"{FIGURE_FOLDER}/ood_class/"):
            os.mkdir(f"{FIGURE_FOLDER}/ood_class/")

        if TEST_MODE:
            plt.savefig(f"{FIGURE_FOLDER}/ood_class/histograms_{uq_name}_{dataset_name}_TEST.pdf")
        else:
            plt.savefig(f"{FIGURE_FOLDER}/ood_class/histograms_{uq_name}_{dataset_name}.pdf")


if __name__ == "__main__":
    start_time = datetime.now()
    ood_class_detection("CIFAR10", get_datasets()["CIFAR10"])
    print(f"Running Decreasing Dataset experiments took: {datetime.now() - start_time}")
