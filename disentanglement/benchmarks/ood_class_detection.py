import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from disentanglement.data.datasets import get_datasets
from disentanglement.models.entropy_models import train_entropy_model, expected_entropy, mutual_information
from disentanglement.models.multi_head_models import train_disentangle_model, uncertainty
from disentanglement.settings import BATCH_SIZE, NUM_SAMPLES


def ood_class_detection(dataset_name, config):
    dataset, architecture_func, epochs = config
    X_train, y_train, X_test, y_test = dataset
    ood_class = y_train.max()
    X_train_id = X_train[y_train[:, 0] != ood_class]  # Effectively removes the last class
    y_train_id = y_train[y_train != ood_class]
    n_classes = len(np.unique(y_train_id))

    disentangle_model = train_disentangle_model(architecture_func, X_train_id, y_train_id, n_classes, epochs=epochs)
    entropy_model = train_entropy_model(architecture_func, X_train_id, y_train_id, n_classes, epochs=epochs)

    pred_mean, pred_ale_std, pred_epi_std = disentangle_model.predict(X_test, batch_size=BATCH_SIZE)
    entropy_preds = entropy_model.predict_samples(X_test, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)

    ale_disentangle = uncertainty(pred_ale_std)
    epi_disentangle = uncertainty(pred_epi_std)

    y_test = y_test.reshape(-1)
    axes[0][0].set_ylabel("Multi-head disentangle")
    axes[0][0].hist(ale_disentangle[y_test != ood_class], label="ID")
    axes[0][0].hist(ale_disentangle[y_test == ood_class], label="OOD")
    axes[0][0].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, ale_disentangle):.3f}")
    axes[0][0].set_xlabel("Aleatoric uncertainty")

    axes[0][1].hist(epi_disentangle[y_test != ood_class], label="ID")
    axes[0][1].hist(epi_disentangle[y_test == ood_class], label="OOD")
    axes[0][1].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, epi_disentangle):.3f}")
    axes[0][1].set_xlabel("Epistemic uncertainty")

    axes[0][1].legend()

    ale_entropy = expected_entropy(entropy_preds)
    epi_entropy = mutual_information(entropy_preds)

    axes[1][0].set_ylabel("Entropy disentangle")
    axes[1][0].hist(ale_entropy[y_test != ood_class], label="ID")
    axes[1][0].hist(ale_entropy[y_test == ood_class], label="OOD")
    axes[1][0].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, ale_entropy):.3f}")
    axes[1][0].set_xlabel("Aleatoric uncertainty")

    axes[1][1].hist(epi_entropy[y_test != ood_class], label="ID")
    axes[1][1].hist(epi_entropy[y_test == ood_class], label="OOD")
    axes[1][1].title.set_text(f"AUROC = {roc_auc_score(y_test == ood_class, epi_entropy):.3f}")
    axes[1][1].set_xlabel("Epistemic uncertainty")

    fig.suptitle(f"Disentangled uncertainty for OOD class detection with {dataset_name}", fontsize=20)
    fig.tight_layout()

    if not os.path.exists("../../figures/ood_class/"):
        os.mkdir("../../figures/ood_class/")

    plt.savefig(f"../../figures/ood_class/histograms_{dataset_name}.pdf")


if __name__ == "__main__":
    ood_class_detection("CIFAR10", get_datasets()["CIFAR10"])
