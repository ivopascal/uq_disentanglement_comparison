import os

import numpy as np
import pandas as pd

from disentanglement.data.eeg import N_EEG_SUBJECTS
from disentanglement.datatypes import UncertaintyResults
from disentanglement.settings import DATA_FOLDER, TEST_MODE, N_CIFAR_REPETITIONS


def normalise(x):
    x = np.array(x)
    return (x - min(x)) / max(x - min(x))


def get_test_append():
    if TEST_MODE:
        test_append = "_test"
    else:
        test_append = ""

    return test_append


def load_and_combine_multiple_logs(experiment_config, architecture, meta_experiment_name, n_logs):
    gaussian_logit_dfs = []
    it_dfs = []
    for log_id in range(n_logs):
        try:
            gaussian_logit_dfs.append(pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                                                  f"{experiment_config.dataset_name} {log_id}_{architecture.uq_name}_"
                                                  f"gaussian_logits_results{get_test_append()}.csv"))
            it_dfs.append(pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                                      f"{experiment_config.dataset_name} {log_id}_{architecture.uq_name}_"
                                      f"it_results{get_test_append()}.csv"))
        except FileNotFoundError:
            print(f"Failed to find data for log {log_id}. Skipping...")

    gaussian_logit_df = pd.concat(gaussian_logit_dfs)
    gaussian_logit_df_std = gaussian_logit_df.groupby(gaussian_logit_df.index).sem()
    gaussian_logit_df = gaussian_logit_df.groupby(gaussian_logit_df.index).mean()

    it_df = pd.concat(it_dfs)
    it_df_std = it_df.groupby(it_df.index).sem()
    it_df = it_df.groupby(it_df.index).mean()

    return (UncertaintyResults(**gaussian_logit_df.to_dict(orient='list')),
            UncertaintyResults(**it_df.to_dict(orient='list')), UncertaintyResults(**gaussian_logit_df_std.to_dict(orient='list')), UncertaintyResults(**it_df_std.to_dict(orient='list')))


def save_results_to_file(experiment_config, architecture, gaussian_logits_results, it_results, meta_experiment_name):
    if not os.path.exists(f"{DATA_FOLDER}/"):
        os.mkdir(f"{DATA_FOLDER}/")
    if not os.path.exists(f"{DATA_FOLDER}/{meta_experiment_name}/"):
        os.mkdir(f"{DATA_FOLDER}/{meta_experiment_name}/")

    df_gaussian_logits = pd.DataFrame(gaussian_logits_results.__dict__)
    df_gaussian_logits.to_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                              f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                              f"gaussian_logits_results{get_test_append()}.csv", index=False)

    df_it_results = pd.DataFrame(it_results.__dict__)
    df_it_results.to_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                         f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                         f"it_results{get_test_append()}.csv", index=False)


def load_results_from_file(experiment_config, architecture, meta_experiment_name):
    if experiment_config.dataset_name == "Motor Imagery BCI":
        return load_and_combine_multiple_logs(experiment_config, architecture, meta_experiment_name, n_logs=N_EEG_SUBJECTS)

    if experiment_config.dataset_name == "CIFAR10" or experiment_config.dataset_name == "Fashion MNIST":
        return load_and_combine_multiple_logs(experiment_config, architecture, meta_experiment_name, n_logs=N_CIFAR_REPETITIONS)

    df_gaussian_logits = pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                                     f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                                     f"gaussian_logits_results{get_test_append()}.csv")
    gaussian_logits_results = UncertaintyResults(**df_gaussian_logits.to_dict(orient='list'))

    df_it = pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                        f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                        f"it_results{get_test_append()}.csv")
    it_results = UncertaintyResults(**df_it.to_dict(orient='list'))
    return gaussian_logits_results, it_results, None, None
