import os

import numpy as np
import pandas as pd

from disentanglement.datatypes import UncertaintyResults
from disentanglement.settings import DATA_FOLDER, TEST_MODE


def normalise(x):
    x = np.array(x)
    return (x - min(x)) / max(x - min(x))


def get_test_append():
    if TEST_MODE:
        test_append = "_test"
    else:
        test_append = ""

    return test_append


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
    df_gaussian_logits = pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                                     f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                                     f"gaussian_logits_results{get_test_append()}.csv")
    gaussian_logits_results = UncertaintyResults(**df_gaussian_logits.to_dict(orient='list'))

    df_it = pd.read_csv(f"{DATA_FOLDER}/{meta_experiment_name}/{meta_experiment_name}_"
                        f"{experiment_config.dataset_name}_{architecture.uq_name}_"
                        f"it_results{get_test_append()}.csv")
    it_results = UncertaintyResults(**df_it.to_dict(orient='list'))
    return gaussian_logits_results, it_results
