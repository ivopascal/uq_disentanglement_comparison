import gc
import os.path
import warnings
from datetime import datetime

import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

import keras.backend as K

from disentanglement.benchmarks.decreasing_dataset import plot_decreasing_dataset
from disentanglement.benchmarks.label_noise import label_noise
from disentanglement.benchmarks.ood_class_detection import plot_ood_class_detection
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.settings import TEST_MODE, GPU_INDEX
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    physical_devices = tf.config.list_physical_devices('GPU')
    plt.rcParams['axes.grid'] = False
    sns.reset_orig()

    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.get_logger().setLevel('ERROR')

    start_time = datetime.now()
    experiment_configs = get_experiment_configs()
    from_folder = True

    if TEST_MODE:
        from_folder = False

    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    for experiment_config in experiment_configs:
        has_dataset = experiment_config.dataset is None

        if "decreasing_dataset" in experiment_config.meta_experiments:
            plot_decreasing_dataset(experiment_config, from_folder=(has_dataset or from_folder))
        if "label_noise" in experiment_config.meta_experiments:
            label_noise(experiment_config, from_folder=(has_dataset or from_folder))
        if "ood_class" in experiment_config.meta_experiments:
            plot_ood_class_detection(experiment_config, from_folder=(has_dataset or from_folder))
        experiment_config.dataset = None
        K.clear_session()
        gc.collect()

    end_time = datetime.now()

    print(f"Ran all experiments in {end_time - start_time}")


if __name__ == "__main__":
    main()
