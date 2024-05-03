import os.path
import warnings
from datetime import datetime

import tensorflow as tf

from disentanglement.benchmarks.decreasing_dataset import plot_decreasing_dataset
from disentanglement.benchmarks.label_noise import label_noise
from disentanglement.benchmarks.ood_class_detection import plot_ood_class_detection
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.settings import TEST_MODE

warnings.filterwarnings("ignore")


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
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
        if "decreasing_dataset" in experiment_config.meta_experiments:
            plot_decreasing_dataset(experiment_config, from_folder)
        if "label_noise" in experiment_config.meta_experiments:
            label_noise(experiment_config, from_folder)
        if "ood_class" in experiment_config.meta_experiments:
            plot_ood_class_detection(experiment_config, from_folder)

    end_time = datetime.now()

    print(f"Ran all experiments in {end_time - start_time}")


if __name__ == "__main__":
    main()
