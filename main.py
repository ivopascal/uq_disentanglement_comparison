import os.path
from datetime import datetime

from tqdm import tqdm

from disentanglement.benchmarks.decreasing_dataset import plot_decreasing_dataset
from disentanglement.benchmarks.label_noise import label_noise
from disentanglement.benchmarks.ood_class_detection import plot_ood_class_detection
from disentanglement.experiment_configs import get_experiment_configs
from disentanglement.settings import TEST_MODE


def main():
    start_time = datetime.now()
    experiment_configs = get_experiment_configs()
    from_folder = False

    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    for experiment_config in tqdm(experiment_configs):
        if TEST_MODE:
            if experiment_config.dataset_name == "blobs":
                plot_decreasing_dataset(experiment_config, from_folder)
                label_noise(experiment_config, from_folder)
            if experiment_config.dataset_name == "CIFAR10":
                plot_ood_class_detection(experiment_config, from_folder)

        else:
            plot_decreasing_dataset(experiment_config, from_folder)
            label_noise(experiment_config, from_folder)

            if experiment_config.dataset_name != "blobs":
                plot_ood_class_detection(experiment_config, from_folder)
    end_time = datetime.now()

    print(f"Ran all experiments in {end_time - start_time}")


if __name__ == "__main__":
    main()
