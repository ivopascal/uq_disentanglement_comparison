from datetime import datetime

from tqdm import tqdm

from disentanglement.benchmarks.decreasing_dataset import plot_decreasing_dataset
from disentanglement.benchmarks.label_noise import label_noise
from disentanglement.benchmarks.ood_class_detection import plot_ood_class_detection
from disentanglement.data.datasets import get_datasets
from disentanglement.settings import TEST_MODE


def main():
    start_time = datetime.now()
    for name, conf in tqdm(get_datasets().items()):
        if TEST_MODE:
            if name == "blobs":
                plot_decreasing_dataset(name, conf)
                label_noise(name, conf)
            if name == "CIFAR10":
                plot_ood_class_detection(name, conf)

        else:
            plot_decreasing_dataset(name, conf)
            label_noise(name, conf)

            if name != "blobs":
                plot_ood_class_detection(name, conf)
    end_time = datetime.now()

    print(f"Ran all experiments in {end_time - start_time}")


if __name__ == "__main__":
    main()
