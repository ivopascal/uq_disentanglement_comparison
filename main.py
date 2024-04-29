from tqdm import tqdm

from disentanglement.benchmarks.decreasing_dataset import plot_decreasing_dataset
from disentanglement.benchmarks.label_noise import label_noise
from disentanglement.benchmarks.ood_class_detection import ood_class_detection
from disentanglement.data.datasets import get_datasets
from disentanglement.settings import TEST_MODE


def main():
    for name, conf in tqdm(get_datasets().items()):
        if TEST_MODE:
            if name == "blobs":
                plot_decreasing_dataset(name, conf)
                label_noise(name, conf)
            if name == "CIFAR10":
                ood_class_detection(name, conf)

        else:
            plot_decreasing_dataset(name, conf)
            label_noise(name, conf)
            ood_class_detection(name, conf)


if __name__ == "__main__":
    main()
