import numpy as np
from disentanglement.experiment_configs import get_experiment_configs
from tqdm import tqdm

TQDM = tqdm(total=np.sum([len(experiment_config.models) for experiment_config in get_experiment_configs()[:1]]) * 3)
