import numpy as np


def normalise(x):
    x = np.array(x)
    return (x - min(x)) / max(x - min(x))