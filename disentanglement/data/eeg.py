from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder

from disentanglement.datatypes import Dataset


def get_eeg_data():
    dataset = BNCI2014_001()        # load dataset
    paradigm = MotorImagery(        # make paradigm, filter between 7.5 and 30 Hz
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None, resample=128
    )

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

    le = LabelEncoder()
    y = le.fit_transform(y)
    return Dataset(X[metadata.session == "0train"], y[metadata.session == "0train"],
                   X[metadata.session == "1test"], y[metadata.session == "1test"])
