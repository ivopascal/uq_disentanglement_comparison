from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


def get_eeg_data():
    dataset = BNCI2014_001()        # load dataset
    paradigm = MotorImagery(        # make paradigm, filter between 7.5 and 30 Hz
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    num_subjects = 9
    for subject_id in range(1, num_subjects + 1):
        X, y, metadata = paradigm.get_data(subject_id)