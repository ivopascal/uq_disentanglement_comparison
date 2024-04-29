import os

BATCH_SIZE = 256
NUM_SAMPLES = 100
NUM_DEEP_ENSEMBLE_ESTIMATORS = 10

TEST_MODE = True

FIGURE_FOLDER = os.path.join(os.path.dirname(__file__), '../figures')

if TEST_MODE:
    NUM_SAMPLES = 3
    NUM_DEEP_ENSEMBLE_ESTIMATORS = 2
