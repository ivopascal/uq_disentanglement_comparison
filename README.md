# Uncertainty Disentanglement
Uncertainty Quantification methods for Neural Networks can make separate predictions for aleatoric and epistemic uncertainty.
This repository implements three experiments that manipulate the underlying aleatoric or epistemic uncertainty, and observes adverse interactions.

The experiments are conducted with different datasets, UQ methods, and disentanglement methods.

# Installation

Create a virtual environment and install dependencies with Pipenv
```bash
python3 -m venv ./venv
source ./venv/bin/activation
pip install pipenv
pipenv install
```

# Usage
You can run the experiments with the simple command `python3 main.py`.
Out of the box, this will run all experiments and take several days to complete.
We recommend first modifying the `disentanglement.experiment_configs.get_experiment_configs()` to select the conditions you are interested in.

Additionally `disentanglement/settings.py` has various hyperparameters that can be used to modify the experiment setup. 
It includes a `TEST_MODE` which runs through all experiments but changes hyperparameters to minimize compute cost. 
The results become meaningless, but will show whether things work.


# Navigating the project
The project comes with three main components:
- The experiments and plotting in `disentanglement/benchmarks`
- The data loading and preprocessing in `disentanglement/data`
- The UQ methods, architectures and disentangling methods in `disentanglement/models`

These all come together in the `experiments_configs.py` which gives the set of benchmarks, models, and datasets to run together.
By default, all results are stored to file through `util.py`. If the results already exist, the experiment will skip over those experiments.

Additionally, the `notebooks` folder has some jupyter notebooks which contain smaller experiments and additional investigations. 

# Attribution and context

This repository was built to run the experiments for a paper that is currently under review.
The paper gives further introduction to the problem and an interpretation of the results. The paper is available upon request. Previous results as figures or data files are also available upon request.

For attribution, citing the to-be-reviewed paper is appreciated.

