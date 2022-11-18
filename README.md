Repository for hosting pre-trained models and data loading functions used for testing.
Due to the Github 100MB file limit only relatively small models can be stored.

The models are part of the package `alibi-testing`. The package also exposes a single function
`load(name: str)` at top level for loading pre-trained TensorFlow and PyTorch models.
The intention is to use this package and the `load` functionality as a dependency for running tests for `alibi`
and `alibi-detect`. 

# Installation

The repo is installed by `alibi` and `alibi-detect` for testing purposes. If installing this repo locally 
in order to train new models (via the training scripts in `scripts/`), it should be installed by running:

```
pip install .[training]
```
