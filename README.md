Repository for hosting pre-trained models used for testing.
Due to the Github 100MB file limit only relatively small models can be stored.

The models are part of the package `alibi-test-models`. The package also exposes a single function
`load(name: str)` at top level for loading pre-trained (currently only TensorFlow) models.
The intention is to use this package and the `load` functionality as a dependency for running tests for `alibi`.
