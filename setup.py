import os
from setuptools import setup, find_packages


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


# load all data files recursively https://stackoverflow.com/a/36693250
model_files = package_files('alibi_testing/models/')

# Optional deps. Deps that are already installed by both alibi and alibi-detect do not need to be duplicated here
extras_require = {
    'training': ['torchvision>=0.10.0, <1.0.0'], # deps required for training, but not to run models
#    'alibi': [], # deps required by alibi to run the alibi-testing models
#    'alibi-detect': [], # deps required by alibi-detect to run the alibi-testing models
}


setup(
    name='alibi-testing',
    version='0.0.11',
    packages=find_packages(),
    python_requires='>=3.7',
    extras_require=extras_require,
    install_requires=[],  # deps installed by both alibi and alibi-detect do not need to be duplicated here
    package_data={'': model_files}
)
