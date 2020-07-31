import os
from setuptools import setup, find_packages

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

# load all data files recursively https://stackoverflow.com/a/36693250
model_files = package_files('alibi_test_models/models/')

setup(
    name='alibi-test-models',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'alibi',
        'numpy',
        'scikit-learn',
        'tensorflow>=2.0.0'
        ],
    package_data={'': model_files}
)
