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

setup(
    name='alibi-testing',
    version='0.0.4',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.2, <2.0.0',
        'scikit-learn>=0.20.2, <0.25.0',
        'tensorflow>=2.0.0, <2.5.0',
    ],
    package_data={'': model_files}
)
