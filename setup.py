from setuptools import setup, find_packages

setup(
    name='alibi-test-models',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['tensorflow>=2.0.0'],
    package_data={'': ['models/*']}
)
