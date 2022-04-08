from setuptools import setup, find_packages
from os import path

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/gboehl/emcwrap",
    name="emcwrap",
    version="0.1.2",
    author="Gregor Boehl",
    author_email="admin@gregorboehl.com",
    description="Tools for Bayesian inference using Ensemble MCMC",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "emcee",
        "pandas",
        "numdifftools",
        "matplotlib",
        "grgrlib>=0.1.3",
    ],
)
