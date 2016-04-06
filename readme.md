# PyMC3
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/pymc-devs/pymc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/pymc-devs/pymc3.png?branch=master)](https://travis-ci.org/pymc-devs/pymc3)

PyMC3 is a python module for Bayesian statistical modeling and model fitting which focuses on advanced Markov chain Monte Carlo fitting algorithms. Its flexibility and extensibility make it applicable to a large suite of problems.

Check out the [Tutorial](http://pymc-devs.github.io/pymc3/getting_started/)!

PyMC3 is Beta software. Users should consider using [PyMC 2 repository](https://github.com/pymc-devs/pymc).

## Features

 * Intuitive model specification syntax, for example, `x ~ N(0,1)` translates to `x = Normal(0,1)`
 * Powerful sampling algorithms, such as the [No U-Turn Sampler](http://arxiv.org/abs/1111.4246), allow complex models with thousands of parameters with little specialized knowledge of fitting algorithms.
 * Easy optimization for finding the *maximum a posteriori*(MAP) point
 * [Theano](http://deeplearning.net/software/theano/) features
  * Numpy broadcasting and advanced indexing
  * Linear algebra operators
  * Computation optimization and dynamic C compilation
 * Simple extensibility
 * Transparent support for missing value imputation

## Getting started
 * [PyMC3 Tutorial](http://pymc-devs.github.io/pymc3/getting_started/)
 * [PyMC3 paper](https://peerj.com/articles/cs-55/)
 * [Bayesian Modelling in Python -- tutorials on Bayesian statistics and PyMC3 as Jupyter Notebooks by Mark Dregan](https://github.com/markdregan/Bayesian-Modelling-in-Python)
 * Coal Mining Disasters model in [PyMC 2](https://github.com/pymc-devs/pymc/blob/master/pymc/examples/disaster_model.py) and [PyMC 3](https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/disaster_model.py)
 * [Global Health Metrics & Evaluation model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc3/master/pymc3/examples/GHME%202013.ipynb) case study for GHME 2013
 * [Stochastic Volatility model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc3/master/pymc3/examples/stochastic_volatility.ipynb)
 * [Several blog posts on linear regression](http://twiecki.github.io/tag/bayesian-statistics.html)
 * [Talk at PyData NYC 2013 on PyMC3](http://twiecki.github.io/blog/2013/12/12/bayesian-data-analysis-pymc3/)
 * [PyMC3 port of the models presented in the book "Doing Bayesian Data Analysis" by John Kruschke](https://github.com/aloctavodia/Doing_bayesian_data_analysis)
 * [The PyMC3 examples folder](https://github.com/pymc-devs/pymc3/tree/master/pymc3/examples)
 * [Manual of PyMC3 distributions](http://pymc-devs.github.io/pymc3/manual/api.html#distributions)

## Installation

The latest version of PyMC3 can be installed from the master branch using pip:

```
pip install --process-dependency-links git+https://github.com/pymc-devs/pymc3
```

The `--process-dependency-links` flag ensures that the developmental branch of Theano, which PyMC3 requires, is installed. If a recent developmental version of Theano has been installed with another method, this flag can be dropped.

Another option is to clone the repository and install PyMC3 using `python setup.py install` or `python setup.py develop`.

**Note:** Running `pip install pymc` will install PyMC 2.3, not PyMC3, from PyPI.

## Dependencies

PyMC3 is tested on Python 2.7 and 3.3 and depends on Theano, NumPy,
SciPy, Pandas, and Matplotlib (see setup.py for version information).

### Optional

In addtion to the above dependencies, the GLM submodule relies on
Patsy.

[`scikits.sparse`](https://github.com/njsmith/scikits-sparse) enables sparse scaling matrices which are useful for large problems. Installation on Ubuntu is easy:

```
sudo apt-get install libsuitesparse-dev
pip install git+https://github.com/njsmith/scikits-sparse.git
```

On Mac OS X you can install libsuitesparse 4.2.1 via homebrew (see http://brew.sh/ to install homebrew), manually add a link so the include files are where scikits-sparse expects them, and then install scikits-sparse:

```
brew install suite-sparse
ln -s /usr/local/Cellar/suite-sparse/4.2.1/include/ /usr/local/include/suitesparse
pip install git+https://github.com/njsmith/scikits-sparse.git
```

## Citing PyMC3

Salvatier J, Wiecki TV, Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 https://doi.org/10.7717/peerj-cs.55

## License
[Apache License, Version 2.0](https://github.com/pymc-devs/pymc3/blob/master/LICENSE)

## Contributors

See the [GitHub contributor page](https://github.com/pymc-devs/pymc3/graphs/contributors)
