# PyMC3
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/pymc-devs/pymc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/pymc-devs/pymc3.png?branch=master)](https://travis-ci.org/pymc-devs/pymc3)

PyMC3 is a python module for Bayesian statistical modeling and model fitting which focuses on advanced Markov chain Monte Carlo fitting algorithms. Its flexibility and extensibility make it applicable to a large suite of problems.

Check out the [getting started guide](http://pymc-devs.github.io/pymc3/notebooks/getting_started.html)!

PyMC3 is beta software. Users should consider using [PyMC 2 repository](https://github.com/pymc-devs/pymc).

## Features

 * Intuitive model specification syntax, for example, `x ~ N(0,1)` translates to `x = Normal(0,1)`
 * **Powerful sampling algorithms**, such as the [No U-Turn Sampler](http://arxiv.org/abs/1111.4246), allow complex models with thousands of parameters with little specialized knowledge of fitting algorithms.
 * **Variational inference**: [ADVI](http://arxiv.org/abs/1506.03431) for fast approximate posterior estimation as well as mini-batch ADVI for large data sets.
 * Easy optimization for finding the *maximum a posteriori* (MAP) point
 * [Theano](http://deeplearning.net/software/theano/) features
  * Numpy broadcasting and advanced indexing
  * Linear algebra operators
  * Computation optimization and dynamic C compilation
 * Simple extensibility
 * Transparent support for missing value imputation

## Getting started
 * The [PyMC3 tutorial](http://pymc-devs.github.io/pymc3/getting_started) or [journal publication](https://peerj.com/articles/cs-55/)
 * [PyMC3 examples](http://pymc-devs.github.io/pymc3/examples.html) and the [API reference](http://pymc-devs.github.io/pymc3/api.html)
 * [Bayesian Modelling in Python -- tutorials on Bayesian statistics and PyMC3 as Jupyter Notebooks by Mark Dregan](https://github.com/markdregan/Bayesian-Modelling-in-Python)
 * [Talk at PyData London 2016 on PyMC3](https://www.youtube.com/watch?v=LlzVlqVzeD8)
 * [PyMC3 port of the models presented in the book "Doing Bayesian Data Analysis" by John Kruschke](https://github.com/aloctavodia/Doing_bayesian_data_analysis)
 * Coal Mining Disasters model in [PyMC 2](https://github.com/pymc-devs/pymc/blob/master/pymc/examples/disaster_model.py) and [PyMC 3](https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/disaster_model.py)

## Installation

The latest version of PyMC3 can be installed from the master branch using pip:

```
pip install git+https://github.com/pymc-devs/pymc3
```

To ensure the development branch of Theano is installed alongside PyMC3 (recommended), you can install PyMC3 using the `requirements.txt` file. This requires cloning the repository to your computer:

```
git clone https://github.com/pymc-devs/pymc3
cd pymc3
pip install -r requirements.txt
```

However, if a recent version of Theano has already been installed on your system, you can install PyMC3 directly from GitHub.

Another option is to clone the repository and install PyMC3 using `python setup.py install` or `python setup.py develop`.

**Note:** Running `pip install pymc` will install PyMC 2.3, not PyMC3, from PyPI.

## Dependencies

PyMC3 is tested on Python 2.7 and 3.3 and depends on Theano, NumPy,
SciPy, Pandas, and Matplotlib (see `requirements.txt` for version information).

### Optional

In addtion to the above dependencies, the GLM submodule relies on
`Patsy`[http://patsy.readthedocs.io/en/latest/].

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

## Support

The best places to ask general questions about PyMC3 is on [StackOverflow](http://stackoverflow.com/questions/tagged/pymc) with the `pymc` tag, or on our [Gitter channel](https://gitter.im/pymc-devs/pymc). If you have discovered a bug in PyMC3, please use our [issue tracker](https://github.com/pymc-devs/pymc3/issues).

## Citing PyMC3

Salvatier J, Wiecki TV, Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 https://doi.org/10.7717/peerj-cs.55

## Conference papers on PyMC3
Peadar Coyle wrote a peer-reviewed paper for the European Scientific Python Conference held in the Summer of 2015 in Cambridge, UK. 
@ARTICLE{2016arXiv160700379C,
   author = {{Coyle}, P.},
    title = "{Probabilistic Programming and PyMC3}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1607.00379},
 primaryClass = "cs.OH",
 keywords = {Computer Science - Other Computer Science},
     year = 2016,
    month = jul,
   adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160700379C},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

## License
[Apache License, Version 2.0](https://github.com/pymc-devs/pymc3/blob/master/LICENSE)

## Contributors

See the [GitHub contributor page](https://github.com/pymc-devs/pymc3/graphs/contributors)
