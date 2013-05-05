# PyMC 3

[![Build Status](https://travis-ci.org/pymc-devs/pymc.png)](https://travis-ci.org/pymc-devs/pymc)

PyMC is a python module for Bayesian statistical modeling and model fitting which focuses on advanced Markov chain Monte Carlo fitting algorithms. Its flexibility and extensibility make it applicable to a large suite of problems.

## Features 

 * Intuitive model specification syntax, for example, `x ~ N(0,1)` translates to `x = Normal(0,1)`
 * Powerful sampling algorithms such as [Hamiltonian Monte Carlo](http://en.wikipedia.org/wiki/Hybrid_Monte_Carlo)
 * Easy optimization for finding the *maximum a posteriori* point
 * [Theano](http://deeplearning.net/software/theano/) features 
  * Numpy broadcasting and advanced indexing 
  * Linear algebra operators
  * Computation optimization and dynamic C compilation
 * Simple extensibility

## Guided Examples
 * [Tutorial model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc/pymc3/examples/tutorial.ipynb)
 * More advanced [Stochastic Volatility model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc/pymc3/examples/stochastic_volatility.ipynb)

## Installation 

```
git clone -b pymc3 git@github.com:pymc-devs/pymc.git
python pymc/setup.py install
```

### Optional

[`scikits.sparse`](https://github.com/njsmith/scikits-sparse) enables sparse scaling matrices which are useful for large problems.

Ubuntu:

```
sudo apt-get install libsuitesparse-dev 
pip install git+https://github.com/njsmith/scikits-sparse.git
```

## License 
[Apache License, Version 2.0](https://github.com/pymc-devs/pymc/blob/pymc3/LICENSE)
