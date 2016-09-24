PyMC3
=====

|Gitter| |Build Status| |Coverage|

PyMC3 is a python module for Bayesian statistical modeling and model
fitting which focuses on advanced Markov chain Monte Carlo fitting
algorithms. Its flexibility and extensibility make it applicable to a
large suite of problems.

Check out the `getting started
guide <http://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`__!

Features
--------

-  Intuitive model specification syntax, for example, ``x ~ N(0,1)``
   translates to ``x = Normal(0,1)``
-  **Powerful sampling algorithms**, such as the `No U-Turn
   Sampler <http://arxiv.org/abs/1111.4246>`__, allow complex models
   with thousands of parameters with little specialized knowledge of
   fitting algorithms.
-  **Variational inference**: `ADVI <http://arxiv.org/abs/1506.03431>`__
   for fast approximate posterior estimation as well as mini-batch ADVI
   for large data sets.
-  Easy optimization for finding the *maximum a posteriori* (MAP) point
-  `Theano <http://deeplearning.net/software/theano/>`__ features
-  Numpy broadcasting and advanced indexing
-  Linear algebra operators
-  Computation optimization and dynamic C compilation
-  Simple extensibility
-  Transparent support for missing value imputation

Getting started
---------------

-  The `PyMC3
   tutorial <http://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`__ or
   `journal publication <https://peerj.com/articles/cs-55/>`__
-  `PyMC3 examples <http://pymc-devs.github.io/pymc3/examples.html>`__
   and the `API reference <http://pymc-devs.github.io/pymc3/api.html>`__
-  `Bayesian Modelling in Python -- tutorials on Bayesian statistics and
   PyMC3 as Jupyter Notebooks by Mark
   Dregan <https://github.com/markdregan/Bayesian-Modelling-in-Python>`__
-  `Talk at PyData London 2016 on
   PyMC3 <https://www.youtube.com/watch?v=LlzVlqVzeD8>`__
-  `PyMC3 port of the models presented in the book "Doing Bayesian Data
   Analysis" by John
   Kruschke <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__
-  Coal Mining Disasters model in `PyMC
   2 <https://github.com/pymc-devs/pymc/blob/master/pymc/examples/disaster_model.py>`__
   and `PyMC
   3 <https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/disaster_model.py>`__

Installation
------------

The latest release of PyMC3 can be installed from PyPI using ``pip``:

::

    pip install pymc3

**Note:** Running ``pip install pymc`` will install PyMC 2.3, not PyMC3,
from PyPI.

The current development branch of PyMC3 can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/pymc-devs/pymc3

To ensure the development branch of Theano is installed alongside PyMC3
(recommended), you can install PyMC3 using the ``requirements.txt``
file. This requires cloning the repository to your computer:

::

    git clone https://github.com/pymc-devs/pymc3
    cd pymc3
    pip install -r requirements.txt

However, if a recent version of Theano has already been installed on
your system, you can install PyMC3 directly from GitHub.

Another option is to clone the repository and install PyMC3 using
``python setup.py install`` or ``python setup.py develop``.


Dependencies
------------

PyMC3 is tested on Python 2.7 and 3.4 and depends on Theano, NumPy,
SciPy, Pandas, and Matplotlib (see ``requirements.txt`` for version
information).

Optional
~~~~~~~~

In addtion to the above dependencies, the GLM submodule relies on
``Patsy``\ [http://patsy.readthedocs.io/en/latest/].

```scikits.sparse`` <https://github.com/njsmith/scikits-sparse>`__
enables sparse scaling matrices which are useful for large problems.
Installation on Ubuntu is easy:

::

    sudo apt-get install libsuitesparse-dev
    pip install git+https://github.com/njsmith/scikits-sparse.git

On Mac OS X you can install libsuitesparse 4.2.1 via homebrew (see
http://brew.sh/ to install homebrew), manually add a link so the include
files are where scikits-sparse expects them, and then install
scikits-sparse:

::

    brew install suite-sparse
    ln -s /usr/local/Cellar/suite-sparse/4.2.1/include/ /usr/local/include/suitesparse
    pip install git+https://github.com/njsmith/scikits-sparse.git


Citing PyMC3
------------

Salvatier J, Wiecki TV, Fonnesbeck C. (2016) Probabilistic programming
in Python using PyMC3. PeerJ Computer Science 2:e55
https://doi.org/10.7717/peerj-cs.55

Coyle P. (2016) Probabilistic programming
and PyMC3. European Scientific Python Conference 2015 (Cambridge, UK)
http://adsabs.harvard.edu/abs/2016arXiv160700379C

License
-------

`Apache License, Version
2.0 <https://github.com/pymc-devs/pymc3/blob/master/LICENSE>`__


Contributors
------------

See the `GitHub contributor
page <https://github.com/pymc-devs/pymc3/graphs/contributors>`__

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/pymc-devs/pymc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Build Status| image:: https://travis-ci.org/pymc-devs/pymc3.png?branch=master
   :target: https://travis-ci.org/pymc-devs/pymc3
.. |Coverage| image:: https://coveralls.io/repos/github/pymc-devs/pymc3/badge.svg?branch=master
   :target: https://coveralls.io/github/pymc-devs/pymc3?branch=master 
