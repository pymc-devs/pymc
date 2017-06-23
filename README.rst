.. image:: https://cdn.rawgit.com/pymc-devs/pymc3/master/docs/logos/svg/PyMC3_banner.svg
    :height: 100px
    :alt: PyMC3 logo
    :align: center

|Build Status| |Coverage|

PyMC3 is a Python package for Bayesian statistical modeling and Probabilistic Machine Learning
which focuses on advanced Markov chain Monte Carlo and variational fitting
algorithms. Its flexibility and extensibility make it applicable to a
large suite of problems.

Check out the `getting started
guide <http://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`__!

Features
========

-  Intuitive model specification syntax, for example, ``x ~ N(0,1)``
   translates to ``x = Normal('x',0,1)``
-  **Powerful sampling algorithms**, such as the `No U-Turn
   Sampler <http://www.jmlr.org/papers/v15/hoffman14a.html>`__, allow complex models
   with thousands of parameters with little specialized knowledge of
   fitting algorithms.
-  **Variational inference**: `ADVI <http://www.jmlr.org/papers/v18/16-107.html>`__
   for fast approximate posterior estimation as well as mini-batch ADVI
   for large data sets.
-  Relies on `Theano <http://deeplearning.net/software/theano/>`__ which provides:
    *  Computation optimization and dynamic C compilation
    *  Numpy broadcasting and advanced indexing
    *  Linear algebra operators
    *  Simple extensibility
-  Transparent support for missing value imputation

Getting started
===============

If you already know about Bayesian statistics:
----------------------------------------------

-  `API quickstart guide <http://pymc-devs.github.io/pymc3/notebooks/api_quickstart.html>`__
-  The `PyMC3 tutorial <http://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`__
-  `PyMC3 examples <http://pymc-devs.github.io/pymc3/examples.html>`__
   and the `API reference <http://pymc-devs.github.io/pymc3/api.html>`__

Learn Bayesian statistics with a book together with PyMC3:
----------------------------------------------------------

-  `Probabilistic Programming and Bayesian Methods for Hackers <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`__: Fantastic book with many applied code examples.
-  `PyMC3 port of the book "Doing Bayesian Data Analysis" by John Kruschke <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__ as well as the `second edition <https://github.com/JWarmenhoven/DBDA-python>`__: Principled introduction to Bayesian data analysis.
-  `PyMC3 port of the book "Statistical Rethinking A Bayesian Course with Examples in R and Stan" by Richard McElreath <https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3>`__
-  `PyMC3 port of the book "Bayesian Cognitive Modeling" by Michael Lee and EJ Wagenmakers <https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3>`__: Focused on using Bayesian statistics in cognitive modeling.
-  `Bayesian Analysis with Python by Osvaldo Martin <https://www.packtpub.com/big-data-and-business-intelligence/bayesian-analysis-python>`__ (and `errata <https://github.com/aloctavodia/BAP>`__): Great introductory book.

PyMC3 talks
-----------

There are also several talks on PyMC3 which are gathered in this `YouTube playlist <https://www.youtube.com/playlist?list=PL1Ma_1DBbE82OVW8Fz_6Ts1oOeyOAiovy>`__

Installation
============

The latest release of PyMC3 can be installed from PyPI using ``pip``:

::

    pip install pymc3

**Note:** Running ``pip install pymc`` will install PyMC 2.3, not PyMC3,
from PyPI.

Or via conda-forge:

::

    conda install -c conda-forge pymc3

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
============

PyMC3 is tested on Python 2.7 and 3.6 and depends on Theano, NumPy,
SciPy, Pandas, and Matplotlib (see ``requirements.txt`` for version
information).

Optional
--------

In addtion to the above dependencies, the GLM submodule relies on
`Patsy <http://patsy.readthedocs.io/en/latest/>`__.

`scikits.sparse <https://github.com/njsmith/scikits-sparse>`__
enables sparse scaling matrices which are useful for large problems.

Citing PyMC3
============

Salvatier J, Wiecki TV, Fonnesbeck C. (2016) Probabilistic programming
in Python using PyMC3. PeerJ Computer Science 2:e55
https://doi.org/10.7717/peerj-cs.55

Contact
=======

We are using `discourse.pymc.io <https://discourse.pymc.io/>`__ as our main communication channel. You can also follow us on `Twitter @pymc_devs <https://twitter.com/pymc_devs>`__ for updates and other announcements. 

To ask a question regarding modeling or usage of PyMC3 we encourage posting to our Discourse forum under the `“Questions” Category <https://discourse.pymc.io/c/questions>`__. You can also suggest feature in the `“Development” Category <https://discourse.pymc.io/c/development>`__.

To report an issue with PyMC3 please use the `issue tracker <https://github.com/pymc-devs/pymc3/issues>`__.

Finally, if you need to get in touch for non-technical information about the project, `send us an e-mail <pymc.devs@gmail.com>`__.

License
=======

`Apache License, Version
2.0 <https://github.com/pymc-devs/pymc3/blob/master/LICENSE>`__


Software using PyMC3
====================

 - `sampled <https://github.com/ColCarroll/sampled>`__: Decorator for PyMC3.
 - `Bambi <https://github.com/bambinos/bambi>`__: BAyesian Model-Building Interface (BAMBI) in Python.
 - `gelato <https://github.com/ferrine/gelato>`__: Bayesian Neural Networks with PyMC3 and Lasagne.
 - `NiPyMC <https://github.com/PsychoinformaticsLab/nipymc>`__: Bayesian mixed-effects modeling of fMRI data in Python.
 - `beat <https://github.com/hvasbath/beat>`__: Bayesian Earthquake Analysis Tool.


Please contact us if your software is not listed here.

Papers citing PyMC3
===================

See `Google Scholar <https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011>`__ for a continuously updated list.

Contributors
============

See the `GitHub contributor
page <https://github.com/pymc-devs/pymc3/graphs/contributors>`__

Support
=======

PyMC3 is a non-profit project under NumFOCUS umbrella. If you want to support PyMC3 financially, you can donate `here <https://www.flipcause.com/widget/widget_home/MTE4OTc=>`__.

Sponsors
========

|NumFOCUS|

|Quantopian|

.. |Build Status| image:: https://travis-ci.org/pymc-devs/pymc3.png?branch=master
   :target: https://travis-ci.org/pymc-devs/pymc3
.. |Coverage| image:: https://coveralls.io/repos/github/pymc-devs/pymc3/badge.svg?branch=master
   :target: https://coveralls.io/github/pymc-devs/pymc3?branch=master
.. |NumFOCUS| image:: https://www.numfocus.org/wp-content/uploads/2017/03/1457562110.png
   :target: http://www.numfocus.org/
.. |Quantopian| image:: https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/quantopianlogo.jpg
   :target: https://quantopian.com
