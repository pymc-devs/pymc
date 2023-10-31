.. image:: https://cdn.rawgit.com/pymc-devs/pymc/main/docs/logos/svg/PyMC_banner.svg
    :height: 100px
    :alt: PyMC logo
    :align: center

|Build Status| |Coverage| |NumFOCUS_badge| |Binder| |Dockerhub| |DOIzenodo|

PyMC (formerly PyMC3) is a Python package for Bayesian statistical modeling
focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI)
algorithms. Its flexibility and extensibility make it applicable to a
large suite of problems.

Check out the `PyMC overview <https://docs.pymc.io/en/latest/learn/core_notebooks/pymc_overview.html>`__,  or
one of `the many examples <https://www.pymc.io/projects/examples/en/latest/gallery.html>`__!
For questions on PyMC, head on over to our `PyMC Discourse <https://discourse.pymc.io/>`__ forum.

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
-  Relies on `PyTensor <https://pytensor.readthedocs.io/en/latest/>`__ which provides:
    *  Computation optimization and dynamic C or JAX compilation
    *  NumPy broadcasting and advanced indexing
    *  Linear algebra operators
    *  Simple extensibility
-  Transparent support for missing value imputation

Getting started
===============

If you already know about Bayesian statistics:
----------------------------------------------

-  `API quickstart guide <https://www.pymc.io/projects/examples/en/latest/howto/api_quickstart.html>`__
-  The `PyMC tutorial <https://docs.pymc.io/en/latest/learn/core_notebooks/pymc_overview.html>`__
-  `PyMC examples <https://www.pymc.io/projects/examples/en/latest/gallery.html>`__ and the `API reference <https://docs.pymc.io/en/stable/api.html>`__

Learn Bayesian statistics with a book together with PyMC
--------------------------------------------------------

-  `Probabilistic Programming and Bayesian Methods for Hackers <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`__: Fantastic book with many applied code examples.
-  `PyMC port of the book "Doing Bayesian Data Analysis" by John Kruschke <https://github.com/cluhmann/DBDA-python>`__ as well as the `first edition <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__.
-  `PyMC port of the book "Statistical Rethinking A Bayesian Course with Examples in R and Stan" by Richard McElreath <https://github.com/pymc-devs/resources/tree/master/Rethinking>`__
-  `PyMC port of the book "Bayesian Cognitive Modeling" by Michael Lee and EJ Wagenmakers <https://github.com/pymc-devs/resources/tree/master/BCM>`__: Focused on using Bayesian statistics in cognitive modeling.
-  `Bayesian Analysis with Python  <https://www.packtpub.com/big-data-and-business-intelligence/bayesian-analysis-python-second-edition>`__ (second edition) by Osvaldo Martin: Great introductory book. (`code <https://github.com/aloctavodia/BAP>`__ and errata).

Audio & Video
-------------

- Here is a `YouTube playlist <https://www.youtube.com/playlist?list=PL1Ma_1DBbE82OVW8Fz_6Ts1oOeyOAiovy>`__ gathering several talks on PyMC.
- You can also find all the talks given at **PyMCon 2020** `here <https://discourse.pymc.io/c/pymcon/2020talks/15>`__.
- The `"Learning Bayesian Statistics" podcast <https://www.learnbayesstats.com/>`__ helps you discover and stay up-to-date with the vast Bayesian community. Bonus: it's hosted by Alex Andorra, one of the PyMC core devs!

Installation
============

To install PyMC on your system, follow the instructions on the `installation guide <https://www.pymc.io/projects/docs/en/latest/installation.html>`__.

Citing PyMC
===========
Please choose from the following:

- |DOIpaper| *PyMC: A Modern and Comprehensive Probabilistic Programming Framework in Python*, Abril-Pla O, Andreani V, Carroll C, Dong L, Fonnesbeck CJ, Kochurov M, Kumar R, Lao J, Luhmann CC, Martin OA, Osthege M, Vieira R, Wiecki T, Zinkov R. (2023)
- |DOIzenodo| A DOI for all versions.
- DOIs for specific versions are shown on Zenodo and under `Releases <https://github.com/pymc-devs/pymc/releases>`_

.. |DOIpaper| image:: https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.1516-blue
     :target: https://doi.org/10.7717/peerj-cs.1516
.. |DOIzenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4603970.svg
   :target: https://doi.org/10.5281/zenodo.4603970

Contact
=======

We are using `discourse.pymc.io <https://discourse.pymc.io/>`__ as our main communication channel.

To ask a question regarding modeling or usage of PyMC we encourage posting to our Discourse forum under the `“Questions” Category <https://discourse.pymc.io/c/questions>`__. You can also suggest feature in the `“Development” Category <https://discourse.pymc.io/c/development>`__.

You can also follow us on these social media platforms for updates and other announcements:

- `LinkedIn @pymc <https://www.linkedin.com/company/pymc/>`__
- `YouTube @PyMCDevelopers <https://www.youtube.com/c/PyMCDevelopers>`__
- `Twitter @pymc_devs <https://twitter.com/pymc_devs>`__
- `Mastodon @pymc@bayes.club <https://bayes.club/@pymc>`__

To report an issue with PyMC please use the `issue tracker <https://github.com/pymc-devs/pymc/issues>`__.

Finally, if you need to get in touch for non-technical information about the project, `send us an e-mail <info@pymc-devs.org>`__.

License
=======

`Apache License, Version
2.0 <https://github.com/pymc-devs/pymc/blob/main/LICENSE>`__


Software using PyMC
===================

General purpose
---------------

- `Bambi <https://github.com/bambinos/bambi>`__: BAyesian Model-Building Interface (BAMBI) in Python.
- `calibr8 <https://calibr8.readthedocs.io>`__: A toolbox for constructing detailed observation models to be used as likelihoods in PyMC.
- `gumbi <https://github.com/JohnGoertz/Gumbi>`__: A high-level interface for building GP models.
- `SunODE <https://github.com/aseyboldt/sunode>`__: Fast ODE solver, much faster than the one that comes with PyMC.
- `pymc-learn <https://github.com/pymc-learn/pymc-learn>`__: Custom PyMC models built on top of pymc3_models/scikit-learn API

Domain specific
---------------

- `Exoplanet <https://github.com/dfm/exoplanet>`__: a toolkit for modeling of transit and/or radial velocity observations of exoplanets and other astronomical time series.
- `beat <https://github.com/hvasbath/beat>`__: Bayesian Earthquake Analysis Tool.
- `CausalPy <https://github.com/pymc-labs/CausalPy>`__: A package focussing on causal inference in quasi-experimental settings.

Please contact us if your software is not listed here.

Papers citing PyMC
==================

See `Google Scholar <https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011>`__ for a continuously updated list.

Contributors
============

See the `GitHub contributor
page <https://github.com/pymc-devs/pymc/graphs/contributors>`__. Also read our `Code of Conduct <https://github.com/pymc-devs/pymc/blob/main/CODE_OF_CONDUCT.md>`__ guidelines for a better contributing experience.

Support
=======

PyMC is a non-profit project under NumFOCUS umbrella. If you want to support PyMC financially, you can donate `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.

Professional Consulting Support
===============================

You can get professional consulting support from `PyMC Labs <https://www.pymc-labs.io>`__.

Sponsors
========

|NumFOCUS|

|PyMCLabs|

|Mistplay|

|ODSC|

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pymc-devs/pymc/main?filepath=%2Fdocs%2Fsource%2Fnotebooks
.. |Build Status| image:: https://github.com/pymc-devs/pymc/workflows/pytest/badge.svg
   :target: https://github.com/pymc-devs/pymc/actions
.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pymc/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/pymc-devs/pymc
.. |Dockerhub| image:: https://img.shields.io/docker/automated/pymc/pymc.svg
   :target: https://hub.docker.com/r/pymc/pymc
.. |NumFOCUS_badge| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: http://www.numfocus.org/
.. |NumFOCUS| image:: https://github.com/pymc-devs/brand/blob/main/sponsors/sponsor_logos/sponsor_numfocus.png?raw=true
   :target: http://www.numfocus.org/
.. |PyMCLabs| image:: https://github.com/pymc-devs/brand/blob/main/sponsors/sponsor_logos/sponsor_pymc_labs.png?raw=true
   :target: https://pymc-labs.io
.. |Mistplay| image:: https://github.com/pymc-devs/brand/blob/main/sponsors/sponsor_logos/sponsor_mistplay.png?raw=true
   :target: https://www.mistplay.com/
.. |ODSC| image:: https://github.com/pymc-devs/brand/blob/main/sponsors/sponsor_logos/odsc/sponsor_odsc.png?raw=true
   :target: https://odsc.com/california/?utm_source=pymc&utm_medium=referral
