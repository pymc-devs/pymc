.. image:: https://cdn.rawgit.com/pymc-devs/pymc3/master/docs/logos/svg/PyMC3_banner.svg
    :height: 100px
    :alt: PyMC3 logo
    :align: center

|Build Status| |Coverage| |NumFOCUS_badge| |Binder| |Dockerhub|

PyMC3 is a Python package for Bayesian statistical modeling and Probabilistic Machine Learning
focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI)
algorithms. Its flexibility and extensibility make it applicable to a
large suite of problems.

Check out the `getting started guide <http://docs.pymc.io/notebooks/getting_started>`__,  or
`interact with live examples <https://mybinder.org/v2/gh/pymc-devs/pymc3/master?filepath=%2Fdocs%2Fsource%2Fnotebooks>`__
using Binder!
For questions on PyMC3, head on over to our `PyMC Discourse <https://discourse.pymc.io/>`__ forum.

The future of PyMC3 & Theano
============================

There have been many questions and uncertainty around the future of PyMC3 since Theano
stopped getting developed by the original authors, and we started experiments with PyMC4.

We are happy to announce that PyMC3 on Theano (which we are `developing further <https://github.com/pymc-devs/Theano-PyMC>`__)
with a new JAX backend is the future. PyMC4 will not be developed further.

See the `full announcement <https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b>`__
for more details.

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
-  Relies on `Theano-PyMC <https://theano-pymc.readthedocs.io/en/latest/>`__ which provides:
    *  Computation optimization and dynamic C or JAX compilation
    *  Numpy broadcasting and advanced indexing
    *  Linear algebra operators
    *  Simple extensibility
-  Transparent support for missing value imputation

Getting started
===============

If you already know about Bayesian statistics:
----------------------------------------------


-  `API quickstart guide <http://docs.pymc.io/notebooks/api_quickstart>`__
-  The `PyMC3 tutorial <http://docs.pymc.io/notebooks/getting_started>`__
-  `PyMC3 examples <https://docs.pymc.io/nb_examples/index.html>`__ and the `API reference <http://docs.pymc.io/api>`__



Learn Bayesian statistics with a book together with PyMC3:
----------------------------------------------------------

-  `Probabilistic Programming and Bayesian Methods for Hackers <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`__: Fantastic book with many applied code examples.
-  `PyMC3 port of the book "Doing Bayesian Data Analysis" by John Kruschke <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__ as well as the `second edition <https://github.com/JWarmenhoven/DBDA-python>`__: Principled introduction to Bayesian data analysis.
-  `PyMC3 port of the book "Statistical Rethinking A Bayesian Course with Examples in R and Stan" by Richard McElreath <https://github.com/pymc-devs/resources/tree/master/Rethinking>`__
-  `PyMC3 port of the book "Bayesian Cognitive Modeling" by Michael Lee and EJ Wagenmakers <https://github.com/pymc-devs/resources/tree/master/BCM>`__: Focused on using Bayesian statistics in cognitive modeling.
-  `Bayesian Analysis with Python  <https://www.packtpub.com/big-data-and-business-intelligence/bayesian-analysis-python-second-edition>`__ (second edition) by Osvaldo Martin: Great introductory book. (`code <https://github.com/aloctavodia/BAP>`__ and errata).

PyMC3 talks
-----------

There are also several talks on PyMC3 which are gathered in this `YouTube playlist <https://www.youtube.com/playlist?list=PL1Ma_1DBbE82OVW8Fz_6Ts1oOeyOAiovy>`__
and as part of `PyMCon 2020 <https://discourse.pymc.io/c/pymcon/2020talks/15>`__

Installation
============

To install PyMC3 on your system, follow the instructions on the appropriate installation guide:

-  `Installing PyMC3 on MacOS <https://github.com/pymc-devs/pymc3/wiki/Installation-Guide-(MacOS)>`__
-  `Installing PyMC3 on Linux <https://github.com/pymc-devs/pymc3/wiki/Installation-Guide-(Linux)>`__
-  `Installing PyMC3 on Windows <https://github.com/pymc-devs/pymc3/wiki/Installation-Guide-(Windows)>`__


Citing PyMC3
============

Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming
in Python using PyMC3. PeerJ Computer Science 2:e55
`DOI: 10.7717/peerj-cs.55 <https://doi.org/10.7717/peerj-cs.55>`__.

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

- `Exoplanet <https://github.com/dfm/exoplanet>`__: a toolkit for modeling of transit and/or radial velocity observations of exoplanets and other astronomical time series.
- `Bambi <https://github.com/bambinos/bambi>`__: BAyesian Model-Building Interface (BAMBI) in Python.
- `pymc3_models <https://github.com/parsing-science/pymc3_models>`__: Custom PyMC3 models built on top of the scikit-learn API.
- `PMProphet <https://github.com/luke14free/pm-prophet>`__: PyMC3 port of Facebook's Prophet model for timeseries modeling
- `webmc3 <https://github.com/AustinRochford/webmc3>`__: A web interface for exploring PyMC3 traces
- `sampled <https://github.com/ColCarroll/sampled>`__: Decorator for PyMC3 models.
- `NiPyMC <https://github.com/PsychoinformaticsLab/nipymc>`__: Bayesian mixed-effects modeling of fMRI data in Python.
- `beat <https://github.com/hvasbath/beat>`__: Bayesian Earthquake Analysis Tool.
- `pymc-learn <https://github.com/pymc-learn/pymc-learn>`__: Custom PyMC models built on top of pymc3_models/scikit-learn API
- `fenics-pymc3 <https://github.com/IvanYashchuk/fenics-pymc3>`__: Differentiable interface to FEniCS, a library for solving partial differential equations.
- `cell2location <https://github.com/BayraktarLab/cell2location>`__: Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics.

Please contact us if your software is not listed here.

Papers citing PyMC3
===================

See `Google Scholar <https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011>`__ for a continuously updated list.

Contributors
============

See the `GitHub contributor
page <https://github.com/pymc-devs/pymc3/graphs/contributors>`__. Also read our `Code of Conduct <https://github.com/pymc-devs/pymc3/blob/master/CODE_OF_CONDUCT.md>`__ guidelines for a better contributing experience.

Support
=======

PyMC3 is a non-profit project under NumFOCUS umbrella. If you want to support PyMC3 financially, you can donate `here <https://numfocus.salsalabs.org/donate-to-pymc3/index.html>`__.

PyMC for enterprise
===================
`PyMC is now available as part of the Tidelift Subscription!`

Tidelift is working with PyMC and the maintainers of thousands of other open source
projects to deliver commercial support and maintenance for the open source dependencies
you use to build your applications. Save time, reduce risk, and improve code health,
while contributing financially to PyMC -- making it even more robust, reliable and,
let's face it, amazing!

|tidelift_learn| |tidelift_demo|

Sponsors
========

|NumFOCUS|

|Quantopian|

|ODSC|

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pymc-devs/pymc3/master?filepath=%2Fdocs%2Fsource%2Fnotebooks
.. |Build Status| image:: https://github.com/pymc-devs/pymc3/workflows/pytest/badge.svg
   :target: https://github.com/pymc-devs/pymc3/actions
.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pymc3/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pymc-devs/pymc3
.. |Dockerhub| image:: https://img.shields.io/docker/automated/pymc/pymc3.svg
  :target: https://hub.docker.com/r/pymc/pymc3
.. |NumFOCUS| image:: https://www.numfocus.org/wp-content/uploads/2017/03/1457562110.png
   :target: http://www.numfocus.org/
.. |Quantopian| image:: https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/quantopianlogo.jpg
   :target: https://quantopian.com
.. |NumFOCUS_badge| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: http://www.numfocus.org/
.. |ODSC| image:: https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/odsc_logo.png
   :target: https://odsc.com
.. |tidelift| image:: https://img.shields.io/badge/-lifted!-2dd160.svg?colorA=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/pkg/pypi-pymc3?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
.. |tidelift_learn| image:: https://img.shields.io/badge/-learn%20more-2dd160.svg?color=orange&labelColor=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/pkg/pypi-pymc3?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
.. |tidelift_demo| image:: https://img.shields.io/badge/-request%20a%20demo-2dd160.svg?color=orange&labelColor=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/request-a-demo?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
