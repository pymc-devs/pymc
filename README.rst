.. image:: https://cdn.rawgit.com/pymc-devs/pymc/main/docs/logos/svg/PyMC_banner.svg
    :height: 100px
    :alt: PyMC logo
    :align: center

|Build Status| |Coverage| |NumFOCUS_badge| |Binder| |Dockerhub| |DOIzenodo|

PyMC (formerly PyMC3) is a Python package for Bayesian statistical modeling
focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI)
algorithms. Its flexibility and extensibility make it applicable to a
large suite of problems.

Check out the `PyMC overview <https://docs.pymc.io/en/latest/learn/examples/pymc_overview.html>`__,  or
`interact with live examples <https://mybinder.org/v2/gh/pymc-devs/pymc/main?filepath=%2Fdocs%2Fsource%2Fnotebooks>`__
using Binder!
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
-  Relies on `Aesara <https://aesara.readthedocs.io/en/latest/>`__ which provides:
    *  Computation optimization and dynamic C or JAX compilation
    *  NumPy broadcasting and advanced indexing
    *  Linear algebra operators
    *  Simple extensibility
-  Transparent support for missing value imputation

Getting started
===============

If you already know about Bayesian statistics:
----------------------------------------------

-  `API quickstart guide <https://docs.pymc.io/en/stable/pymc-examples/examples/pymc3_howto/api_quickstart.html>`__
-  The `PyMC tutorial <https://docs.pymc.io/en/latest/learn/examples/pymc_overview.html>`__
-  `PyMC examples <https://docs.pymc.io/projects/examples/en/latest/>`__ and the `API reference <https://docs.pymc.io/en/stable/api.html>`__

Learn Bayesian statistics with a book together with PyMC
--------------------------------------------------------

-  `Probabilistic Programming and Bayesian Methods for Hackers <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`__: Fantastic book with many applied code examples.
-  `PyMC port of the book "Doing Bayesian Data Analysis" by John Kruschke <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__ as well as the `second edition <https://github.com/JWarmenhoven/DBDA-python>`__: Principled introduction to Bayesian data analysis.
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

To install PyMC on your system, follow the instructions on the appropriate installation guide:

-  `Installing PyMC on MacOS <https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(MacOS)>`__
-  `Installing PyMC on Linux <https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)>`__
-  `Installing PyMC on Windows <https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Windows)>`__


Citing PyMC
===========
Please choose from the following:

- |DOIpaper| *Probabilistic programming in Python using PyMC3*, Salvatier J., Wiecki T.V., Fonnesbeck C. (2016)
- |DOIzenodo| A DOI for all versions.
- DOIs for specific versions are shown on Zenodo and under `Releases <https://github.com/pymc-devs/pymc/releases>`_

.. |DOIpaper| image:: https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.55-blue
     :target: https://doi.org/10.7717/peerj-cs.55
.. |DOIzenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4603970.svg
   :target: https://doi.org/10.5281/zenodo.4603970

Contact
=======

We are using `discourse.pymc.io <https://discourse.pymc.io/>`__ as our main communication channel. You can also follow us on `Twitter @pymc_devs <https://twitter.com/pymc_devs>`__ for updates and other announcements.

To ask a question regarding modeling or usage of PyMC we encourage posting to our Discourse forum under the `“Questions” Category <https://discourse.pymc.io/c/questions>`__. You can also suggest feature in the `“Development” Category <https://discourse.pymc.io/c/development>`__.

To report an issue with PyMC please use the `issue tracker <https://github.com/pymc-devs/pymc/issues>`__.

Finally, if you need to get in touch for non-technical information about the project, `send us an e-mail <pymc.devs@gmail.com>`__.

License
=======

`Apache License, Version
2.0 <https://github.com/pymc-devs/pymc/blob/main/LICENSE>`__


Software using PyMC
===================

General purpose
---------------

- `Bambi <https://github.com/bambinos/bambi>`__: BAyesian Model-Building Interface (BAMBI) in Python.
- `SunODE <https://github.com/aseyboldt/sunode>`__: Fast ODE solver, much faster than the one that comes with PyMC.
- `pymc-learn <https://github.com/pymc-learn/pymc-learn>`__: Custom PyMC models built on top of pymc3_models/scikit-learn API
- `fenics-pymc3 <https://github.com/IvanYashchuk/fenics-pymc3>`__: Differentiable interface to FEniCS, a library for solving partial differential equations.

Domain specific
---------------

- `Exoplanet <https://github.com/dfm/exoplanet>`__: a toolkit for modeling of transit and/or radial velocity observations of exoplanets and other astronomical time series.
- `NiPyMC <https://github.com/PsychoinformaticsLab/nipymc>`__: Bayesian mixed-effects modeling of fMRI data in Python.
- `beat <https://github.com/hvasbath/beat>`__: Bayesian Earthquake Analysis Tool.
- `cell2location <https://github.com/BayraktarLab/cell2location>`__: Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics.

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

PyMC for enterprise
===================
`PyMC is now available as part of the Tidelift Subscription!`

Tidelift is working with PyMC and the maintainers of thousands of other open source
projects to deliver commercial support and maintenance for the open source dependencies
you use to build your applications. Save time, reduce risk, and improve code health,
while contributing financially to PyMC -- making it even more robust, reliable and,
let's face it, amazing!

|tidelift_learn| |tidelift_demo|

You can also get professional consulting support from `PyMC Labs <https://www.pymc-labs.io>`__.

Sponsors
========

|NumFOCUS|

|PyMCLabs|

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pymc-devs/pymc/main?filepath=%2Fdocs%2Fsource%2Fnotebooks
.. |Build Status| image:: https://github.com/pymc-devs/pymc/workflows/pytest/badge.svg
   :target: https://github.com/pymc-devs/pymc/actions
.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pymc/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/pymc-devs/pymc
.. |Dockerhub| image:: https://img.shields.io/docker/automated/pymc/pymc.svg
   :target: https://hub.docker.com/r/pymc/pymc
.. |NumFOCUS| image:: https://www.numfocus.org/wp-content/uploads/2017/03/1457562110.png
   :target: http://www.numfocus.org/
.. |NumFOCUS_badge| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: http://www.numfocus.org/
.. |PyMCLabs| image:: https://raw.githubusercontent.com/pymc-devs/pymc/main/docs/logos/sponsors/pymc-labs.png
   :target: https://pymc-labs.io
.. |tidelift| image:: https://img.shields.io/badge/-lifted!-2dd160.svg?colorA=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/pkg/pypi-pymc3?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
.. |tidelift_learn| image:: https://img.shields.io/badge/-learn%20more-2dd160.svg?color=orange&labelColor=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/pkg/pypi-pymc3?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
.. |tidelift_demo| image:: https://img.shields.io/badge/-request%20a%20demo-2dd160.svg?color=orange&labelColor=58595b&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAOCAYAAADJ7fe0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAAVhJREFUKBV1kj0vBFEUhmd2sdZHh2IlGhKFQuOviEYiNlFodCqtUqPxA%2FwCjUTnDygkGoVERFQaZFlE9nreO%2BdM5u5wkifvuee892Pu3CyEcA0DeIc%2B9IwftJsR6Cko3uCjguZdjuBZhhwmYDjGrOC96WED41UtsgEdGEAPlmAfpuAbFF%2BFZLfoMfRBGzThDtLgePPwBIpdddGzOArhPHUXowbNptE2www6a%2Fm96Y3pHN7oQ1s%2B13pxt1ENaKzBFWyWzaJ%2BRO0C9Jny6VPSoKjLVbMDC5bn5OPuJF%2BBSe95PVEMuugY5AegS9fCh7BedP45hRnj8TC34QQUe9bTZyh2KgvFk2vc8GIlXyTfsvqr6bPpNgv52ynnlomZJNpB70Xhl%2Bf6Sa02p1bApEfnETwxVa%2Faj%2BW%2FFtHltmxS%2FO3krvpTtTnVgu%2F6gvHRFvG78Ef3kOe5PimJXycY74blT5R%2BAAAAAElFTkSuQmCC
   :target: https://tidelift.com/subscription/request-a-demo?utm_source=pypi-pymc3&utm_medium=referral&utm_campaign=enterprise
