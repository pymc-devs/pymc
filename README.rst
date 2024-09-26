.. image:: https://cdn.rawgit.com/pymc-devs/pymc/main/docs/logos/svg/PyMC_banner.svg
    :height: 100px
    :alt: PyMC logo
    :align: center

|Build Status| |Coverage| |NumFOCUS_badge| |Binder| |Dockerhub| |DOIzenodo| |Conda Downloads|

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


Linear Regression Example
==========================


Plant growth can be influenced by multiple factors, and understanding these relationships is crucial for optimizing agricultural practices.

Imagine we conduct an experiment to predict the growth of a plant based on different environmental variables.

.. code-block:: python

   import pymc as pm

   # Taking draws from a normal distribution
   seed = 42
   x_dist = pm.Normal.dist(shape=(100, 3))
   x_data = pm.draw(x_dist, random_seed=seed)

   # Independent Variables:
   # Sunlight Hours: Number of hours the plant is exposed to sunlight daily.
   # Water Amount: Daily water amount given to the plant (in milliliters).
   # Soil Nitrogen Content: Percentage of nitrogen content in the soil.


   # Dependent Variable:
   # Plant Growth (y): Measured as the increase in plant height (in centimeters) over a certain period.


   # Define coordinate values for all dimensions of the data
   coords={
    "trial": range(100),
    "features": ["sunlight hours", "water amount", "soil nitrogen"],
   }

   # Define generative model
   with pm.Model(coords=coords) as generative_model:
      x = pm.Data("x", x_data, dims=["trial", "features"])

      # Model parameters
      betas = pm.Normal("betas", dims="features")
      sigma = pm.HalfNormal("sigma")

      # Linear model
      mu = x @ betas

      # Likelihood
      # Assuming we measure deviation of each plant from baseline
      plant_growth = pm.Normal("plant growth", mu, sigma, dims="trial")


   # Generating data from model by fixing parameters
   fixed_parameters = {
    "betas": [5, 20, 2],
    "sigma": 0.5,
   }
   with pm.do(generative_model, fixed_parameters) as synthetic_model:
      idata = pm.sample_prior_predictive(random_seed=seed) # Sample from prior predictive distribution.
      synthetic_y = idata.prior["plant growth"].sel(draw=0, chain=0)


   # Infer parameters conditioned on observed data
   with pm.observe(generative_model, {"plant growth": synthetic_y}) as inference_model:
      idata = pm.sample(random_seed=seed)

      summary = pm.stats.summary(idata, var_names=["betas", "sigma"])
      print(summary)


From the summary, we can see that the mean of the inferred parameters are very close to the fixed parameters

=====================  ======  =====  ========  =========  ===========  =========  ==========  ==========  =======
Params                  mean     sd    hdi_3%    hdi_97%    mcse_mean    mcse_sd    ess_bulk    ess_tail    r_hat
=====================  ======  =====  ========  =========  ===========  =========  ==========  ==========  =======
betas[sunlight hours]   4.972  0.054     4.866      5.066        0.001      0.001        3003        1257        1
betas[water amount]    19.963  0.051    19.872     20.062        0.001      0.001        3112        1658        1
betas[soil nitrogen]    1.994  0.055     1.899      2.107        0.001      0.001        3221        1559        1
sigma                   0.511  0.037     0.438      0.575        0.001      0            2945        1522        1
=====================  ======  =====  ========  =========  ===========  =========  ==========  ==========  =======

.. code-block:: python

   # Simulate new data conditioned on inferred parameters
   new_x_data = pm.draw(
      pm.Normal.dist(shape=(3, 3)),
      random_seed=seed,
   )
   new_coords = coords | {"trial": [0, 1, 2]}

   with inference_model:
      pm.set_data({"x": new_x_data}, coords=new_coords)
      pm.sample_posterior_predictive(
         idata,
         predictions=True,
         extend_inferencedata=True,
         random_seed=seed,
      )

   pm.stats.summary(idata.predictions, kind="stats")

The new data conditioned on inferred parameters would look like:

================ ======== ======= ======== =========
Output            mean     sd      hdi_3%   hdi_97%
================ ======== ======= ======== =========
plant growth[0]   14.229   0.515   13.325   15.272
plant growth[1]   24.418   0.511   23.428   25.326
plant growth[2]   -6.747   0.511   -7.740   -5.797
================ ======== ======= ======== =========

.. code-block:: python

   # Simulate new data, under a scenario where the first beta is zero
   with pm.do(
    inference_model,
    {inference_model["betas"]: inference_model["betas"] * [0, 1, 1]},
   ) as plant_growth_model:
      new_predictions = pm.sample_posterior_predictive(
         idata,
         predictions=True,
         random_seed=seed,
      )

   pm.stats.summary(new_predictions, kind="stats")

The new data, under the above scenario would look like:

================ ======== ======= ======== =========
Output            mean     sd      hdi_3%   hdi_97%
================ ======== ======= ======== =========
plant growth[0]   12.149   0.515   11.193   13.135
plant growth[1]   29.809   0.508   28.832   30.717
plant growth[2]   -0.131   0.507   -1.121    0.791
================ ======== ======= ======== =========

Getting started
===============

If you already know about Bayesian statistics:
----------------------------------------------

-  `API quickstart guide <https://www.pymc.io/projects/examples/en/latest/introductory/api_quickstart.html>`__
-  The `PyMC tutorial <https://docs.pymc.io/en/latest/learn/core_notebooks/pymc_overview.html>`__
-  `PyMC examples <https://www.pymc.io/projects/examples/en/latest/gallery.html>`__ and the `API reference <https://docs.pymc.io/en/stable/api.html>`__

Learn Bayesian statistics with a book together with PyMC
--------------------------------------------------------

-  `Bayesian Analysis with Python  <http://bap.com.ar/>`__ (third edition) by Osvaldo Martin: Great introductory book.
-  `Probabilistic Programming and Bayesian Methods for Hackers <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`__: Fantastic book with many applied code examples.
-  `PyMC port of the book "Doing Bayesian Data Analysis" by John Kruschke <https://github.com/cluhmann/DBDA-python>`__ as well as the `first edition <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__.
-  `PyMC port of the book "Statistical Rethinking A Bayesian Course with Examples in R and Stan" by Richard McElreath <https://github.com/pymc-devs/resources/tree/master/Rethinking>`__
-  `PyMC port of the book "Bayesian Cognitive Modeling" by Michael Lee and EJ Wagenmakers <https://github.com/pymc-devs/resources/tree/master/BCM>`__: Focused on using Bayesian statistics in cognitive modeling.

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

.. |DOIpaper| image:: https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.1516-blue.svg
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
- `X @pymc_devs <https://x.com/pymc_devs>`__
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

See Google Scholar `here <https://scholar.google.com/scholar?cites=6357998555684300962>`__ and `here <https://scholar.google.com/scholar?cites=6936955228135731011>`__ for a continuously updated list.

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

Thanks to our contributors
==========================

|contributors|

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
.. |contributors| image:: https://contrib.rocks/image?repo=pymc-devs/pymc
   :target: https://github.com/pymc-devs/pymc/graphs/contributors
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/pymc/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/pymc
