.. _intro:

************
Introduction
************


Purpose
=======

PyMC3 is a probabilistic programming package for Python that allows users to fit Bayesian models using a variety of numerical methods, most notably Markov chain Monte Carlo (MCMC) and variational inference (VI). Its flexibility and extensibility make it applicable to a large suite of problems. Along with core model specification and fitting functionality, PyMC3 includes functionality for summarizing output and for model diagnostics.



Features
========

PyMC3 strives to make Bayesian modeling as simple and painless as possible,  allowing users to focus on their scientific problem, rather than on the methods used to solve it. Here is a partial list of its features:

* Modern methods for fitting Bayesian models, including MCMC and VI.

* Includes a large suite of well-documented statistical distributions.

* Uses Theano as the computational backend, allowing for fast expression evaluation, automatic gradient calculation, and GPU computing.

* Built-in support for Gaussian process modeling.

* Model summarization and plotting.

* Model checking and convergence detection.

* Extensible: easily incorporates custom step methods and unusual probability
  distributions.

* Bayesian models can be embedded in larger programs, and results can be analyzed
  with the full power of Python.


What's new in version 3
=======================

The third major version of PyMC has benefitted from being re-written from scratch. Substantial improvements in the user interface and performance have resulted from this. While PyMC2 relied on Fortran extensions (via f2py) for most of the computational heavy-lifting, PyMC3 leverages Theano, a library from the Montréal Institute for Learning Algorithms (MILA), for array-based expression evaluation, to perform its computation. What this provides, above all else, is fast automatic differentiation, which is at the heart of the gradient-based sampling and optimization methods currently providing inference for probabilistic programming.

Major changes from previous versions:

* New flexible object model and syntax (not backward-compatible with PyMC2).

* Gradient-based MCMC methods, including Hamiltonian Monte Carlo (HMC), the No U-turn Sampler (NUTS), and Stein Variational Gradient Descent.

* Variational inference methods, including automatic differentiation variational inference (ADVI) and operator variational inference (OPVI).

* An interface for easy formula-based specification of generalized linear models (GLM).

* Elliptical slice sampling.

* Specialized distributions for representing time series.

* A library of Jupyter notebooks that provide case studies and fully developed usage examples.

* Much more!

While the addition of Theano adds a level of complexity to the development of PyMC, fundamentally altering how the underlying computation is performed, we have worked hard to maintain the elegant simplicity of the original PyMC model specification syntax.


History
=======

PyMC began development in 2003, as an effort to generalize the process of
building Metropolis-Hastings samplers, with an aim to making Markov chain Monte
Carlo (MCMC) more accessible to applied scientists.
The choice to develop PyMC as a python module, rather than a standalone
application, allowed the use of MCMC methods in a larger modeling framework. By
2005, PyMC was reliable enough for version 1.0 to be released to the public. A
small group of regular users, most associated with the University of Georgia,
provided much of the feedback necessary for the refinement of PyMC to a usable
state.

In 2006, David Huard and Anand Patil joined Chris Fonnesbeck on the development
team for PyMC 2.0. This iteration of the software strives for more flexibility,
better performance and a better end-user experience than any previous version
of PyMC. PyMC 2.2 was released in April 2012. It contained numerous bugfixes and
optimizations, as well as a few new features, including improved output
plotting, csv table output, improved imputation syntax, and posterior
predictive check plots. PyMC 2.3 was released on October 31, 2013. It included
Python 3 compatibility, improved summary plots, and some important bug fixes.

In 2011, John Salvatier began thinking about implementing gradient-based MCMC samplers, and developed the ``mcex`` package to experiment with his ideas. The following year, John was invited by the team to re-engineer PyMC to accomodate Hamiltonian Monte Carlo sampling. This led to the adoption of Theano as the computational back end, and marked the beginning of PyMC3's development. The first alpha version of PyMC3 was released in June 2015. Over the following 2 years, the core development team grew to 12 members, and the first release, PyMC3 3.0, was launched in January 2017.


Usage Overview
==============

For a detailed overview of building models in PyMC3, please read the appropriate sections in the rest of the documentation. For a flavor of what PyMC3 models look like, here is a quick example.

First, let's import PyMC3 and `ArviZ <https://arviz-devs.github.io/arviz/>`__ (which handles plotting and diagnostics):

    import arviz as az
    import numpy as np
    import pymc3 as pm

Models are defined using a context manager (``with`` statement). The model is specified declaratively inside the context manager, instantiating model variables and transforming them as necessary. Here is an example of a model for a bioassay experiment:

::

    # Set style
    az.style.use("arviz-darkgrid")

    # Data
    n = np.ones(4)*5
    y = np.array([0, 1, 3, 5])
    dose = np.array([-.86,-.3,-.05,.73])

    with pm.Model() as bioassay_model:

        # Prior distributions for latent variables
        alpha = pm.Normal('alpha', 0, sigma=10)
        beta = pm.Normal('beta', 0, sigma=1)

        # Linear combination of parameters
        theta = pm.invlogit(alpha + beta * dose)

        # Model likelihood
        deaths = pm.Binomial('deaths', n=n, p=theta, observed=y)

Save this file, then from a python shell (or another file in the same directory), call:

::

    with bioassay_model:

        # Draw samples
        trace = pm.sample(1000, tune=2000, cores=2)
        # Plot two parameters
        az.plot_forest(trace, var_names=['alpha', 'beta'], r_hat=True)

This example will generate 1000 posterior samples on each of two cores using the NUTS algorithm, preceded by 2000 tuning samples (these are good default numbers for most models).

::

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [beta, alpha]
    |██████████████████████████████████████| 100.00% [6000/6000 00:04<00:00 Sampling 2 chains, 0 divergences]

The sample is returned as arrays inside a ``MultiTrace`` object, which is then passed to the plotting function. The resulting graph shows a forest plot of the random variables in the model, along with a convergence diagnostic (R-hat) that indicates our model has converged.

.. image:: ./images/forestplot.png

See also
========

* `Tutorials <nb_tutorials/index.html>`__
* `Examples <nb_examples/index.html>`__
