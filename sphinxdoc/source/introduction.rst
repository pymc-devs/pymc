************
Introduction
************

Purpose
=======

PyMC is a python module that implements Bayesian statistical models and
fitting algorithms, including Markov chain Monte Carlo.
Its flexibility makes it applicable to a large suite of problems as well as
easily extensible. Along with core sampling functionality, PyMC includes
methods for summarizing output, plotting, goodness-of-fit and convergence
diagnostics.


Features
========

* Fits Bayesian statistical models you create with Markov chain Monte Carlo and
  other algorithms.

* Large suite of well-documented statistical distributions.

* Gaussian processes.

* Sampling loops can be paused and tuned manually, or saved and restarted later.

* Creates summaries including tables and plots.

* Traces can be saved to the disk as plain text, Python pickles, SQLite or MySQL
  database, or hdf5 archives.

* Convergence diagnostics.

* Extensible: easily incorporates custom step methods and unusual probability
  distributions.

* MCMC loops can be embedded in larger programs, and results can be analyzed
  with the full power of Python.


What's new in 2.1
=================

To do


Usage
=====

First, define your model in a file, say mymodel.py (with comments, of course!)::

   # Import relevant modules
   import pymc
   import numpy as np

   # Some data
   n = 5*np.ones(4,dtype=int)
   x = np.array([-.86,-.3,-.05,.73])

   # Priors on unknown parameters
   alpha = pymc.Normal('alpha',mu=0,tau=.01)
   beta = pymc.Normal('beta',mu=0,tau=.01)

   # Arbitrary deterministic function of parameters
   @pymc.deterministic
   def theta(a=alpha, b=beta):
       """theta = logit^{-1}(a+b)"""
       return pymc.invlogit(a+b*x)

   # Binomial likelihood for data
   d = pymc.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),
                     \observed=True)

Save this file, then from a python shell (or another filein the same directory), call::

	import pymc
	import mymodel

	S = pymc.MCMC(mymodel, db='pickle')
	S.sample(iter=10000, burn=5000, thin=2)
	pymc.Matplot.plot(S)

This will generate 10000 posterior samples, thinned by a factor of 2, with the first half discarded as burn-in. The sample is stored in a Python serialization (pickle) database.


History
=======

PyMC began development in 2003, as an effort to generalize the process of building Metropolis-Hastings samplers, with an aim to making Markov chain Monte Carlo (MCMC) more accessible to non-statisticians (particularly ecologists). The choice to develop PyMC as a python module, rather than a standalone application, allowed the use MCMC methods in a larger modeling framework. By 2005, PyMC was reliable enough for version 1.0 to be released to the public. A small group of regular users, most associated with the University of Georgia, provided much of the feedback necessary for the refinement of PyMC to a usable state.

In 2006, David Huard and Anand Patil joined Chris Fonnesbeck on the development team for PyMC 2.0. This iteration of the software strives for more flexibility, better performance and a better end-user experience than any previous version of PyMC.



Getting started
===============

This user guide provides all the information needed to install PyMC, code
a Bayesian statistical model, run the sampler, save and analyze the results.
In addition, the appendix contains a chapter on MCMC theory as well as the list of the
available statistical distributions. More `examples`_ of usage as well as
`tutorials`_  are available from the PyMC web site.

.. _`examples`: http://code.google.com/p/pymc/

.. _`tutorials`: http://code.google.com/p/pymc/wiki/TutorialsAndRecipes



