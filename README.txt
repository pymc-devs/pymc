==========================================
PyMC -- Markov chain Monte Carlo in Python
==========================================

:Date: April 6, 2007
:Authors: Chris Fonnesbeck, Anand Patil, David Huard
:Contact: chris@trichech.us
:Web site: http://code.google.com/p/pymc/
:Copyright: This document has been placed in the public domain.
:License: PyMC is released under the MIT license. 
:Version: 2.0

Purpose
=======


PyMC is a python module that implements the Metropolis-Hastings algorithm as a 
python class. It is extremely flexible and applicable to a large suite of 
problems. PyMC includes methods for summarizing output, plotting, goodness-of-
fit and convergence diagnostics. 


Features
========

* Fits Bayesian statistical models you create with Markov chain Monte Carlo and 
  other algorithms.
  
* Sampling loops can be paused and tuned manually, or saved and restarted later.

* 24 well-documented statistical distributions.

* Creates summaries including tables and plots.

* Convergence diagnostics. 

* Traces can be saved to the disk as plain text, pickles, sqlite or MySQL
  database, or hdf5 archives. The hdf5 option allows traces to be streamed
  to the disk during sampling, reducing memory usage.

* Extensible: easily incorporates custom step methods and unusual probability 
  distributions and stochastic processes.

* MCMC loops can be embedded in larger programs, and results can be analyzed 
  with the full power of Python.

What's new in 2.0
=================

* New flexible object model and syntax (non backward compatible).

* Reduced redundant computations: only relevant log-probability terms are 
  computed, and these are cached.

* Optimized probability distributions.

* New adaptive blocked Metropolis step method.

* Much more!

Usage
=====

First, define your model in mymodel.py::

   import pymc
   import numpy

   n = 5*numpy.ones(4,dtype=int)
   x = numpy.array([-.86,-.3,-.05,.73])

   alpha = pymc.Normal('alpha',mu=0,tau=.01)
   beta = pymc.Normal('beta',mu=0,tau=.01)   

   @deterministic
   def theta(a=alpha, b=beta, d=dose):
       """theta = logit^{-1}(a+b)"""
       return pymc.invlogit(a+b*d)


   d = Binomial('d', n=n, p=theta, value=numpy.array([0.,1.,3.,5.]), isdata=True)

From a python shell, type::

	import pymc
	import mymodel
	
	S = pymc.MCMC(mymodel, db='pickle')
	S.sample(iter=10000, burn=5000, thin=2)

where problem_definition is a module or a dictionary containing Node, Data and 
Parameter instances defining your problem. 


History
=======

PyMC began development in 2003, as an effort to generalize the process of building Metropolis-Hastings samplers, with an aim to making Markov chain Monte Carlo more accessible to non-statisticians (particularly ecologists). The choice to develop PyMC as a python module, rather than a standalone application, allowed the use MCMC methods in a larger modeling framework, in contrast to the BUGS environment. By 2005, PyMC was reliable enough for version 1.0 to be released to the public. A small group of regular users, most associated with the University of Georgia, provided much of the feedback necessary for the refinement of PyMC to its current state.

In 2006, David Huard and Anand Patil joined Chris Fonnesbeck on the development team for PyMC 2.0. This iteration of the software strives for more flexibility, better performance and a better end-user experience than any previous version of PyMC.


Installation
============

See the `INSTALL.txt`_ file. 


.. _`INSTALL.txt`:
   ./INSTALL.txt

