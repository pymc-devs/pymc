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

* Implements the Metropolis-Hastings algorithm so you can focus on your 
  application instead of on gory numerical algorithms.

* Define your distribution from 24 well-documented statistical distributions,

* Summarize your results in tables and plots.

* Run convergence diagnostics. 


What's new in 2.0
=================

* Faster internal logic by coding the bottlenecks with Pyrex,

* Faster distributions by an optimization of the Fortran functions,

* Added a Joint Metropolis and a Gibbs sampler,

* Define your problem in a separate file using the Node, Data and Parameter 
  classes. Use decorators to improve code readability.

* Save your samples directly to a database. Select one from sqlite, MySQL, HDF5,
  pickle files, text files or write a custom database backend from a template.

* Run interactive convergence diagnostics,

* Stop a sampling run in the middle, save it's state and restart the sampler 
  later,

* Seed multiple chains on different processors. 


Usage
=====

From a python shell, type::

	import PyMC
	S = PyMC.Sampler(problem_definition, db='pickle')
	S.sample(iter=10000, burn=5000, thin=2)

where problem_definition is a module or a dictionary containing Node, Data and 
Parameter instances defining your problem. Read the `user guide`_ for a 
complete description of the package, classes and some examples to get started.


History
=======

Chris ...


Installation
============

See the `INSTALL.txt`_ file. 


.. _`INSTALL.txt`:
   ./INSTALL.txt

.. _`user guide`:
   docs/pdf/new_interface.pdf
