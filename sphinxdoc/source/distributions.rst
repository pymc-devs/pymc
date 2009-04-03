.. _chap:distributions:

*************************
Probability distributions
*************************


PyMC provides 35 built-in probability distributions. For each distribution, it provides:

* A function that evaluates its log-probability or log-density: normal_like().
* A function that draws random variables: rnormal().
* A function that computes the expectation associated with the distribution: normal_expval().
* A Stochastic subclass generated from the distribution: Normal.

This section describes the likelihood functions of these distributions.

.. module:: pymc.distributions

Discrete distributions
======================

.. autofunction:: bernoulli_like
   
.. autofunction:: binomial_like

.. autofunction:: categorical_like

.. autofunction:: discrete_uniform_like

.. autofunction:: geometric_like

.. autofunction:: hypergeometric_like

.. autofunction:: negative_binomial_like

.. autofunction:: poisson_like


   
   
Continuous distributions
========================

.. autofunction:: beta_like


