.. _chap_distributions:

*************************
Probability distributions
*************************

PyMC provides a large suite of built-in probability distributions. For each distribution, it provides:

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

.. autofunction:: truncated_poisson_like


Continuous distributions
========================

.. autofunction:: beta_like

.. autofunction:: cauchy_like

.. autofunction:: chi2_like

.. autofunction:: degenerate_like

.. autofunction:: exponential_like

.. autofunction:: exponweib_like

.. autofunction:: gamma_like

.. autofunction:: half_cauchy_like

.. autofunction:: half_normal_like

.. autofunction:: inverse_gamma_like

.. autofunction:: laplace_like

.. autofunction:: logistic_like

.. autofunction:: lognormal_like

.. autofunction:: noncentral_t_like

.. autofunction:: normal_like

.. autofunction:: pareto_like

.. autofunction:: skew_normal_like

.. autofunction:: t_like

.. autofunction:: truncated_normal_like

.. autofunction:: truncated_pareto_like

.. autofunction:: uniform_like

.. autofunction:: von_mises_like

.. autofunction:: weibull_like


Multivariate discrete distributions
===================================

.. autofunction:: multivariate_hypergeometric_like

.. autofunction:: multinomial_like


Multivariate continuous distributions
=====================================

.. autofunction:: dirichlet_like

.. autofunction:: mv_normal_like

.. autofunction:: mv_normal_chol_like

.. autofunction:: mv_normal_cov_like

.. autofunction:: wishart_like

.. autofunction:: wishart_cov_like
