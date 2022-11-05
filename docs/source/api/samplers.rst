Samplers
========

This submodule contains functions for MCMC and forward sampling.


.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   sample
   sample_prior_predictive
   sample_posterior_predictive
   sample_posterior_predictive_w
   sampling.jax.sample_blackjax_nuts
   sampling.jax.sample_numpyro_nuts
   iter_sample
   init_nuts
   draw

Step methods
************

.. currentmodule:: pymc

HMC family
----------

.. autosummary::
   :toctree: generated/

   NUTS
   HamiltonianMC

Metropolis family
-----------------

.. autosummary::
    :toctree: generated/

    BinaryGibbsMetropolis
    BinaryMetropolis
    CategoricalGibbsMetropolis
    CauchyProposal
    DEMetropolis
    DEMetropolisZ
    LaplaceProposal
    Metropolis
    MultivariateNormalProposal
    NormalProposal
    PoissonProposal
    UniformProposal

Other step methods
------------------

.. autosummary::
   :toctree: generated/

   CompoundStep
   Slice
