Samplers
========

This submodule contains functions for MCMC and forward sampling.


.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   sample
   sample_prior_predictive
   sample_posterior_predictive
   draw
   compute_deterministics
   init_nuts
   sampling.jax.sample_blackjax_nuts
   sampling.jax.sample_numpyro_nuts


Step methods
************

HMC family
----------
.. currentmodule:: pymc.step_methods.hmc

.. autosummary::
   :toctree: generated/

   NUTS
   HamiltonianMC

Metropolis family
-----------------
.. currentmodule:: pymc.step_methods

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
.. currentmodule:: pymc.step_methods

.. autosummary::
   :toctree: generated/

   CompoundStep
   Slice
