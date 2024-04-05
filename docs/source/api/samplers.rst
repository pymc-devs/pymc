Samplers
========

This submodule contains functions for MCMC and forward sampling.


.. currentmodule:: pymc.sampling.forward

.. autosummary::
   :toctree: generated/

   sample_prior_predictive
   sample_posterior_predictive
   draw

.. currentmodule:: pymc.sampling.deterministic
   compute_deterministics


.. currentmodule:: pymc.sampling.mcmc

.. autosummary::
   :toctree: generated/

   sample
   init_nuts

.. currentmodule:: pymc.sampling.jax

.. autosummary::
   :toctree: generated/

   sample_blackjax_nuts
   sample_numpyro_nuts


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
