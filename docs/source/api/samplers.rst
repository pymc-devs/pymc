Samplers
========

This submodule contains functions for MCMC and forward sampling.


.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   sample_prior_predictive
   sample_posterior_predictive
   draw


.. currentmodule:: pymc

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
.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   NUTS
   HamiltonianMC

Metropolis family
-----------------
.. currentmodule:: pymc

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
.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   CompoundStep
   Slice
