*********************
Variational Inference
*********************

.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   ADVI
   ASVGD
   SVGD
   FullRankADVI
   fit

.. currentmodule:: pymc.variational

.. autosummary::
   :toctree: generated/

   ImplicitGradient
   Inference
   KLqp

Approximations
--------------

.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   Empirical
   FullRank
   MeanField
   sample_approx

OPVI
----

.. autosummary::
   :toctree: generated/

   Group

.. currentmodule:: pymc.variational

.. autosummary::
   :toctree: generated/

   Approximation

Operators
---------

.. automodule:: pymc.variational.operators
.. autosummary::
   :toctree: generated/

   KL
   KSD

Special
-------
.. currentmodule:: pymc.variational
.. autosummary::
   :toctree: generated/

   Stein

.. currentmodule:: pymc
.. autosummary::
   :toctree: generated/

   adadelta
   adagrad
   adagrad_window
   adam
   adamax
   apply_momentum
   apply_nesterov_momentum
   momentum
   nesterov_momentum
   norm_constraint
   rmsprop
   sgd
   total_norm_constraint
