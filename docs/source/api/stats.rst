Stats
*****

.. currentmodule:: pymc.stats

.. autosummary::
   :toctree: generated/

   compute_log_prior
   compute_log_likelihood

PyMC also re-exports the ``arviz_stats`` namespace under ``pymc.stats`` so that
functions like ``summary``, ``ess``, ``rhat``, ``loo`` etc. can be accessed as
``pymc.stats.<function>``. For the API documentation of those functions, see
the :ref:`ArviZ documentation <arviz:stats_api>`.
