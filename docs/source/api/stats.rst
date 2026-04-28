Stats
*****

.. currentmodule:: pymc.stats

.. autosummary::
   :toctree: generated/

   compute_log_prior
   compute_log_likelihood

PyMC re-exports functions from the ``arviz_stats`` library under the ``pymc.stats``
namespace, allowing functions like ``summary``, ``ess``, ``rhat``, ``loo`` etc. to be
accessed as ``pymc.stats.<function>``. For the API documentation of those functions,
see the :doc:`arviz_stats documentation <arviz_stats:index>`.
