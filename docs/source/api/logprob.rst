***********
Probability
***********

.. currentmodule:: pymc

.. autosummary::
   :toctree: generated/

   logp
   logcdf
   icdf

Conditional probability
-----------------------

.. currentmodule:: pymc.logprob

.. autosummary::
   :toctree: generated/

   conditional_logp
   transformed_conditional_logp


Derived probability functions
-----------------------------

In PyMC, users can create their own custom distributions, which refers to the ability to define and utilize probability distributions not found in the standard set provided.
While PyMC provides a wide range of common probability distributions (e.g., Normal, Bernoulli, etc.), there may be cases where you need to use a distribution that is not available by default. In such cases, you can create your own custom distribution using the :class:`~pm.CustomDist` class provided by PyMC.
Use of pm.CustomDist can be illustrated with the help of the following example:

.. code-block:: python

    import pymc as pm
    from pytensor.tensor import TensorVariable
    def dist(
        lam: TensorVariable,
        shift: TensorVariable,
        size: TensorVariable,
    ) -> TensorVariable:
        return pm.Exponential.dist(lam, size=size) + shift
    with pm.Model() as m:
        lam = pm.HalfNormal("lam")
        shift = -1
        pm.CustomDist(
    	"custom_dist",
    	lam,
    	shift,
    	dist=dist,
    	observed=[-1, -1, 0],
        )
        prior = pm.sample_prior_predictive()
        posterior = pm.sample()

Here, we build a PyTensor graph from other PyMC distributions through :class:`~pymc.CustomDist`. PyMC can automatically infer that the logp of this graph that corresponds to a shifted Exponential distribution.

For more details, check out:

.. currentmodule:: pymc
.. autosummary::
   :toctree: generated/

   CustomDist

Let's dive into some applications:

.. toctree::
   :maxdepth: 2

   order_stats
