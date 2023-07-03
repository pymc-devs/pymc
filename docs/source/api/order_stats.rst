================
Order_Statistics
================

------------
Introduction
------------

In statistics, the kth order statistic of a statistical sample is equal to its kth-smallest value.
In this section, we'll tackle how users can find the Logarithmic probability corresponding to the nth order statistic (maximum value) using PyMC for their own Custom distributions.

In PyMC users can derive their own custom distributions. Custom distribution refers to the ability to define and use probability distributions that are not included in the standard set of distributions provided.
While PyMC provides a wide range of common probability distributions (e.g., Normal, Bernoulli, etc.), there may be cases where you need to use a distribution that is not available by default. In such cases, you can create your own custom distribution using the pm.DensityDist class provided by PyMC.
Simplest way to define a Custom Distribution can be better understood from the following example:

.. code-block:: python

    import numpy as np
    import pymc as pm
    from pytensor.tensor import TensorVariable

    def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
        return -(value - mu)**2

    with pm.Model():
        mu = pm.Normal('mu',0,1)
        pm.CustomDist(
            'custom_dist',
            mu,
            logp=logp,
            observed=np.random.randn(100),
        )
        idata = pm.sample(100)

Here, we create a CustomDist that wraps a black-box logp function. This variable cannot be used in prior or posterior predictive sampling because no random function was provided.

------------------------
`Max`
------------------------
Using PyMC and Pytensor, users can extract the maximum of a distribution and derive the log-probablity corresponding to this operation.

.. autofunction:: pymc.logprob.order.max_logprob
