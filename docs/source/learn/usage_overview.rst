TODO: incorporate the useful bits of this page into the learning section

**************
Usage Overview
**************

For a detailed overview of building models in PyMC, please read the appropriate sections in the rest of the documentation. For a flavor of what PyMC models look like, here is a quick example.

First, let's import PyMC and :doc:`ArviZ <arviz:index>` (which handles plotting and diagnostics):

::

    import arviz as az
    import numpy as np
    import pymc as pm

Models are defined using a context manager (``with`` statement). The model is specified declaratively inside the context manager, instantiating model variables and transforming them as necessary. Here is an example of a model for a bioassay experiment:

::

    # Set style
    az.style.use("arviz-darkgrid")

    # Data
    n = np.ones(4)*5
    y = np.array([0, 1, 3, 5])
    dose = np.array([-.86,-.3,-.05,.73])

    with pm.Model() as bioassay_model:

        # Prior distributions for latent variables
        alpha = pm.Normal('alpha', 0, sigma=10)
        beta = pm.Normal('beta', 0, sigma=1)

        # Linear combination of parameters
        theta = pm.invlogit(alpha + beta * dose)

        # Model likelihood
        deaths = pm.Binomial('deaths', n=n, p=theta, observed=y)

Save this file, then from a python shell (or another file in the same directory), call:

::

    with bioassay_model:

        # Draw samples
        idata = pm.sample(1000, tune=2000, cores=2)
        # Plot two parameters
        az.plot_forest(idata, var_names=['alpha', 'beta'], r_hat=True)

This example will generate 1000 posterior samples on each of two cores using the NUTS algorithm, preceded by 2000 tuning samples (these are good default numbers for most models).

::

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [beta, alpha]
    |██████████████████████████████████████| 100.00% [6000/6000 00:04<00:00 Sampling 2 chains, 0 divergences]

The sample is returned as arrays inside a ``MultiTrace`` object, which is then passed to the plotting function. The resulting graph shows a forest plot of the random variables in the model, along with a convergence diagnostic (R-hat) that indicates our model has converged.

.. image:: ./images/forestplot.png
   :width: 1000px

See also
========

* `Tutorials <nb_tutorials/index.html>`__
* `Examples <nb_examples/index.html>`__


.. |NumFOCUS| image:: https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png
   :target: http://www.numfocus.org/
   :height: 120px
.. |PyMCLabs| image:: https://raw.githubusercontent.com/pymc-devs/pymc/main/docs/pymc-labs-logo.png
   :target: https://pymc-labs.io
   :height: 120px
