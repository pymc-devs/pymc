---
sd_hide_title: true
---

# PyMC Documentation

:::{card}
:img-top: https://raw.githubusercontent.com/pymc-devs/pymc/main/docs/logos/PyMC.jpg
:margin: 0 2 auto auto
:width: 50%
:text-align: center
:shadow: none

+++
Probabilistic Programming in Python
:::

::::{div} sd-d-flex-row sd-align-major-center
:::{button-ref} learning
:color: primary

Get started!
:::
::::


:::{card} Announcements: library name change and launching PyMC 4.0!
:width: 75%
:margin: auto

We have two major announcements that we're excited to share. First of all, a new name for our library: the PyMC3 library you know and love is now called PyMC. PyMC3 still exists, as a specific major release between PyMC2 and PyMC 4.0. Read more about the renaming and how to solve related issues you might experience from this update [here]().

This ties into our second announcement, which is that we are hereby launching the newest version of PyMC: PyMC 4.0! Read more about this new release [here]().
:::

---

# Main Features & Benefits

::::::{grid} 1 1 2 2
:gutter: 1

:::::{grid-item}

::::{grid} 1
:gutter: 1

:::{grid-item}

**Friendly modelling API**

PyMC allows you to write down models using an intuitive syntax to describe a data generating process.
:::
:::{grid-item}

**Cutting edge algorithms and model building blocks**

Fit your model using gradient-based MCMC algorithms like NUTS, using ADVI for fast approximate inference — including minibatch-ADVI for scaling to large datasets — or using Gaussian processes to build Bayesian nonparametric models.
:::
::::
:::::

:::::{grid-item}

```{code-block} python
---
---
    import pymc3 as pm

    X, y = linear_training_data()
    with pm.Model() as linear_model:
        weights = pm.Normal("weights", mu=0, sigma=1)
        noise = pm.Gamma("noise", alpha=2, beta=1)
        y_observed = pm.Normal(
            "y_observed",
            mu=X @ weights,
            sigma=noise,
            observed=y,
        )

        prior = pm.sample_prior_predictive()
        posterior = pm.sample()
        posterior_pred = pm.sample_posterior_predictive(posterior)
```
:::::

::::::

# Support

PyMC is a non-profit project under NumFOCUS umbrella. If you value PyMC and want to support its development, consider donating to the project.

::::{div} sd-d-flex-row sd-align-major-center
:::{button-link} https://numfocus.org/donate-to-pymc3
:color: secondary


Donate
:::
::::

## Our sponsors

::::{grid} 2 4 4 4

:::{grid-item}
:::

:::{grid-item-card}
:img-background: https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png
:link: https://numfocus.org/
:shadow: none
:::

:::{grid-item-card}
:img-background: https://github.com/pymc-devs/pymc/raw/main/docs/pymc-labs-logo.png
:link: https://www.pymc-labs.io/
:shadow: none
:::

:::{grid-item}
:::

::::

# Testimonials

::::{card-carousel} 2

:::{card}
:class-body: small
**[SpaceX](https://www.spacex.com/)**
^^^
> At SpaceX PyMC helped us estimate supplier delivery uncertainty and quantify which suppliers were consistent in sending us rocket parts and which suppliers we needed to partner with to understand how we could reduce variability

_Ravin Kumar_
:::

:::{card}
:class-body: small
**[Salesforce](http://www.salesforce.com)**
^^^
> PyMC is my primary tool for statistical modeling at Salesforce. I use it to combine disparate sources of information and pretty much anywhere that quantifying uncertainty is important. We've also been experimenting with gaussian processes to model time series data for forecasting.

_Eddie Landesberg. Manager, Data Scientist_
:::

:::{card}
:class-body: small
**[Novartis Institutes for Biomedical Research](https://www.novartis.com/our-science/novartis-institutes-biomedical-research)**
^^^
> At the Novartis Institutes for Biomedical Research, we use PyMC for a wide variety of scientific and business use cases. The API is incredible, making it easy to express probabilistic models of our scientific experimental data and business processes, such as models of electrophysiology and molecular dose response.

_Eric J. Ma_
:::

:::{card}
:class-body: small
**[Intercom](https://www.intercom.com)**
^^^
> At Intercom, we've adopted PyMC as part of our A/B testing framework. The API made it easy to integrate into our existing experimentation framework and the methodology has made communication of experiment results much more straightforward for non technical stakeholders.

_Louis Ryan_
:::

:::{card}
:class-body: small
**[Channel 4](http://www.channel4.co.uk)**
^^^
> Used in research code at Channel 4 for developing internal forecasting tools.

_Peader Coyle_
:::

::::

Find more testimonials [here](https://github.com/pymc-devs/pymc/wiki/Testimonials).

# Citing PyMC

Use this to cite the library:

Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 [DOI: 10.7717/peerj-cs.55.](https://doi.org/10.7717/peerj-cs.55)

See [Google Scholar](https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011) for a continuously updated list of papers citing PyMC3.

:::{toctree}
:maxdepth: 1
:hidden:

installation
learning
api
developers
community
contributing/index
about
:::
