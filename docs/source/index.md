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
:ref-type: ref
:class: sd-fs-2 sd-px-5

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
    import numpy as np
    import pymc as pm

    X = np.random.normal(size=100)
    y = np.random.normal(X) * 1.2

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
:class: sd-fs-2 sd-px-5


Donate
:::
::::

## Our sponsors

::::{grid} 2 4 4 4

:::{grid-item}
:::

:::{grid-item-card}
:img-background: _static/sponsors/numfocus.png
:link: https://numfocus.org/
:shadow: none
:::

:::{grid-item-card}
:img-background: _static/sponsors/pymc-labs.png
:link: https://www.pymc-labs.io/
:shadow: none
:::

:::{grid-item}
:::

::::

# Testimonials

:::::{card-carousel} 2

::::{include} about/featured_testimonials.md
::::

:::::

Find more testimonials {ref}`here <testimonials>`

# Citing PyMC

Use this to cite the library:

Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 [DOI: 10.7717/peerj-cs.55.](https://doi.org/10.7717/peerj-cs.55)

The BibTeX entry is:

```bibtex
@article{pymc,
  title={Probabilistic programming in Python using PyMC3},
  author={Salvatier, John and Wiecki, Thomas V and Fonnesbeck, Christopher},
  journal={PeerJ Computer Science},
  volume={2},
  pages={e55},
  year={2016},
  publisher={PeerJ Inc.}
}
```

See [Google Scholar](https://scholar.google.de/scholar?oi=bibs&hl=en&authuser=1&cites=6936955228135731011) for a continuously updated list of papers citing PyMC.

:::{toctree}
:maxdepth: 1
:hidden:

learning
api
community
contributing/index
:::
