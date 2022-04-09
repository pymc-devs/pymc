(learning)=
# Learning

## Install PyMC
Refer to the {ref}`installation guide <installation>` to set up PyMC in your
own environment.

## Getting started

Start here to get acquainted with the core concepts of Bayesian analysis and PyMC. The following resources only assume a very basic knowledge of code and statistics.

### {octicon}`book;1em;sd-text-info` Introductory books

#### Bayesian Methods for Hackers

By Cameron Davidson-Pilon

The "hacker" in the title  means learn-as-you-code. This hands-on introduction teaches intuitive definitions of the Bayesian approach to statistics, worklflow and decision-making by applying them using PyMC.

[Github repo](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

[Project homepage](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)

#### Bayesian Analysis with Python

By Osvaldo Martin

A great introductory book written by a maintainer of PyMC. It provides a hands-on introduction to the main concepts of Bayesian statistics using synthetic and real data sets. Mastering the concepts in this book is a great foundation to pursue more advanced knowledge.

[Book website](https://www.packtpub.com/big-data-and-business-intelligence/bayesian-analysis-python-second-edition)

[Code and errata in PyMC](https://github.com/aloctavodia/BAP)

### {octicon}`mortar-board;1em;sd-text-info` PyMC core features

#### PyMC overview
The {ref}`pymc_overview` notebook in our documentation shows the PyMC 4.0 code in action

#### Predictive model checking

The {ref}`posterior_predictive` notebooks explains what prior and posterior predictive checks are and how to implement them in PyMC to validate your model.
It also introduces how to generate predictions on unseen data with PyMC.

#### General Linear Models: Linear regression

The {ref}`GLM_linear` notebook provides a gentle introduction to Bayesian linear regression and how it differs from the frequentist approach, and showcases how to implement it using PyMC.

#### Comparing models

The {ref}`model_comparison` notebook demonstrates the use of model comparison criteria in PyMC.

#### Size and dimensionality
The {ref}`dimensionality` notebook explains the different ways to define and annotate
the shape and dimensions of variables in PyMC models.

### {octicon}`people;1em;sd-text-info` Videos and podcasts
PyMC is also covered in many talks, videos and podcasts.
If you prefer these formats to written resources, go to {ref}`videos_and_podcasts`

### {octicon}`list-unordered;1em;sd-text-info` Glossary

PyMC's own {doc}`glossary` defines many core terms and provides useful references.

---

## Using PyMC

### {octicon}`code-square;1em;sd-text-info` Example notebooks
The {doc}`nb:index` hosts a collection of over 90 Jupyter notebooks using PyMC.
They range from notebooks addressed to people who are new to PyMC to notebooks
addressed to expert Bayesian practitioners that use PyMC.

The collection is organized using tags and categories to make it easy to explore
and focus on the examples that match your interests. You can also
check the {ref}`nb:object_index` to find examples using a specific
PyMC function or class.

### {octicon}`repo;1em;sd-text-info` Intermediate books
There are several great books available to deepen your understanding
of Bayesian statistics. Many of them also have their code
examples translated to PyMC and publicly available at
the [pymc-resources](https://github.com/pymc-devs/pymc-resources) repository.

In this section we highlight one book that uses PyMC and TensorFlow
Probability written by members of the PyMC team.
For the complete list see the {ref}`books` page.

#### Bayesian Modeling and Computation in Python

By Osvaldo Martin, Ravin Kumar and Junpeng Lao

Bayesian Modeling and Computation in Python aims to help beginner Bayesian practitioners to become intermediate modelers. It uses a hands on approach with PyMC and ArviZ focusing on the practice of applied statistics with references to the underlying mathematical theory.

[Book website](https://bayesiancomputationbook.com/welcome.html)
(the book website contains the whole book, code examples, and notebook in online format)

---
## Diving deeper

### {octicon}`plug;1em;sd-text-info` Experimental and cutting edge functionality
The {doc}`pmx:index` library extends PyMC with functionality
in active research until it becomes mature enough to be added
to PyMC.

### {octicon}`gear;1em;sd-text-info` PyMC internals guides
To be outlined and referenced here once [pymc#5538](https://github.com/pymc-devs/pymc/issues/5538)
is addressed.

:::{toctree}
:hidden:

installation
learn/core_notebooks/index
learn/books
learn/videos_and_podcasts
glossary
:::
