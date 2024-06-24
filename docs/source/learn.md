(learn)=
# Learn PyMC & Bayesian modeling

:::{toctree}
:maxdepth: 1
installation
learn/core_notebooks/index
learn/books
learn/videos_and_podcasts
learn/consulting
glossary
:::

## At a glance
### Beginner
  - Book: [Bayesian Analysis with Python](http://bap.com.ar/)
  - Book: [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)


### Intermediate
  - {ref}`pymc_overview` shows PyMC 4.0 code in action
  - Example notebooks: {doc}`nb:gallery`
    - {ref}`GLM_linear`
    - {ref}`posterior_predictive`
    - Comparing models: {ref}`model_comparison`
    - Shapes and dimensionality {ref}`dimensionality`
  - {ref}`videos_and_podcasts`
  - Book: [Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/welcome.html)

### Advanced
  - {octicon}`plug;1em;sd-text-info` Experimental and cutting edge functionality: {doc}`pmx:index` library
  - {octicon}`gear;1em;sd-text-info` PyMC internals guides (To be outlined and referenced here once [pymc#5538](https://github.com/pymc-devs/pymc/issues/5538)
is addressed)


```{jupyter-execute}
import inspect
import sys
import pymc as pm
print(sys.executable)
print(pm.__version__)
print(inspect.signature(pm.sample_prior_predictive))
print(inspect.getfile(pm))
```
