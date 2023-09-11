.. _api:

***
API
***

.. toctree::
   :maxdepth: 1

   api/distributions
   api/gp
   api/model/core
   api/model/fgraph
   api/model/transform/conditioning
   api/samplers
   api/vi
   api/smc
   api/data
   api/ode
   api/logprob
   api/tuning
   api/math
   api/pytensorf
   api/shape_utils
   api/backends
   api/misc

------------------
Dimensionality
------------------
PyMC provides numerous methods, and syntactic sugar, to easily specify the dimensionality of
Random Variables in modeling. Refer to :ref:`dimensionality` notebook to see examples
demonstrating the functionality.

--------------
API extensions
--------------

Plots, stats and diagnostics
----------------------------
Plots, stats and diagnostics are delegated to the
:doc:`ArviZ <arviz:index>`.
library, a general purpose library for
"exploratory analysis of Bayesian models".

* Functions from the `arviz.plots` module are available through ``pymc.<function>`` or ``pymc.plots.<function>``,
  but for their API documentation please refer to the :ref:`ArviZ documentation <arviz:plot_api>`.

* Functions from the `arviz.stats` module are available through ``pymc.<function>`` or ``pymc.stats.<function>``,
  but for their API documentation please refer to the :ref:`ArviZ documentation <arviz:stats_api>`.

ArviZ is a dependency of PyMC and so, in addition to the locations described above,
importing ArviZ and using ``arviz.<function>`` will also work without any extra installation.

Generalized Linear Models (GLMs)
--------------------------------

Generalized Linear Models are delegated to the
`Bambi <https://bambinos.github.io/bambi>`_.
library, a high-level Bayesian model-building
interface built on top of PyMC.

Bambi is not a dependency of PyMC and should be installed in addition to PyMC
to use it to generate PyMC models via formula syntax.
