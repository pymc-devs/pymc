***************
Transformations
***************

.. currentmodule:: pymc.transforms

Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
``transform`` parameter to a random variable constructor.

.. autosummary::
   :toctree: generated

    simplex
    logodds
    interval
    log_exp_m1
    ordered
    log
    sum_to_1
    circular

Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    Chain
    CholeskyCovPacked

Specific Transform Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    LogExpM1
    Ordered
    SumTo1
