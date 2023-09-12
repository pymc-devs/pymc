***************
Transformations
***************

.. currentmodule:: pymc.distributions.transforms

Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
``transform`` parameter to a random variable constructor.

.. autosummary::
   :toctree: generated

    circular
    log
    log_exp_m1
    logodds
    simplex
    sum_to_1
    ordered


Specific Transform Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    CholeskyCovPacked
    Interval
    LogExpM1
    Ordered
    SumTo1
    ZeroSumTransform


Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    Chain
