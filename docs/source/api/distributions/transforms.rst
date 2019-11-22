*******************************************
Transformations of a random variable from one space to another.
*******************************************

.. currentmodule:: pymc3.distributions.transforms
.. autosummary::

transform
stick_breaking
logodds
interval
log_exp_m1
lowerbound
upperbound
ordered
log
sum_to_1
t_stick_breaking
circular
CholeskyCovPacked
Chain

Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
`transform` parameter to a random variable constructor.  These are
initialized instances of the Transform Classes, which are described
below.

.. autodata::  stick_breaking
.. autodata::  logodds
.. autodata::  interval
.. autodata::  log_exp_m1
.. autodata::  lowerbound
.. autodata::  upperbound
.. autodata::  ordered
.. autodata::  log
.. autodata::  sum_to_1
.. autodata::  t_stick_breaking 
.. autodata::  circular

Transform Base Classes
~~~~~~~~~~~~~~~~~~~~~~

Typically the programmer will not use these directly.

.. autoclass::  Transform
    :members:
.. autoclass::  transform
    :members:
.. autoclass::  TransformedDistribution
    :members:


Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Chain
    :members:
.. autoclass:: CholeskyCovPacked


Specific Transform Classes
~~~~~~~~~~~~~~~~~

.. autoclass::  Log
    :members:
.. autoclass::  LogExpM1
    :members:
.. autoclass::  LogOdds
    :members:
.. autoclass::  Interval
    :members:
.. autoclass::  LowerBound
    :members:
.. autoclass::  UpperBound
    :members:
.. autoclass::  Ordered
    :members:
.. autoclass::  SumTo1
    :members:
.. autoclass::  StickBreaking
    :members:
.. autoclass::  Circular
    :members:







