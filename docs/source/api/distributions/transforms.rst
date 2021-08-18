****************************************************************
Transformations of a random variable from one space to another.
****************************************************************

Note that for convenience these entities can be addressed as
``pm.transforms.``\ *X* for any name *X*, although they are actually
implemented as ``pm.distributions.transforms.``\*X*.

.. currentmodule:: pymc3.distributions.transforms


.. contents ::

..
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
   circular
   CholeskyCovPacked
   Chain



Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
``transform`` parameter to a random variable constructor.  These are
initialized instances of the Transform Classes, which are described
below.

.. glossary::

``stick_breaking``
    Instantiation of :class:`~pymc3.distributions.transforms.StickBreaking`
    :class:`~pymc3.distributions.transforms.Transform` class for use in the ``transform``
    argument of a random variable.

``logodds``
    Instantiation of
    :class:`~pymc3.distributions.transforms.LogOdds` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``interval``
    Alias of
    :class:`~pymc3.distributions.transforms.Interval` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``log_exp_m1``
    Instantiation of
    :class:`~pymc3.distributions.transforms.LogExpM1` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``lowerbound``
    Alias of
    :class:`~pymc3.distributions.transforms.LowerBound` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``upperbound``
    Alias of
    :class:`~pymc3.distributions.transforms.UpperBound` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``ordered``
    Instantiation of
    :class:`~pymc3.distributions.transforms.Ordered` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

``log``
    Instantiation of
    :class:`~pymc3.distributions.transforms.Log` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.


``sum_to_1``
    Instantiation of
    :class:`~pymc3.distributions.transforms.SumTo1` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.


``circular``
    Instantiation of
    :class:`~pymc3.distributions.transforms.Circular` :class:`~pymc3.distributions.transforms.Transform` class
    for use in the ``transform`` argument of a random variable.

Transform Base Classes
~~~~~~~~~~~~~~~~~~~~~~

Typically the programmer will not use these directly.

.. autoclass::  Transform
    :members:


Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Chain
    :members:
.. autoclass:: CholeskyCovPacked
    :members:


Specific Transform Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass::  Log
    :members:
.. autoclass::  LogExpM1
    :members:
.. autoclass::  LogOdds
    :members:
.. autoclass::  Interval
    :members:
.. autoclass::  Ordered
    :members:
.. autoclass::  SumTo1
    :members:
.. autoclass::  StickBreaking
    :members:
.. autoclass::  Circular
    :members:
