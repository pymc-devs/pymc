***********
shape_utils
***********

This submodule contains various functions that apply numpy's broadcasting rules to shape tuples, and also to samples drawn from probability distributions.

The main challenge when broadcasting samples drawn from a generative model, is that each random variate has a core shape. When we draw many i.i.d samples from a given RV, for example if we ask for `size_tuple` i.i.d draws, the result usually is a `size_tuple + RV_core_shape`. In the generative model's hierarchy, the downstream RVs that are conditionally dependent on our above sampled values, will get an array with a shape that is incosistent with the core shape they expect to see for their parameters. This is a problem sometimes because it prevents regular broadcasting in complex hierachical models, and thus make prior and posterior predictive sampling difficult.

This module introduces functions that are made aware of the requested `size_tuple` of i.i.d samples, and does the broadcasting on the core shapes, transparently ignoring or moving the i.i.d `size_tuple` prepended axes around.

.. currentmodule:: pymc.distributions.shape_utils

.. autosummary::
   :toctree: generated/

   to_tuple
   shapes_broadcasting
   broadcast_dist_samples_shape
   get_broadcastable_dist_samples
   broadcast_distribution_samples
   broadcast_dist_samples_to
   rv_size_is_none
   change_dist_size
