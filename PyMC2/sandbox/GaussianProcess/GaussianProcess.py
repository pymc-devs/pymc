"""
Imports all the GP components and implements the GaussianProcess class.

Returns a GPRealization-valued parameter subclass called GP. GP's attributes are:

- cov_fun: a function
- mean_fun: a function
- base_mesh: A mesh
- obs_mesh: A mesh
- lintrans: A mesh
- obs_values: A mesh

- C: a GPCovariance-valued Node depending on cov_fun, base_mesh, obs_mesh, lintrans.
- sig: a matrix-valued Node giving the matrix square-root of C
- tau: a matrix-valued Node giving the inverse matrix of C

- mu: a GPMean-valued Node depending on cov_fun, mean_fun, base_mesh, obs_mesh, lintrans, obs_values.

- value: a GPRealization instance. If the value is set from a matrix or an ndarray, it's viewed as a GPRealization.
         That means GPRealization's constructor needs to take a matrix.
         
- logp: The log-probability of value evaluated on the base mesh only. If the base mesh has changed so that the
        current value is not appropriate, raises an error. Any other change is fine, though.

GP's methods are:

- random(): Replaces value with a new GPRealization instance and returns it.
- touch(): Same as in Parameter, but may require some extra thinking.
"""

from GPCovariance import GPCovariance
from GPMean import GPMean
from GPRealization import GPRealization
from PyMC2 import Parameter, Node

class GaussianProcess(Parameter):
    pass