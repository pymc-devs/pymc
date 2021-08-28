# Implementing a Distribution (developer guide)

This guide provides an overview on how to implement a distribution for version 4 of PyMC3.
It is designed for developers who wish to add a new distribution to the library.
Users will not be aware of all this complexity and should instead make use of helper methods such as (TODO).

PyMC3 {class}`~pymc3.distributions.Distribution` build on top of Aesara's {class}`~aesara.tensor.random.op.RandomVariable`, and implement `logp` and `logcdf` methods as well as other initialization and validation helpers, most notably `shape/dims`, alternative parametrizations, and default `transforms`.

Here is a summary check-list of the steps needed to implement a new distribution.
Each section will be expanded below:

1. Creating a new `RandomVariable` `Op`
1. Implementing the corresponding `Distribution` class
1. Adding tests for the new `RandomVariable`
1. Adding tests for the `logp` / `logcdf` methods
1. Documenting the new `Distribution`.

This guide does not attempt to explain the rationale behind the `Distributions` current implementation, and details are provided only insofar as they help to implement new "standard" distributions.

## 1. Creating a new `RandomVariable` `Op`

{class}`~aesara.tensor.random.op.RandomVariable` are responsible for implementing the random sampling methods, which in version 3 of PyMC3 used to be one of the standard `Distribution` methods, alongside `logp` and `logcdf`.
The `RandomVariable` is also responsible for parameter broadcasting and shape inference.

Before creating a new `RandomVariable` make sure that it is not offered in the [Numpy library](https://numpy.org/doc/stable/reference/random/generator.html#distributions).
If it is, it should be added to the [Aesara library](https://github.com/aesara-devs/aesara) first and then imported into the PyMC3 library.

In addition, it might not always be necessary to implement a new `RandomVariable`.
For example if the new `Distribution` is just a special parametrization of an existing `Distribution`.
This is the case of the `OrderedLogistic` and `OrderedProbit`, which are just special parametrizations of the `Categorial` distribution.

The following snippet illustrates how to create a new `RandomVariable`:

```python

from aesara.tensor.var import TensorVariable
from aesara.tensor.random.op import RandomVariable
from typing import List, Tuple

# Create your own `RandomVariable`...
class BlahRV(RandomVariable):
    name: str = "blah"

    # Provide the minimum number of (output) dimensions for this RV
    # (e.g. `0` for a scalar, `1` for a vector, etc.)
    ndim_supp: int = 0

    # Provide the number of (input) dimensions for each parameter of the RV
    # (e.g. if there's only one vector parameter, `[1]`; for two parameters,
    # one a matrix and the other a scalar, `[2, 0]`; etc.)
    ndims_params: List[int] = [0, 0]

    # The NumPy/Aesara dtype for this RV (e.g. `"int32"`, `"int64"`).
    # The standard in the library is `"int64"` for discrete variables
    # and `"floatX"` for continuous variables
    dtype: str = "floatX"

    # A pretty text and LaTeX representation for the RV
    _print_name: Tuple[str, str] = ("blah", "\\operatorname{blah}")

    # If you want to add a custom signature and default values for the
    # parameters, do it like this. Otherwise this can be left out.
    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(loc, scale, size=size, **kwargs)

    # This is the Python code that produces samples.  Its signature will always
    # start with a NumPy `RandomState` object, then the distribution
    # parameters, and, finally, the size.
    #
    # This is effectively the v4 replacement for `Distribution.random`.
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        loc: np.ndarray,
        scale: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        return scipy.stats.blah.rvs(loc, scale, random_state=rng, size=size)

# Create the actual `RandomVariable` `Op`...
blah = BlahRV()

```

Some important things to keep in mind:

1. Everything inside the `rng_fn` method is pure Python code (as are the inputs) and should not make use of other `Aesara` symbolic ops. The random method should make use of the `rng` which is a Numpy    {class}`~numpy.random.RandomState`, so that samples are reproducible.
1. The `size` argument (together with the inputs shape) are the only way for the user to specify non-default `RandomVariable` dimensions. The `rng_fn` will have to take this into consideration for correct output. `size` is the specification used by `Numpy` and `Scipy` and works like PyMC3 `shape` for univariate distributions, but is different for multivariate distributions. Unfortunately there is no general reference documenting how `size` ought to work for multivariate distributions. This [discussion](https://github.com/numpy/numpy/issues/17669) may be helpful to get more context.
1. `Aesara` tries to infer the output shape of the `RandomVariable` (given a user-specified size) by introspection of the `ndim_supp` and `ndim_params` attributes. However, the default method may not work for more complex distributions. In that case, custom `_shape_from_params` (and less probably, `_infer_shape`) should also be implemented in the new `RandomVariable` class. One simple example is seen in the {class}`~pymc3.distributions.multivariate.DirichletMultinomialRV` where it was necessary to specify the `rep_param_idx` so that the `default_shape_from_params` helper method could do its job. In more complex cases, it may not be possible to make use of the default helper, but those have not been found yet!
1. It's okay to use the `rng_fn` `classmethods` of other Aesara and PyMC3 `RandomVariables` inside the new `rng_fn`. For example if you are implementing a negative HalfNormal `RandomVariable`, your `rng_fn` can simply return `- halfnormal.rng_fn(rng, scale, size)`.

*Note: In addition to `size`, the `PyMC3` API also provides `shape` and `dims` as alternatives to define a distribution dimensionality, but this is taken care of by {class}`~pymc3.distributions.Distribution`, and should not require any extra changes.*

For a quick test that your new `RandomVariable` `Op` is working, you can call the `Op` with the necessary parameters and then call `eval()` on the returned object:

```python

# blah = aesara.tensor.random.uniform in this example
blah([0, 0], [1, 2], size=(10, 2)).eval()

# array([[0.83674527, 0.76593773],
#    [0.00958496, 1.85742402],
#    [0.74001876, 0.6515534 ],
#    [0.95134629, 1.23564938],
#    [0.41460156, 0.33241175],
#    [0.66707807, 1.62134924],
#    [0.20748312, 0.45307477],
#    [0.65506507, 0.47713784],
#    [0.61284429, 0.49720329],
#    [0.69325978, 0.96272673]])

```

## 2. Inheriting from a PyMC3 base `Distribution` class

After implementing the new `RandomVariable` `Op`, it's time to make use of it in a new PyMC3 {class}`pymc3.distributions.Distribution`.
PyMC3 works in a very {term}`functional <Functional Programming>` way, and the `distribution` classes are there mostly to facilitate porting the `v3` code to the new `v4` version, add PyMC3 API features and keep related methods organized together.
In practice, they take care of:

1. Linking ({term}`Dispatching`) a rv_op class with the corresponding logp and logcdf methods.
1. Defining a standard transformation (for continuous distributions) that converts a bounded variable domain (e.g., positive line) to an unbounded domain (i.e., the real line), which many samplers prefer.
1. Validating the parametrization of a distribution and converting non-symbolic inputs (i.e., numeric literals or numpy arrays) to symbolic variables.
1. Converting multiple alternative parametrizations to the standard parametrization that the `RandomVariable` is defined in terms of.

Here is how the example continues:

```python

from pymc3.aesaraf import floatX, intX
from pymc3.distributions.continuous import PositiveContinuous
from pymc3.distributions.dist_math import bound

# Subclassing `PositiveContinuous` will dispatch a default `log` transformation
class Blah(PositiveContinuous):

    # This will be used by the metaclass `DistributionMeta` to dispatch the
    # class `logp` and `logcdf` methods to the `blah` `op`
    rv_op = blah

    # dist() is responsible for returning an instance of the rv_op. We pass
    # the standard parametrizations to super().dist
    @classmethod
    def dist(cls, param1, param2=None, alt_param2=None, **kwargs):
        param1 = at.as_tensor_variable(intX(param1))
        if param2 is not None and alt_param2 is not None:
            raise ValueError('Only one of param2 and alt_param2 is allowed')
        if alt_param2 is not None:
            param2 = 1 / alt_param2
        param2 = at.as_tensor_variable(floatX(param2))

        # The first value-only argument should be a list of the parameters that
        # the rv_op needs in order to be instantiated
        return super().dist([param1, param2], **kwargs)

    # Logp returns a symbolic expression for the logp evaluation of the variable
    # given the `value` of the variable and the parameters `param1` ... `paramN`
    def logp(value, param1, param2):
        logp_expression = value * (param1 + at.log(param2))

         # We use `bound` for parameter validation. After the default expression,
         # multiple comma-separated symbolic conditions can be added. Whenever
         # a bound is invalidated, the returned expression evaluates to `-np.inf`
         return bound(
            logp_expression,
            value >= 0,
            param2 >= 0,
            # There is one sneaky optional keyowrd argument, that converts an
            # otherwise elemwise `bound` to a reduced scalar bound. This is usually
            # needed for multivariate distributions where the dimensionality
            # of the bound conditions does not match that of the "value" / "logp"
            # By default it is set to `True`.
            broadcast_conditions=True,
        )

    # logcdf works the same way as logp. For bounded variables, it is expected
    # to return `-inf` for values below the domain start and `0` for values
    # above the domain end, but only when the parameters are valid.
    def logcdf(value, param1, param2):
        ...

```

Some notes:

1. A distribution should at the very least inherit from {class}`~pymc3.distributions.Discrete` or {class}`~pymc3.distributions.Continuous`. For the latter, more specific subclasses exist: `PositiveContinuous`, `UnitContinuous`, `BoundedContinuous`, `CircularContinuous`, which specify default transformations for the variables. If you need to specify a one-time custom transform you can also override the `__new__` method, as is done for the {class}`~pymc3.distributions.multivariate.Dirichlet`.
1. If a distribution does not have a corresponding `random` implementation, a `RandomVariable` should still be created that raises a `NotImplementedError`. This is the case for the {class}`~pymc3.distributions.continuous.Flat`. In this case it will be necessary to provide a standard `initval` by
   overriding `__new__`.
1. As mentioned above, `v4` works in a very {term}`functional <Functional Programming>` way, and all the information that is needed in the `logp` and `logcdf` methods is expected to be "carried" via the `RandomVariable` inputs. You may pass numerical arguments that are not strictly needed for the `rng_fn` method but are used in the `logp` and `logcdf` methods. Just keep in mind whether this affects the correct shape inference behavior of the `RandomVariable`. If specialized non-numeric information is needed you might need to define your custom`_logp` and `_logcdf` {term}`Dispatching` functions, but this should be done as a last resort.
1. The `logcdf` method is not a requirement, but it's a nice plus!

For a quick check that things are working you can try the following:

```python

import pymc3 as pm

# pm.blah = pm.Uniform in this example
blah = pm.Blah.dist([0, 0], [1, 2])

# Test that the returned blah_op is still working fine
blah.eval()
# array([0.62778803, 1.95165513])

# Test the logp
pm.logp(blah, [1.5, 1.5]).eval()
# array([       -inf, -0.69314718])

# Test the logcdf
pm.logcdf(blah, [1.5, 1.5]).eval()
# array([ 0.        , -0.28768207])
```

## 3. Adding tests for the new `RandomVariable`

Tests for new `RandomVariables` are mostly located in `pymc3/tests/test_distributions_random.py`.
Most tests can be accommodated by the default `BaseTestDistribution` class, which provides default tests for checking:
1. Expected inputs are passed to the `rv_op` by the `dist` `classmethod`, via `check_pymc_params_match_rv_op`
1. Expected (exact) draws are being returned, via `check_pymc_draws_match_reference`
1. Shape variable inference is correct, via `check_rv_size`

```python

class TestBlah(BaseTestDistribution):

    pymc_dist = pm.Blah
    # Parameters with which to test the blah pymc3 Distribution
    pymc_dist_params = {"param1": 0.25, "param2": 2.0}
    # Parameters that are expected to have passed as inputs to the RandomVariable op
    expected_rv_op_params = {"param1": 0.25, "param2": 2.0}
    # If the new `RandomVariable` is simply calling a `numpy`/`scipy` method,
    # we can make use of `seeded_[scipy|numpy]_distribution_builder` which
    # will prepare a seeded reference distribution for us.
    reference_dist_params = {"mu": 0.25, "loc": 2.0}
    reference_dist = seeded_scipy_distribution_builder("blah")
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]
```

Additional tests should be added for each optional parametrization of the distribution.
In this case it's enough to include the test `check_pymc_params_match_rv_op` since only this differs.

Make sure the tested alternative parameter value would lead to a different value for the associated default parameter.
For instance, if it's just the inverse, testing with `1.0` is not very informative, since the conversion would return `1.0` as well, and we can't be (as) sure that is working correctly.

```python

class TestBlahAltParam2(BaseTestDistribution):

    pymc_dist = pm.Blah
    # param2 is equivalent to 1 / alt_param2
    pymc_dist_params = {"param1": 0.25, "alt_param2": 4.0}
    expected_rv_op_params = {"param1": 0.25, "param2": 2.0}
    tests_to_run = ["check_pymc_params_match_rv_op"]

```

Custom tests can also be added to the class as is done for the {class}`~pymc3.tests.test_random.TestFlat`.

### Note on `check_rv_size` test:

Custom input sizes (and expected output shapes) can be defined for the `check_rv_size` test, by adding the optional class attributes `sizes_to_check` and `sizes_expected`:

```python
sizes_to_check = [None, (1), (2, 3)]
sizes_expected = [(3,), (1, 3), (2, 3, 3)]
tests_to_run = ["check_rv_size"]
```

This is usually needed for Multivariate distributions.
You can see an example in {class}`~pymc3.test.test_random.TestDirichlet`.

### Notes on `check_pymcs_draws_match_reference` test

The `check_pymcs_draws_match_reference` is a very simple test for the equality of draws from the `RandomVariable` and the exact same python function, given the same inputs and random seed.
A small number (`size=15`) is checked. This is not supposed to be a test for the correctness of the random generator.
The latter kind of test (if warranted) can be performed with the aid of `pymc3_random` and `pymc3_random_discrete` methods in the same test file, which will perform an expensive statistical comparison between the RandomVariable `rng_fn` and a reference Python function.
This kind of test only makes sense if there is a good independent generator reference (i.e., not just the same composition of numpy / scipy python calls that is done inside `rng_fn`).

Finally, when your `rng_fn` is doing something more than just calling a `numpy` or `scipy` method, you will need to setup an equivalent seeded function with which to compare for the exact draws (instead of relying on `seeded_[scipy|numpy]_distribution_builder`).
You can find an example in {class}`~pymc3.tests.test_distributions_random.TestWeibull`, whose `rng_fn` returns `beta * np.random.weibull(alpha, size=size)`.


## 4. Adding tests for the `logp` / `logcdf` methods

Tests for the `logp` and `logcdf` methods are contained in `pymc3/tests/test_distributions.py`, and most make use of the `TestMatchesScipy` class, which provides `check_logp`, `check_logcdf`, and
`check_selfconsistency_discrete_logcdf` standard methods.
These will suffice for most distributions.

```python

from pymc3.tests.helpers import select_by_precision

R = Domain([-np.inf, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, np.inf])
Rplus = Domain([0, 0.01, 0.1, 0.9, 0.99, 1, 1.5, 2, 100, np.inf])

...

def test_blah(self):

  self.check_logp(
      pymc3_dist=pm.Blah,
      # Domain of the distribution values
      domain=R,
      # Domains of the distribution parameters
      paramdomains={"mu": R, "sigma": Rplus},
      # Reference scipy (or other) logp function
      scipy_logp = lambda value, mu, sigma: sp.norm.logpdf(value, mu, sigma),
      # Number of decimal points expected to match between the pymc3 and reference functions
      decimal=select_by_precision(float64=6, float32=3),
      # Maximum number of combinations of domain * paramdomains to test
      n_samples=100,
  )

  self.check_logcdf(
      pymc3_dist=pm.Blah,
      domain=R,
      paramdomains={"mu": R, "sigma": Rplus},
      scipy_logcdf=lambda value, mu, sigma: sp.norm.logcdf(value, mu, sigma),
      decimal=select_by_precision(float64=6, float32=1),
      n_samples=-1,
  )

```

These methods will perform a grid evaluation on the combinations of domain and paramdomains values, and check that the pymc3 methods and the reference functions match.
There are a couple of details worth keeping in mind:

1. By default the first and last values (edges) of the `Domain` are not compared (they are used for other things). If it is important to test the edge of the `Domain`, the edge values can be repeated. This is done by the `Bool`: `Bool = Domain([0, 0, 1, 1], "int64")`
1. There are some default domains (such as `R` and `Rplus`) that you can use for testing your new distribution, but it's also perfectly fine to create your own domains inside the test function if there is a good reason for it (e.g., when the default values lead too many extreme unlikely combinations that are not very informative about the correctness of the implementation).
1. By default, a random subset of 100 `param` x `paramdomain` combinations is tested, in order to keep the test runtime under control. When testing your shiny new distribution, you can temporarily set `n_samples=-1` to force all combinations to be tested. This is important to avoid the your `PR` leading to surprising failures in future runs whenever some bad combinations of parameters are randomly tested.
1. On Github all tests are run twice, under the `aesara.config.floatX` flags of `"float64"` and `"float32"`. However, the reference Python functions will run in a pure "float64" environment, which means the reference and the `PyMC3` results can diverge quite a lot (e.g., underflowing to `-np.inf` for extreme parameters). You should therefore make sure you test locally in both regimes. A quick and dirty way of doing this is to temporariliy add `aesara.config.floatX = "float32"` at the very top of file, immediately after `import aesara`. Remember to set `n_samples=-1` as well to test all combinations. The test output will show what exact parameter values lead to a failure. If you are confident that your implementation is correct, you may opt to tweak the decimal precision with `select_by_precision`, or adjust the tested `Domain` values. In extreme cases, you can mark the test with a conditional `xfail` (if only one of the sub-methods is failing, they should be separated, so that the `xfail` is as narrow as possible):

```python

def test_blah_logp(self):
    ...


@pytest.mark.xfail(
   condition=(aesara.config.floatX == "float32"),
   reason="Fails on float32 due to numerical issues",
)
def test_blah_logcdf(self):
    ...


```


## 5. Documenting the new `Distribution`

New distributions should have a rich docstring, following the same format as that of previously implemented distributions.
It generally looks something like this:

```python
 r"""Univariate blah distribution.

 The pdf of this distribution is

 .. math::

    f(x \mid \param1, \param2) = \exp{x * (param1 + \log{param2})}

 .. plot::

     import matplotlib.pyplot as plt
     import numpy as np
     import scipy.stats as st
     import arviz as az
     x = np.linspace(-5, 5, 1000)
     params1 = [0., 0., 0., -2.]
     params2 = [0.4, 1., 2., 0.4]
     for param1, param2 in zip(params1, params2):
         pdf = st.blah.pdf(x, param1, param2)
         plt.plot(x, pdf, label=r'$\param1$ = {}, $\param2$ = {}'.format(param1, param2))
     plt.xlabel('x', fontsize=12)
     plt.ylabel('f(x)', fontsize=12)
     plt.legend(loc=1)
     plt.show()

 ========  ==========================================
 Support   :math:`x \in [0, \infty)`
 ========  ==========================================

 Blah distribution can be parameterized either in terms of param2 or
 alt_param2. The link between the two parametrizations is
 given by

 .. math::

    \param2 = \dfrac{1}{\alt_param2}


 Parameters
 ----------
 param1: float
     Interpretation of param1.
 param2: float
     Interpretation of param2 (param2 > 0).
 alt_param2: float
     Interpretation of alt_param2 (alt_param2 > 0) (alternative to param2).

 Examples
 --------
 .. code-block:: python

     with pm.Model():
         x = pm.Blah('x', param1=0, param2=10)
 """
```

The new distribution should be referenced in the respective API page in the `docs` module (e.g., `pymc3/docs/api/distributions.continuous.rst`).
If appropriate, a new notebook example should be added to [pymc-examples](https://github.com/pymc-devs/pymc-examples/) illustrating how this distribution can be used and how it relates (and/or differs) from other distributions that users are more likely to be familiar with.
