(implementing-a-distribution)=
# Implementing a RandomVariable Distribution

This guide provides an overview on how to implement a distribution for PyMC.
It is designed for developers who wish to add a new distribution to the library.
Users will not be aware of all this complexity and should instead make use of helper methods such as `~pymc.CustomDist`.

PyMC {class}`~pymc.Distribution` builds on top of PyTensor's {class}`~pytensor.tensor.random.op.RandomVariable`, and implements `logp`, `logcdf`, `icdf` and `support_point` methods as well as other initialization and validation helpers.
Most notably `shape/dims/observed` kwargs, alternative parametrizations, and default `transform`.

Here is a summary check-list of the steps needed to implement a new distribution.
Each section will be expanded below:

1. Creating a new `RandomVariable` `Op`
1. Implementing the corresponding `Distribution` class
1. Adding tests for the new `RandomVariable`
1. Adding tests for `logp` / `logcdf` / `icdf` and `support_point` methods
1. Documenting the new `Distribution`.

This guide does not attempt to explain the rationale behind the `Distributions` current implementation, and details are provided only insofar as they help to implement new "standard" distributions.

## 1. Creating a new `RandomVariable` `Op`

{class}`~pytensor.tensor.random.op.RandomVariable` are responsible for implementing the random sampling methods.
The `RandomVariable` is also responsible for parameter broadcasting and shape inference.

Before creating a new `RandomVariable` make sure that it is not already offered in the {mod}`NumPy library <numpy.random>`.
If it is, it should be added to the {doc}`PyTensor library <pytensor:index>` first and then imported into the PyMC library.

In addition, it might not always be necessary to implement a new `RandomVariable`.
For example if the new `Distribution` is just a special parametrization of an existing `Distribution`.
This is the case of the `OrderedLogistic` and `OrderedProbit`, which are just special parametrizations of the `Categorical` distribution.

The following snippet illustrates how to create a new `RandomVariable`:

```python

from pytensor.tensor.var import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from typing import List, Tuple

# Create your own `RandomVariable`...
class BlahRV(RandomVariable):
    name: str = "blah"

    # Provide a numpy-style signature for this RV, which indicates
    # the number and core dimensionality of each input and output.
    signature: "(),()->()"

    # The NumPy/PyTensor dtype for this RV (e.g. `"int32"`, `"int64"`).
    # The standard in the library is `"int64"` for discrete variables
    # and `"floatX"` for continuous variables
    dtype: str = "floatX"

    # A pretty text and LaTeX representation for the RV
    _print_name: Tuple[str, str] = ("blah", "\\operatorname{blah}")

    # If you want to add a custom signature and default values for the
    # parameters, do it like this. Otherwise this can be left out.
    def __call__(self, loc=0.0, scale=1.0, **kwargs) -> TensorVariable:
        return super().__call__(loc, scale, **kwargs)

    # This is the Python code that produces samples.  Its signature will always
    # start with a NumPy `RandomState` object, then the distribution
    # parameters, and, finally, the size.

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

1. Everything inside the `rng_fn` method is pure Python code (as are the inputs) and should __not__ make use of other `PyTensor` symbolic ops. The random method should make use of the `rng` which is a NumPy {class}`~numpy.random.RandomGenerator`, so that samples are reproducible.
1. Non-default `RandomVariable` dimensions will end up in the `rng_fn` via the `size` kwarg. The `rng_fn` will have to take this into consideration for correct output. `size` is the specification used by NumPy and SciPy and works like PyMC `shape` for univariate distributions, but is different for multivariate distributions. For multivariate distributions the __`size` excludes the support dimensions__, whereas the __`shape` of the resulting `TensorVariable` or `ndarray` includes the support dimensions__. For more context check {ref}`The dimensionality notebook <dimensionality>`.
1. `PyTensor` can automatically infer the output shape of univariate `RandomVariable`s. For multivariate distributions, the method `_supp_shape_from_params` must be implemented in the new `RandomVariable` class. This method returns the support dimensionality of an RV given its parameters. In some cases this can be derived from the shape of one of its parameters, in which case the helper {func}`pytensor.tensor.random.utils.supp_shape_from_ref_param_shape` cand be used as is in {class}`~pymc.DirichletMultinomialRV`. In other cases the argument values (and not their shapes) may determine the support shape of the distribution, as happens in the `~pymc.distributions.multivarite._LKJCholeskyCovRV`. In simpler cases they may be constant.
1. It's okay to use the `rng_fn` `classmethods` of other PyTensor and PyMC `RandomVariables` inside the new `rng_fn`. For example if you are implementing a negative HalfNormal `RandomVariable`, your `rng_fn` can simply return `- halfnormal.rng_fn(rng, scale, size)`.

*Note: In addition to `size`, the PyMC API also provides `shape`, `dims` and `observed` as alternatives to define a distribution dimensionality, but this is taken care of by {class}`~pymc.Distribution`, and should not require any extra changes.*

For a quick test that your new `RandomVariable` `Op` is working, you can call the `Op` with the necessary parameters and then call {class}`~pymc.draw` on the returned object:

```python

# blah = pytensor.tensor.random.uniform in this example
# multiple calls with the same seed should return the same values
pm.draw(blah([0, 0], [1, 2], size=(10, 2)), random_seed=1)

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

## 2. Inheriting from a PyMC base `Distribution` class

After implementing the new `RandomVariable` `Op`, it's time to make use of it in a new PyMC {class}`~pymc.Distribution`.
PyMC works in a very {term}`functional <Functional Programming>` way, and the `distribution` classes are there mostly to add PyMC API features and keep related methods organized together.
In practice, they take care of:

1. Linking ({term}`Dispatching`) an `rv_op` class with the corresponding `support_point`, `logp`, `logcdf` and `icdf` methods.
1. Defining a standard transformation (for continuous distributions) that converts a bounded variable domain (e.g., positive line) to an unbounded domain (i.e., the real line), which many samplers prefer.
1. Validating the parametrization of a distribution and converting non-symbolic inputs (i.e., numeric literals or NumPy arrays) to symbolic variables.
1. Converting multiple alternative parametrizations to the standard parametrization that the `RandomVariable` is defined in terms of.

Here is how the example continues:

```python

import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none


# Subclassing `PositiveContinuous` will dispatch a default `log` transformation
class Blah(PositiveContinuous):
    # This will be used by the metaclass `DistributionMeta` to dispatch the
    # class `logp` and `logcdf` methods to the `blah` `Op` defined in the last line of the code above.
    rv_op = blah

    # dist() is responsible for returning an instance of the rv_op.
    # We pass the standard parametrizations to super().dist
    @classmethod
    def dist(cls, param1, param2=None, alt_param2=None, **kwargs):
        param1 = pt.as_tensor_variable(param1)
        if param2 is not None and alt_param2 is not None:
            raise ValueError("Only one of param2 and alt_param2 is allowed.")
        if alt_param2 is not None:
            param2 = 1 / alt_param2
        param2 = pt.as_tensor_variable(param2)

        # The first value-only argument should be a list of the parameters that
        # the rv_op needs in order to be instantiated
        return super().dist([param1, param2], **kwargs)

    # support_point returns a symbolic expression for the stable point from which to start sampling
    # the variable, given the implicit `rv`, `size` and `param1` ... `paramN`.
    # This is typically a "representative" point such as the the mean or mode.
    def support_point(rv, size, param1, param2):
        support_point, _ = pt.broadcast_arrays(param1, param2)
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    # Logp returns a symbolic expression for the elementwise log-pdf or log-pmf evaluation
    # of the variable given the `value` of the variable and the parameters `param1` ... `paramN`.
    def logp(value, param1, param2):
        logp_expression = value * (param1 + pt.log(param2))

        # A switch is often used to enforce the distribution support domain
        bounded_logp_expression = pt.switch(
            pt.gt(value >= 0),
            logp_expression,
            -np.inf,
        )

        # We use `check_parameters` for parameter validation. After the default expression,
        # multiple comma-separated symbolic conditions can be added.
        # Whenever a bound is invalidated, the returned expression raises an error
        # with the message defined in the optional `msg` keyword argument.
        return check_parameters(
            bounded_logp_expression,
            param2 >= 0,
            msg="param2 >= 0",
        )

    # logcdf works the same way as logp. For bounded variables, it is expected to return
    # `-inf` for values below the domain start and `0` for values above the domain end.
    def logcdf(value, param1, param2):
        ...

    def icdf(value, param1, param2):
        ...

```

Some notes:

1. A distribution should at the very least inherit from {class}`~pymc.Discrete` or {class}`~pymc.Continuous`. For the latter, more specific subclasses exist: `PositiveContinuous`, `UnitContinuous`, `BoundedContinuous`, `CircularContinuous`, `SimplexContinuous`, which specify default transformations for the variables. If you need to specify a one-time custom transform you can also create a `_default_transform` dispatch function as is done for the {class}`~pymc.distributions.multivariate.LKJCholeskyCov`.
1. If a distribution does not have a corresponding `rng_fn` implementation, a `RandomVariable` should still be created to raise a `NotImplementedError`. This is, for example, the case in {class}`~pymc.distributions.continuous.Flat`. In this case it will be necessary to provide a `support_point` method, because without a `rng_fn`, PyMC can't fall back to a random draw to use as an initial point for MCMC.
1. As mentioned above, PyMC works in a very {term}`functional <Functional Programming>` way, and all the information that is needed in the `logp`, `logcdf`, `icdf` and `support_point` methods is expected to be "carried" via the `RandomVariable` inputs. You may pass numerical arguments that are not strictly needed for the `rng_fn` method but are used in the those methods. Just keep in mind whether this affects the correct shape inference behavior of the `RandomVariable`.
1. The `logcdf`, and `icdf` methods is not a requirement, but it's a nice plus!
1. Currently, only one moment is supported in the `support_point` method, and probably the "higher-order" one is the most useful (that is `mean` > `median` > `mode`)... You might need to truncate the moment if you are dealing with a discrete distribution. `support_point` should return a valid point for the random variable (i.e., it always has non-zero probability when evaluated at that point)
1. When creating the `support_point` method, be careful with `size != None` and broadcast properly also based on parameters that are not necessarily used to calculate the moment. For example, the `sigma` in `pm.Normal.dist(mu=0, sigma=np.arange(1, 6))` is irrelevant for the moment, but may nevertheless inform about the shape. In this case, the `support_point` should return `[mu, mu, mu, mu, mu]`.

For a quick check that things are working you can try the following:

```python

import pymc as pm
from pymc.distributions.distribution import support_point

# pm.blah = pm.Normal in this example
blah = pm.blah.dist(mu=0, sigma=1)

# Test that the returned blah_op is still working fine
pm.draw(blah, random_seed=1)
# array(-1.01397228)

# Test the support_point method
support_point(blah).eval()
# array(0.)

# Test the logp method
pm.logp(blah, [-0.5, 1.5]).eval()
# array([-1.04393853, -2.04393853])

# Test the logcdf method
pm.logcdf(blah, [-0.5, 1.5]).eval()
# array([-1.17591177, -0.06914345])
```

## 3. Adding tests for the new `RandomVariable`

Tests for new `RandomVariables` are mostly located in `tests/distributions/test_*.py`.
Most tests can be accommodated by the default `BaseTestDistributionRandom` class, which provides default tests for checking:
1. Expected inputs are passed to the `rv_op` by the `dist` `classmethod`, via `check_pymc_params_match_rv_op`
1. Expected (exact) draws are being returned, via `check_pymc_draws_match_reference`
1. Shape variable inference is correct, via `check_rv_size`

```python

from pymc.testing import BaseTestDistributionRandom, seeded_scipy_distribution_builder


class TestBlah(BaseTestDistributionRandom):
    pymc_dist = pm.Blah
    # Parameters with which to test the blah pymc Distribution
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

class TestBlahAltParam2(BaseTestDistributionRandom):

    pymc_dist = pm.Blah
    # param2 is equivalent to 1 / alt_param2
    pymc_dist_params = {"param1": 0.25, "alt_param2": 4.0}
    expected_rv_op_params = {"param1": 0.25, "param2": 2.0}
    tests_to_run = ["check_pymc_params_match_rv_op"]

```

Custom tests can also be added to the class as is done for the {class}`~tests.distributions.test_continuous.TestFlat`.

### Note on `check_rv_size` test:

Custom input sizes (and expected output shapes) can be defined for the `check_rv_size` test, by adding the optional class attributes `sizes_to_check` and `sizes_expected`:

```python
sizes_to_check = [None, (1), (2, 3)]
sizes_expected = [(3,), (1, 3), (2, 3, 3)]
tests_to_run = ["check_rv_size"]
```

This is usually needed for Multivariate distributions.
You can see an example in {class}`~tests.distributions.test_multivariate.TestDirichlet`.

### Notes on `check_pymcs_draws_match_reference` test

The `check_pymcs_draws_match_reference` is a very simple test for the equality of draws from the `RandomVariable` and the exact same python function, given the same inputs and random seed.
A small number (`size=15`) is checked. This is not supposed to be a test for the correctness of the random number generator.
The latter kind of test (if warranted) can be performed with the aid of `pymc_random` and `pymc_random_discrete` methods, which will perform an expensive statistical comparison between the `RandomVariable.rng_fn` and a reference Python function.
This kind of test only makes sense if there is a good independent generator reference (i.e., not just the same composition of NumPy / SciPy calls that is done inside `rng_fn`).

Finally, when your `rng_fn` is doing something more than just calling a NumPy or SciPy method, you will need to set up an equivalent seeded function with which to compare for the exact draws (instead of relying on `seeded_[scipy|numpy]_distribution_builder`).
You can find an example in {class}`~tests.distributions.test_continuous.TestWeibull`, whose `rng_fn` returns `beta * np.random.weibull(alpha, size=size)`.


## 4. Adding tests for the `logp` / `logcdf` / `icdf` methods

Tests for the `logp`, `logcdf` and `icdf` mostly make use of the helpers `check_logp`, `check_logcdf`, `check_icdf` and
`check_selfconsistency_discrete_logcdf` implemented in `~testing`

```python

from pymc.testing import Domain, check_logp, check_logcdf, select_by_precision

R = Domain([-np.inf, -2.1, -1, -0.01, 0.0, 0.01, 1, 2.1, np.inf])
Rplus = Domain([0, 0.01, 0.1, 0.9, 0.99, 1, 1.5, 2, 100, np.inf])


def test_blah():
    check_logp(
        pymc_dist=pm.Blah,
        # Domain of the distribution values
        domain=R,
        # Domains of the distribution parameters
        paramdomains={"mu": R, "sigma": Rplus},
        # Reference scipy (or other) logp function
        scipy_logp=lambda value, mu, sigma: sp.norm.logpdf(value, mu, sigma),
        # Number of decimal points expected to match between the pymc and reference functions
        decimal=select_by_precision(float64=6, float32=3),
        # Maximum number of combinations of domain * paramdomains to test
        n_samples=100,
    )

    check_logcdf(
        pymc_dist=pm.Blah,
        domain=R,
        paramdomains={"mu": R, "sigma": Rplus},
        scipy_logcdf=lambda value, mu, sigma: sp.norm.logcdf(value, mu, sigma),
        decimal=select_by_precision(float64=6, float32=1),
        n_samples=-1,
    )

```

These methods will perform a grid evaluation on the combinations of `domain` and `paramdomains` values, and check that the PyMC methods and the reference functions match.
There are a couple of details worth keeping in mind:

1. By default, the first and last values (edges) of the `Domain` are not compared (they are used for other things). If it is important to test the edge of the `Domain`, the edge values can be repeated. This is done by the `Bool`: `Bool = Domain([0, 0, 1, 1], "int64")`
3. There are some default domains (such as `R` and `Rplus`) that you can use for testing your new distribution, but it's also perfectly fine to create your own domains inside the test function if there is a good reason for it (e.g., when the default values lead too many extreme unlikely combinations that are not very informative about the correctness of the implementation).
4. By default, a random subset of 100 `param` x `paramdomain` combinations is tested, to keep the test runtime under control. When testing your shiny new distribution, you can temporarily set `n_samples=-1` to force all combinations to be tested. This is important to avoid your `PR` leading to surprising failures in future runs whenever some bad combinations of parameters are randomly tested.
5. On GitHub some tests run twice, under the `pytensor.config.floatX` flags of `"float64"` and `"float32"`. However, the reference Python functions will run in a pure "float64" environment, which means the reference and the PyMC results can diverge quite a lot (e.g., underflowing to `-np.inf` for extreme parameters). You should therefore make sure you test locally in both regimes. A quick and dirty way of doing this is to temporarily add `pytensor.config.floatX = "float32"` at the very top of file, immediately after `import pytensor`. Remember to set `n_samples=-1` as well to test all combinations. The test output will show what exact parameter values lead to a failure. If you are confident that your implementation is correct, you may opt to tweak the decimal precision with `select_by_precision`, or adjust the tested `Domain` values. In extreme cases, you can mark the test with a conditional `xfail` (if only one of the sub-methods is failing, they should be separated, so that the `xfail` is as narrow as possible):

```python

def test_blah_logp(self):
    ...


@pytest.mark.xfail(
   condition=(pytensor.config.floatX == "float32"),
   reason="Fails on float32 due to numerical issues",
)
def test_blah_logcdf(self):
    ...


```

## 5. Adding tests for the `support_point` method

Tests for the `support_point` make use of the function `assert_support_point_is_expected`
which checks if:
1. Moments return the `expected` values
1. Moments have the expected size and shape
1. Moments have a finite logp

```python

import pytest
from pymc.distributions import Blah
from pymc.testing import assert_support_point_is_expected


@pytest.mark.parametrize(
    "param1, param2, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(5), None, np.zeros(5)),
        (np.arange(5), 1, None, np.arange(5)),
        (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_blah_support_point(param1, param2, size, expected):
    with Model() as model:
        Blah("x", param1=param1, param2=param2, size=size)
    assert_support_point_is_expected(model, expected)

```

Here are some details worth keeping in mind:

1. In the case where you have to manually broadcast the parameters with each other it's important to add test conditions that would fail if you were not to do that. A straightforward way to do this is to make the used parameter a scalar, the unused one(s) a vector (one at a time) and size `None`.
1. In other words, make sure to test different combinations of size and broadcasting to cover these cases.

## 6. Documenting the new `Distribution`

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

The new distribution should be referenced in the respective API page in the `docs` module (e.g., `pymc/docs/api/distributions.continuous.rst`).
If appropriate, a new notebook example should be added to [pymc-examples](https://github.com/pymc-devs/pymc-examples/) illustrating how this distribution can be used and how it relates (and/or differs) from other distributions that users are more likely to be familiar with.
