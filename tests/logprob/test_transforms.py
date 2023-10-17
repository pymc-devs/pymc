#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy as sp
import scipy.special

from pytensor.graph.basic import equal_computations

from pymc.distributions.continuous import Cauchy, ChiSquared
from pymc.distributions.discrete import Bernoulli
from pymc.logprob.basic import conditional_logp, icdf, logcdf, logp
from pymc.logprob.transforms import (
    ArccoshTransform,
    ArcsinhTransform,
    ArctanhTransform,
    ChainedTransform,
    CoshTransform,
    ErfcTransform,
    ErfcxTransform,
    ErfTransform,
    ExpTransform,
    LocTransform,
    LogTransform,
    ScaleTransform,
    SinhTransform,
    TanhTransform,
    Transform,
)
from pymc.logprob.utils import ParameterValueError
from pymc.testing import Rplusbig, Vector, assert_no_rvs
from tests.distributions.test_transform import check_jacobian_det


class DirichletScipyDist:
    def __init__(self, alphas):
        self.alphas = alphas

    def rvs(self, size=None, random_state=None):
        if size is None:
            size = ()
        samples_shape = tuple(np.atleast_1d(size)) + self.alphas.shape
        samples = np.empty(samples_shape)
        alphas_bcast = np.broadcast_to(self.alphas, samples_shape)

        for index in np.ndindex(*samples_shape[:-1]):
            samples[index] = random_state.dirichlet(alphas_bcast[index])

        return samples

    def logpdf(self, value):
        res = np.sum(
            scipy.special.xlogy(self.alphas - 1, value) - scipy.special.gammaln(self.alphas),
            axis=-1,
        ) + scipy.special.gammaln(np.sum(self.alphas, axis=-1))
        return res


class TestTransform:
    @pytest.mark.parametrize("ndim", (0, 1))
    def test_fallback_log_jac_det(self, ndim):
        """
        Test fallback log_jac_det in RVTransform produces correct the graph for a
        simple transformation: x**2 -> -log(2*x)
        """

        class SquareTransform(Transform):
            name = "square"
            ndim_supp = ndim

            def forward(self, value, *inputs):
                return pt.power(value, 2)

            def backward(self, value, *inputs):
                return pt.sqrt(value)

        square_tr = SquareTransform()

        value = pt.vector("value")
        value_tr = square_tr.forward(value)
        log_jac_det = square_tr.log_jac_det(value_tr)

        test_value = np.r_[3, 4]
        expected_log_jac_det = -np.log(2 * test_value)
        if ndim == 1:
            expected_log_jac_det = expected_log_jac_det.sum()
        np.testing.assert_array_equal(log_jac_det.eval({value: test_value}), expected_log_jac_det)

    @pytest.mark.parametrize("ndim", (None, 2))
    def test_fallback_log_jac_det_undefined_ndim(self, ndim):
        class SquareTransform(Transform):
            name = "square"
            ndim_supp = ndim

            def forward(self, value, *inputs):
                return pt.power(value, 2)

            def backward(self, value, *inputs):
                return pt.sqrt(value)

        with pytest.raises(
            NotImplementedError, match=r"only implemented for ndim_supp in \(0, 1\)"
        ):
            SquareTransform().log_jac_det(0)

    def test_chained_transform(self):
        loc = 5
        scale = 0.1

        ch = ChainedTransform(
            transform_list=[
                ScaleTransform(
                    transform_args_fn=lambda *inputs: pt.constant(scale),
                ),
                ExpTransform(),
                LocTransform(
                    transform_args_fn=lambda *inputs: pt.constant(loc),
                ),
            ],
        )

        x = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
        x_val = x.eval()

        x_val_forward = ch.forward(x_val, *x.owner.inputs).eval()
        np.testing.assert_allclose(
            x_val_forward,
            np.exp(x_val * scale) + loc,
        )

        x_val_backward = ch.backward(x_val_forward, *x.owner.inputs, scale, loc).eval()
        np.testing.assert_allclose(
            x_val_backward,
            x_val,
        )

        log_jac_det = ch.log_jac_det(x_val_forward, *x.owner.inputs, scale, loc)
        np.testing.assert_allclose(
            pt.sum(log_jac_det).eval(),
            np.sum(-np.log(scale) - np.log(x_val_forward - loc)),
        )

    @pytest.mark.parametrize(
        "transform",
        [
            ErfTransform(),
            ErfcTransform(),
            ErfcxTransform(),
            SinhTransform(),
            CoshTransform(),
            TanhTransform(),
            ArcsinhTransform(),
            ArccoshTransform(),
            ArctanhTransform(),
            LogTransform(),
            ExpTransform(),
        ],
    )
    def test_check_jac_det(self, transform):
        check_jacobian_det(
            transform,
            Vector(Rplusbig, 2),
            pt.dvector,
            [0.1, 0.1],
            elemwise=True,
            rv_var=pt.random.normal(0.5, 1, name="base_rv"),
        )


def test_exp_transform_rv():
    base_rv = pt.random.normal(0, 1, size=3, name="base_rv")
    y_rv = pt.exp(base_rv)
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logp_fn = pytensor.function([y_vv], logp(y_rv, y_vv))
    logcdf_fn = pytensor.function([y_vv], logcdf(y_rv, y_vv))
    icdf_fn = pytensor.function([y_vv], icdf(y_rv, y_vv))

    y_val = [-2.0, 0.1, 0.3]
    q_val = [0.2, 0.5, 0.9]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.lognorm(s=1).logpdf(y_val),
    )
    np.testing.assert_almost_equal(
        logcdf_fn(y_val),
        sp.stats.lognorm(s=1).logcdf(y_val),
    )
    np.testing.assert_almost_equal(
        icdf_fn(q_val),
        sp.stats.lognorm(s=1).ppf(q_val),
    )


def test_meta_exp_transform_rv():
    base_rv = pt.random.normal(0, 1, size=3, name="base_rv")
    y_rv = pt.exp(base_rv)
    y_rv.name = "y"

    y_vv = y_rv.clone()

    ndim_supp_base, supp_axes_base, measure_type_base = get_measurable_meta_info(base_rv)

    ndim_supp, supp_axes, measure_type = meta_info_helper(y_rv, y_vv)

    assert np.isclose(
        ndim_supp_base,
        ndim_supp,
    )
    assert supp_axes_base == supp_axes

    assert measure_type_base == measure_type


def test_log_transform_rv():
    base_rv = pt.random.lognormal(0, 1, size=2, name="base_rv")
    y_rv = pt.log(base_rv)
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logprob = logp(y_rv, y_vv)
    logp_fn = pytensor.function([y_vv], logprob)

    y_val = [0.1, 0.3]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.norm().logpdf(y_val),
    )


class TestLocScaleRVTransform:
    @pytest.mark.parametrize(
        "rv_size, loc_type, addition",
        [
            (None, pt.scalar, True),
            (2, pt.vector, False),
            ((2, 1), pt.col, True),
        ],
    )
    def test_loc_transform_rv(self, rv_size, loc_type, addition):
        loc = loc_type("loc")
        if addition:
            y_rv = loc + pt.random.normal(0, 1, size=rv_size, name="base_rv")
        else:
            y_rv = pt.random.normal(0, 1, size=rv_size, name="base_rv") - pt.neg(loc)
        y_rv.name = "y"
        y_vv = y_rv.clone()

        logprob = logp(y_rv, y_vv)
        assert_no_rvs(logprob)
        logp_fn = pytensor.function([loc, y_vv], logprob)
        logcdf_fn = pytensor.function([loc, y_vv], logcdf(y_rv, y_vv))
        icdf_fn = pytensor.function([loc, y_vv], icdf(y_rv, y_vv))

        loc_test_val = np.full(rv_size, 4.0)
        y_test_val = np.full(rv_size, 1.0)
        q_test_val = np.full(rv_size, 0.7)
        np.testing.assert_allclose(
            logp_fn(loc_test_val, y_test_val),
            sp.stats.norm(loc_test_val, 1).logpdf(y_test_val),
        )
        np.testing.assert_allclose(
            logcdf_fn(loc_test_val, y_test_val),
            sp.stats.norm(loc_test_val, 1).logcdf(y_test_val),
        )
        np.testing.assert_allclose(
            icdf_fn(loc_test_val, q_test_val),
            sp.stats.norm(loc_test_val, 1).ppf(q_test_val),
        )

    @pytest.mark.parametrize(
        "rv_size, scale_type, product",
        [
            (None, pt.scalar, True),
            (1, pt.TensorType("floatX", (True,)), True),
            ((2, 3), pt.matrix, False),
        ],
    )
    def test_scale_transform_rv(self, rv_size, scale_type, product):
        scale = scale_type("scale")
        if product:
            y_rv = pt.random.normal(0, 1, size=rv_size, name="base_rv") * scale
        else:
            y_rv = pt.random.normal(0, 1, size=rv_size, name="base_rv") / pt.reciprocal(scale)
        y_rv.name = "y"
        y_vv = y_rv.clone()

        logprob = logp(y_rv, y_vv)
        assert_no_rvs(logprob)
        logp_fn = pytensor.function([scale, y_vv], logprob)
        logcdf_fn = pytensor.function([scale, y_vv], logcdf(y_rv, y_vv))
        icdf_fn = pytensor.function([scale, y_vv], icdf(y_rv, y_vv))

        scale_test_val = np.full(rv_size, 4.0)
        y_test_val = np.full(rv_size, 1.0)
        q_test_val = np.full(rv_size, 0.3)
        np.testing.assert_allclose(
            logp_fn(scale_test_val, y_test_val),
            sp.stats.norm(0, scale_test_val).logpdf(y_test_val),
        )
        np.testing.assert_allclose(
            logcdf_fn(scale_test_val, y_test_val),
            sp.stats.norm(0, scale_test_val).logcdf(y_test_val),
        )
        np.testing.assert_allclose(
            icdf_fn(scale_test_val, q_test_val),
            sp.stats.norm(0, scale_test_val).ppf(q_test_val),
        )

    def test_negated_rv_transform(self):
        x_rv = -pt.random.halfnormal()
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))
        x_logcdf_fn = pytensor.function([x_vv], logcdf(x_rv, x_vv))
        x_icdf_fn = pytensor.function([x_vv], icdf(x_rv, x_vv))

        np.testing.assert_allclose(x_logp_fn(-1.5), sp.stats.halfnorm.logpdf(1.5))
        np.testing.assert_allclose(x_logcdf_fn(-1.5), sp.stats.halfnorm.logsf(1.5))
        np.testing.assert_allclose(x_icdf_fn(0.3), -sp.stats.halfnorm.ppf(1 - 0.3))

    def test_subtracted_rv_transform(self):
        # Choose base RV that is asymmetric around zero
        x_rv = 5.0 - pt.random.normal(1.0)
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], pt.sum(logp(x_rv, x_vv)))

        np.testing.assert_allclose(x_logp_fn(7.3), sp.stats.norm.logpdf(5.0 - 7.3, 1.0))

    def test_loc_transform_multiple_rvs_fails1(self):
        x_rv1 = pt.random.normal(name="x_rv1")
        x_rv2 = pt.random.normal(name="x_rv2")
        y_rv = x_rv1 + x_rv2

        y = y_rv.clone()

        with pytest.raises(RuntimeError, match="could not be derived"):
            conditional_logp({y_rv: y})

    def test_nested_loc_transform_multiple_rvs_fails2(self):
        x_rv1 = pt.random.normal(name="x_rv1")
        x_rv2 = pt.cos(pt.random.normal(name="x_rv2"))
        y_rv = x_rv1 + x_rv2

        y = y_rv.clone()

        with pytest.raises(RuntimeError, match="could not be derived"):
            conditional_logp({y_rv: y})


class TestPowerRVTransform:
    @pytest.mark.parametrize("numerator", (1.0, 2.0))
    def test_reciprocal_rv_transform(self, numerator):
        shape = 3
        scale = 5
        x_rv = numerator / pt.random.gamma(shape, scale, size=(2,))
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))
        x_logcdf_fn = pytensor.function([x_vv], logcdf(x_rv, x_vv))

        with pytest.raises(NotImplementedError):
            icdf(x_rv, x_vv)

        x_test_val = np.r_[-0.5, 1.5]
        np.testing.assert_allclose(
            x_logp_fn(x_test_val),
            sp.stats.invgamma(shape, scale=scale * numerator).logpdf(x_test_val),
        )
        np.testing.assert_allclose(
            x_logcdf_fn(x_test_val),
            sp.stats.invgamma(shape, scale=scale * numerator).logcdf(x_test_val),
        )

    def test_reciprocal_real_rv_transform(self):
        # 1 / Cauchy(mu, sigma) = Cauchy(mu / (mu^2 + sigma ^2), sigma / (mu ^ 2, sigma ^ 2))
        test_value = [-0.5, 0.9]
        test_rv = Cauchy.dist(1, 2, size=(2,)) ** (-1)

        np.testing.assert_allclose(
            logp(test_rv, test_value).eval(),
            sp.stats.cauchy(1 / 5, 2 / 5).logpdf(test_value),
        )
        np.testing.assert_allclose(
            logcdf(test_rv, test_value).eval(),
            sp.stats.cauchy(1 / 5, 2 / 5).logcdf(test_value),
        )
        with pytest.raises(NotImplementedError):
            icdf(test_rv, test_value)

    def test_sqr_transform(self):
        # The square of a normal with unit variance is a noncentral chi-square with 1 df and nc = mean ** 2
        x_rv = pt.random.normal(0.5, 1, size=(4,)) ** 2
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))

        with pytest.raises(NotImplementedError):
            logcdf(x_rv, x_vv)

        with pytest.raises(NotImplementedError):
            icdf(x_rv, x_vv)

        x_test_val = np.r_[-0.5, 0.5, 1, 2.5]
        np.testing.assert_allclose(
            x_logp_fn(x_test_val),
            sp.stats.ncx2(df=1, nc=0.5**2).logpdf(x_test_val),
        )

    def test_sqrt_transform(self):
        # The sqrt of a chisquare with n df is a chi distribution with n df
        x_rv = pt.sqrt(ChiSquared.dist(nu=3, size=(4,)))
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))
        x_logcdf_fn = pytensor.function([x_vv], logcdf(x_rv, x_vv))

        x_test_val = np.r_[-2.5, 0.5, 1, 2.5]
        np.testing.assert_allclose(
            x_logp_fn(x_test_val),
            sp.stats.chi(df=3).logpdf(x_test_val),
        )
        np.testing.assert_allclose(
            x_logcdf_fn(x_test_val),
            sp.stats.chi(df=3).logcdf(x_test_val),
        )

        # ICDF is not implemented for chisquare, so we have to test with another identity
        # sqrt(exponential(lam)) = rayleigh(1 / sqrt(2 * lam))
        lam = 2.5
        y_rv = pt.sqrt(pt.random.exponential(scale=1 / lam))
        y_vv = x_rv.clone()
        y_icdf_fn = pytensor.function([y_vv], icdf(y_rv, y_vv))
        q_test_val = np.r_[0.2, 0.5, 0.7, 0.9]
        np.testing.assert_allclose(
            y_icdf_fn(q_test_val),
            (1 / np.sqrt(2 * lam)) * np.sqrt(-2 * np.log(1 - q_test_val)),
        )

    @pytest.mark.parametrize("power", (-3, -1, 1, 5, 7))
    def test_negative_value_odd_power_transform(self, power):
        # check that negative values and odd powers evaluate to a finite logp
        x_rv = pt.random.normal() ** power
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))

        assert np.isfinite(x_logp_fn(1))
        assert np.isfinite(x_logp_fn(-1))

    @pytest.mark.parametrize("power", (-2, 2, 4, 6, 8))
    def test_negative_value_even_power_transform_logp(self, power):
        # check that negative values and odd powers evaluate to -inf logp
        x_rv = pt.random.normal() ** power
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))

        assert np.isfinite(x_logp_fn(1))
        assert np.isneginf(x_logp_fn(-1))

    @pytest.mark.parametrize("power", (-1 / 3, -1 / 2, 1 / 2, 1 / 3))
    def test_negative_value_frac_power_transform_logp(self, power):
        # check that negative values and fractional powers evaluate to -inf logp
        x_rv = pt.random.normal() ** power
        x_rv.name = "x"

        x_vv = x_rv.clone()
        x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))

        assert np.isfinite(x_logp_fn(2.5))
        assert np.isneginf(x_logp_fn(-2.5))


@pytest.mark.parametrize("test_val", (2.5, -2.5))
def test_absolute_rv_transform(test_val):
    x_rv = pt.abs(pt.random.normal())
    y_rv = pt.random.halfnormal()

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    x_logp_fn = pytensor.function([x_vv], logp(x_rv, x_vv))
    with pytest.raises(NotImplementedError):
        logcdf(x_rv, x_vv)
    with pytest.raises(NotImplementedError):
        icdf(x_rv, x_vv)

    y_logp_fn = pytensor.function([y_vv], logp(y_rv, y_vv))
    np.testing.assert_allclose(x_logp_fn(test_val), y_logp_fn(test_val))


@pytest.mark.parametrize(
    "pt_transform, transform",
    [
        (pt.erf, ErfTransform()),
        (pt.erfc, ErfcTransform()),
        (pt.erfcx, ErfcxTransform()),
        (pt.sinh, SinhTransform()),
        (pt.tanh, TanhTransform()),
        (pt.arcsinh, ArcsinhTransform()),
        (pt.arccosh, ArccoshTransform()),
        (pt.arctanh, ArctanhTransform()),
    ],
)
def test_extra_bijective_rv_transforms(pt_transform, transform):
    base_rv = pt.random.normal(
        0.5, 1, name="base_rv"
    )  # Something not centered around 0 is usually better
    rv = pt_transform(base_rv)

    vv = rv.clone()
    rv_logp = logp(rv, vv)

    expected_logp = logp(base_rv, transform.backward(vv)) + transform.log_jac_det(vv)

    vv_test = np.array(0.25)  # Arbitrary test value
    np.testing.assert_allclose(
        rv_logp.eval({vv: vv_test}),
        np.nan_to_num(expected_logp.eval({vv: vv_test}), nan=-np.inf),
    )


def test_cosh_rv_transform():
    # Something not centered around 0 is usually better
    base_rv = pt.random.normal(0.5, 1, size=(2,), name="base_rv")
    rv = pt.cosh(base_rv)
    vv = rv.clone()
    rv_logp = logp(rv, vv)
    with pytest.raises(NotImplementedError):
        logcdf(rv, vv)
    with pytest.raises(NotImplementedError):
        icdf(rv, vv)

    transform = CoshTransform()
    [back_neg, back_pos] = transform.backward(vv)
    expected_logp = pt.logaddexp(
        logp(base_rv, back_neg), logp(base_rv, back_pos)
    ) + transform.log_jac_det(vv)
    vv_test = np.array([0.25, 1.5])
    np.testing.assert_allclose(
        rv_logp.eval({vv: vv_test}),
        np.nan_to_num(expected_logp.eval({vv: vv_test}), nan=-np.inf),
    )


@pytest.mark.parametrize(
    "canonical_func,raw_func",
    [
        (pt.log1p, lambda x: pt.log(1 + x)),
        (pt.softplus, lambda x: pt.log(1 + pt.exp(x))),
        (pt.log1mexp, lambda x: pt.log(1 - pt.exp(x))),
        (pt.log2, lambda x: pt.log(x) / pt.log(2)),
        (pt.log10, lambda x: pt.log(x) / pt.log(10)),
        (pt.exp2, lambda x: pt.exp(pt.log(2) * x)),
        (pt.expm1, lambda x: pt.exp(x) - 1),
        (pt.sigmoid, lambda x: 1 / (1 + pt.exp(-x))),
        (pt.sigmoid, lambda x: pt.exp(x) / (1 + pt.exp(x))),
    ],
)
def test_special_log_exp_transforms(canonical_func, raw_func):
    base_rv = pt.random.normal(name="base_rv")
    vv = pt.scalar("vv")

    transformed_rv = raw_func(base_rv)
    ref_transformed_rv = canonical_func(base_rv)

    logp_test = logp(transformed_rv, vv)
    logp_ref = logp(ref_transformed_rv, vv)

    if canonical_func in (pt.log2, pt.log10):
        # in the cases of log2 and log10 floating point inprecision causes failure
        # from equal_computations so evaluate logp and check all close instead
        vv_test = np.array(0.25)
        np.testing.assert_allclose(logp_ref.eval({vv: vv_test}), logp_test.eval({vv: vv_test}))
    else:
        assert equal_computations([logp_test], [logp_ref])


def test_measurable_power_exponent_with_constant_base():
    # test power(2, rv) = exp2(rv)
    # test negative base fails
    x_rv_pow = pt.pow(2, pt.random.normal())
    x_rv_exp2 = pt.exp2(pt.random.normal())

    x_vv_pow = x_rv_pow.clone()
    x_vv_exp2 = x_rv_exp2.clone()

    x_logp_fn_pow = pytensor.function([x_vv_pow], pt.sum(logp(x_rv_pow, x_vv_pow)))
    x_logp_fn_exp2 = pytensor.function([x_vv_exp2], pt.sum(logp(x_rv_exp2, x_vv_exp2)))

    np.testing.assert_allclose(x_logp_fn_pow(0.1), x_logp_fn_exp2(0.1))

    with pytest.raises(ParameterValueError, match="base >= 0"):
        x_rv_neg = pt.pow(-2, pt.random.normal())
        x_vv_neg = x_rv_neg.clone()
        logp(x_rv_neg, x_vv_neg)


def test_measurable_power_exponent_with_variable_base():
    # test with RV when logp(<0) we raise error
    base_rv = pt.random.normal([2])
    x_raw_rv = pt.random.normal()
    x_rv = pt.power(base_rv, x_raw_rv)

    x_rv.name = "x"
    base_rv.name = "base"
    base_vv = base_rv.clone()
    x_vv = x_rv.clone()

    res = conditional_logp({base_rv: base_vv, x_rv: x_vv})
    x_logp = res[x_vv]
    logp_vals_fn = pytensor.function([base_vv, x_vv], x_logp)

    with pytest.raises(ParameterValueError, match="base >= 0"):
        logp_vals_fn(np.array([-2]), np.array([2]))


def test_base_exponent_non_measurable():
    # test dual sources of measuravility fails
    base_rv = pt.random.normal([2])
    x_raw_rv = pt.random.normal()
    x_rv = pt.power(base_rv, x_raw_rv)
    x_rv.name = "x"

    x_vv = x_rv.clone()

    with pytest.raises(
        RuntimeError,
        match="The logprob terms of the following value variables could not be derived: {x}",
    ):
        conditional_logp({x_rv: x_vv})


@pytest.mark.parametrize("shift", [1.5, np.array([-0.5, 1, 0.3])])
@pytest.mark.parametrize("scale", [2.0, np.array([1.5, 3.3, 1.0])])
def test_multivariate_rv_transform(shift, scale):
    mu = np.array([0, 0.9, -2.1])
    cov = np.array([[1, 0, 0.9], [0, 1, 0], [0.9, 0, 1]])
    x_rv_raw = pt.random.multivariate_normal(mu, cov=cov)
    x_rv = shift + x_rv_raw * scale
    x_rv.name = "x"

    x_vv = x_rv.clone()
    logp = conditional_logp({x_rv: x_vv})[x_vv]
    assert_no_rvs(logp)

    x_vv_test = np.array([5.0, 4.9, -6.3])
    scale_mat = scale * np.eye(x_vv_test.shape[0])
    np.testing.assert_allclose(
        logp.eval({x_vv: x_vv_test}),
        sp.stats.multivariate_normal.logpdf(
            x_vv_test,
            shift + mu * scale,
            scale_mat @ cov @ scale_mat.T,
        ),
    )


def test_not_implemented_discrete_rv_transform():
    y_rv = pt.exp(pt.random.poisson(1))
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({y_rv: y_rv.clone()})

    y_rv = 5 * pt.random.poisson(1)
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({y_rv: y_rv.clone()})


def test_negated_discrete_rv_transform():
    p = 0.7
    rv = -Bernoulli.dist(p=p)
    vv = rv.type()
    logp_fn = pytensor.function([vv], logp(rv, vv))

    # A negated Bernoulli has pmf {p if x == -1; 1-p if x == 0; 0 otherwise}
    assert logp_fn(-2) == -np.inf
    np.testing.assert_allclose(logp_fn(-1), np.log(p))
    np.testing.assert_allclose(logp_fn(0), np.log(1 - p))
    assert logp_fn(1) == -np.inf

    # Logcdf and icdf not supported yet
    for func in (logcdf, icdf):
        with pytest.raises(NotImplementedError):
            func(rv, 0)


def test_shifted_discrete_rv_transform():
    p = 0.7
    rv = Bernoulli.dist(p=p) + 5
    vv = rv.type()
    rv_logp_fn = pytensor.function([vv], logp(rv, vv))

    assert rv_logp_fn(4) == -np.inf
    np.testing.assert_allclose(rv_logp_fn(5), np.log(1 - p))
    np.testing.assert_allclose(rv_logp_fn(6), np.log(p))
    assert rv_logp_fn(7) == -np.inf

    # Logcdf and icdf not supported yet
    for func in (logcdf, icdf):
        with pytest.raises(NotImplementedError):
            func(rv, 0)


@pytest.mark.xfail(reason="Check not implemented yet")
def test_invalid_broadcasted_transform_rv_fails():
    loc = pt.vector("loc")
    y_rv = loc + pt.random.normal(0, 1, size=1, name="base_rv")
    y_rv.name = "y"
    y_vv = y_rv.clone()

    # This logp derivation should fail or count only once the values that are broadcasted
    logprob = logp(y_rv, y_vv)
    assert logprob.eval({y_vv: [0, 0, 0, 0], loc: [0, 0, 0, 0]}).shape == ()
