#   Copyright 2023 The PyMC Developers
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

from numdifftools import Jacobian
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.scan import scan

from pymc.distributions.continuous import Cauchy
from pymc.distributions.transforms import _default_transform, log, logodds
from pymc.logprob.abstract import MeasurableVariable, _logprob
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
    IntervalTransform,
    LocTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    ScaleTransform,
    SinhTransform,
    TanhTransform,
    TransformValuesMapping,
    TransformValuesRewrite,
)
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


@pytest.fixture(scope="module")
def multiout_measurable_op():
    # Create a dummy Op that just returns the two inputs
    mu1, mu2 = pt.scalars("mu1", "mu2")

    class TestOpFromGraph(OpFromGraph):
        def do_constant_folding(self, fgraph, node):
            False

    multiout_op = TestOpFromGraph([mu1, mu2], [mu1 + 0.0, mu2 + 0.0])

    MeasurableVariable.register(TestOpFromGraph)

    @_logprob.register(TestOpFromGraph)
    def logp_multiout(op, values, mu1, mu2):
        value1, value2 = values
        return value1 + mu1, value2 + mu2

    return multiout_op


class TestValueTransformRewrite:
    def test_TransformValuesMapping(self):
        x = pt.vector()
        fg = FunctionGraph(outputs=[x])

        tvm = TransformValuesMapping({})
        fg.attach_feature(tvm)

        tvm2 = TransformValuesMapping({})
        fg.attach_feature(tvm2)

        assert fg._features[-1] is tvm

    def test_original_values_output_dict(self):
        """
        Test that the original unconstrained value variable appears an the key of
        the logprob factor
        """
        p_rv = pt.random.beta(1, 1, name="p")
        p_vv = p_rv.clone()

        tr = TransformValuesRewrite({p_vv: logodds})
        logp_dict = conditional_logp({p_rv: p_vv}, extra_rewrites=tr)

        assert p_vv in logp_dict

    @pytest.mark.parametrize(
        "pt_dist, dist_params, sp_dist, size",
        [
            (pt.random.uniform, (0, 1), sp.stats.uniform, ()),
            (
                pt.random.pareto,
                (1.5, 10.5),
                lambda b, scale: sp.stats.pareto(b, scale=scale),
                (),
            ),
            (
                pt.random.triangular,
                (1.5, 3.0, 10.5),
                lambda lower, mode, upper: sp.stats.triang(
                    (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
                ),
                (),
            ),
            (
                pt.random.halfnormal,
                (0, 1),
                sp.stats.halfnorm,
                (),
            ),
            pytest.param(
                pt.random.wald,
                (1.5, 10.5),
                lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
                (),
                marks=pytest.mark.xfail(
                    reason="We don't use PyTensor's Wald operator",
                    raises=NotImplementedError,
                ),
            ),
            (
                pt.random.exponential,
                (1.5,),
                lambda mu: sp.stats.expon(scale=mu),
                (),
            ),
            pytest.param(
                pt.random.lognormal,
                (-1.5, 10.5),
                lambda mu, sigma: sp.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)),
                (),
            ),
            (
                pt.random.lognormal,
                (-1.5, 1.5),
                lambda mu, sigma: sp.stats.lognorm(s=sigma, scale=np.exp(mu)),
                (),
            ),
            (
                pt.random.halfcauchy,
                (1.5, 10.5),
                lambda alpha, beta: sp.stats.halfcauchy(loc=alpha, scale=beta),
                (),
            ),
            (
                pt.random.gamma,
                (1.5, 10.5),
                lambda alpha, inv_beta: sp.stats.gamma(alpha, scale=1.0 / inv_beta),
                (),
            ),
            (
                pt.random.invgamma,
                (1.5, 10.5),
                lambda alpha, beta: sp.stats.invgamma(alpha, scale=beta),
                (),
            ),
            (
                pt.random.chisquare,
                (1.5,),
                lambda df: sp.stats.chi2(df),
                (),
            ),
            pytest.param(
                pt.random.weibull,
                (1.5,),
                lambda c: sp.stats.weibull_min(c),
                (),
                marks=pytest.mark.xfail(
                    reason="We don't use PyTensor's Weibull operator",
                    raises=NotImplementedError,
                ),
            ),
            (
                pt.random.beta,
                (1.5, 1.5),
                lambda alpha, beta: sp.stats.beta(alpha, beta),
                (),
            ),
            (
                pt.random.vonmises,
                (1.5, 10.5),
                lambda mu, kappa: sp.stats.vonmises(kappa, loc=mu),
                (),
            ),
            (
                pt.random.dirichlet,
                (np.array([0.7, 0.3]),),
                lambda alpha: sp.stats.dirichlet(alpha),
                (),
            ),
            (
                pt.random.dirichlet,
                (np.array([[0.7, 0.3], [0.9, 0.1]]),),
                lambda alpha: DirichletScipyDist(alpha),
                (),
            ),
            pytest.param(
                pt.random.dirichlet,
                (np.array([0.3, 0.7]),),
                lambda alpha: DirichletScipyDist(alpha),
                (3, 2),
            ),
        ],
    )
    def test_value_transform_logprob(self, pt_dist, dist_params, sp_dist, size):
        """
        This test takes a `RandomVariable` type, plus parameters, and uses it to
        construct a variable ``a`` that's used in the graph ``b =
        pt.random.normal(a, 1.0)``.  The transformed log-probability is then
        computed for ``b``.  We then test that the log-probability of ``a`` is
        properly transformed, as well as any instances of ``a`` that are used
        elsewhere in the graph (i.e. in ``b``), by comparing the graph for the
        transformed log-probability with the SciPy-derived log-probability--using a
        numeric approximation to the Jacobian term.

        TODO: This test is rather redundant with those in tess/distributions/test_transform.py
        """

        a = pt_dist(*dist_params, size=size)
        a.name = "a"
        a_value_var = pt.tensor(dtype=a.dtype, shape=(None,) * a.ndim)
        a_value_var.name = "a_value"

        b = pt.random.normal(a, 1.0)
        b.name = "b"
        b_value_var = b.clone()
        b_value_var.name = "b_value"

        transform = _default_transform(a.owner.op, a)
        transform_rewrite = TransformValuesRewrite({a_value_var: transform})
        res = conditional_logp({a: a_value_var, b: b_value_var}, extra_rewrites=transform_rewrite)
        res_combined = pt.sum([pt.sum(factor) for factor in res.values()])

        test_val_rng = np.random.RandomState(3238)

        logp_vals_fn = pytensor.function([a_value_var, b_value_var], res_combined)

        a_forward_fn = pytensor.function(
            [a_value_var], transform.forward(a_value_var, *a.owner.inputs)
        )
        a_backward_fn = pytensor.function(
            [a_value_var], transform.backward(a_value_var, *a.owner.inputs)
        )
        log_jac_fn = pytensor.function(
            [a_value_var],
            transform.log_jac_det(a_value_var, *a.owner.inputs),
            on_unused_input="ignore",
        )

        for i in range(10):
            a_dist = sp_dist(*dist_params)
            a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(a_value_var.dtype)
            b_dist = sp.stats.norm(a_val, 1.0)
            b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

            a_trans_value = a_forward_fn(a_val)

            if a_val.ndim > 0:

                def jacobian_estimate_novec(value):
                    dim_diff = a_val.ndim - value.ndim  # pylint: disable=cell-var-from-loop
                    if dim_diff > 0:
                        # Make sure the dimensions match the expected input
                        # dimensions for the compiled backward transform function
                        def a_backward_fn_(x):
                            x_ = np.expand_dims(x, axis=list(range(dim_diff)))
                            return a_backward_fn(x_).squeeze()

                    else:
                        a_backward_fn_ = a_backward_fn

                    jacobian_val = Jacobian(a_backward_fn_)(value)

                    n_missing_dims = jacobian_val.shape[0] - jacobian_val.shape[1]
                    if n_missing_dims > 0:
                        missing_bases = np.eye(jacobian_val.shape[0])[..., -n_missing_dims:]
                        jacobian_val = np.concatenate([jacobian_val, missing_bases], axis=-1)

                    return np.linalg.slogdet(jacobian_val)[-1]

                jacobian_estimate = np.vectorize(jacobian_estimate_novec, signature="(n)->()")

                exp_log_jac_val = jacobian_estimate(a_trans_value)
            else:
                jacobian_val = np.atleast_2d(
                    sp.misc.derivative(a_backward_fn, a_trans_value, dx=1e-6)
                )
                exp_log_jac_val = np.linalg.slogdet(jacobian_val)[-1]

            log_jac_val = log_jac_fn(a_trans_value)
            np.testing.assert_allclose(exp_log_jac_val, log_jac_val, rtol=1e-4, atol=1e-10)

            exp_logprob_val = a_dist.logpdf(a_val).sum()
            exp_logprob_val += exp_log_jac_val.sum()
            exp_logprob_val += b_dist.logpdf(b_val).sum()

            logprob_val = logp_vals_fn(a_trans_value, b_val)

            np.testing.assert_allclose(exp_logprob_val, logprob_val, rtol=1e-4, atol=1e-10)

    @pytest.mark.parametrize("use_jacobian", [True, False])
    def test_value_transform_logprob_nojac(self, use_jacobian):
        X_rv = pt.random.halfnormal(0, 3, name="X")
        x_vv = X_rv.clone()
        x_vv.name = "x"

        transform_rewrite = TransformValuesRewrite({x_vv: log})
        tr_logp = conditional_logp(
            {X_rv: x_vv}, extra_rewrites=transform_rewrite, use_jacobian=use_jacobian
        )
        tr_logp_combined = pt.sum([pt.sum(factor) for factor in tr_logp.values()])

        np.testing.assert_allclose(
            tr_logp_combined.eval({x_vv: np.log(2.5)}),
            sp.stats.halfnorm(0, 3).logpdf(2.5) + (np.log(2.5) if use_jacobian else 0.0),
        )

    def test_hierarchical_value_transform(self):
        """
        This model requires rv-value replacements in the backward transformation of
        the value var `x`
        """

        lower_rv = pt.random.uniform(0, 1, name="lower")
        upper_rv = pt.random.uniform(9, 10, name="upper")
        x_rv = pt.random.uniform(lower_rv, upper_rv, name="x")

        lower = lower_rv.clone()
        upper = upper_rv.clone()
        x = x_rv.clone()

        transform_rewrite = TransformValuesRewrite(
            {
                lower: _default_transform(lower_rv.owner.op, lower_rv),
                upper: _default_transform(upper_rv.owner.op, upper_rv),
                x: _default_transform(x_rv.owner.op, x_rv),
            }
        )
        logp = conditional_logp(
            {lower_rv: lower, upper_rv: upper, x_rv: x},
            extra_rewrites=transform_rewrite,
        )
        logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

        assert_no_rvs(logp_combined)
        assert not np.isinf(logp_combined.eval({lower: -10, upper: 20, x: -20}))

    def test_nondefault_value_transform(self):
        loc_rv = pt.random.uniform(-10, 10, name="loc")
        scale_rv = pt.random.uniform(-1, 1, name="scale")
        x_rv = pt.random.normal(loc_rv, scale_rv, name="x")

        loc = loc_rv.clone()
        scale = scale_rv.clone()
        x = x_rv.clone()

        transform_rewrite = TransformValuesRewrite(
            {
                loc: None,
                scale: LogOddsTransform(),
                x: LogTransform(),
            }
        )

        logp = conditional_logp(
            {loc_rv: loc, scale_rv: scale, x_rv: x},
            extra_rewrites=transform_rewrite,
        )
        logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

        # Check numerical evaluation matches with expected transforms
        loc_val = 0
        scale_val_tr = -1
        x_val_tr = -1

        scale_val = sp.special.expit(scale_val_tr)
        x_val = np.exp(x_val_tr)

        exp_logp = 0
        exp_logp += sp.stats.uniform(-10, 20).logpdf(loc_val)
        exp_logp += sp.stats.uniform(-1, 2).logpdf(scale_val)
        exp_logp += np.log(scale_val) + np.log1p(-scale_val)  # logodds log_jac_det
        exp_logp += sp.stats.norm(loc_val, scale_val).logpdf(x_val)
        exp_logp += x_val_tr  # log log_jac_det

        np.testing.assert_allclose(
            logp_combined.eval({loc: loc_val, scale: scale_val_tr, x: x_val_tr}),
            exp_logp,
        )

    def test_no_value_transform_multiout_input(self):
        r"""Make sure that `Op`\s with multiple outputs are handled correctly."""

        # This SVD value is necessarily `1`, but it's generated by an `Op` with
        # multiple outputs and no default output.
        sd = pt.linalg.svd(pt.eye(1))[1][0]
        x_rv = pt.random.normal(0, sd, name="x")
        x = x_rv.clone()

        transform_rewrite = TransformValuesRewrite({x: None})

        logp = conditional_logp(
            {x_rv: x},
            extra_rewrites=transform_rewrite,
        )
        logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

        np.testing.assert_allclose(
            logp_combined.eval({x: 1}),
            sp.stats.norm(0, 1).logpdf(1),
        )

    @pytest.mark.parametrize("transform_x", (True, False))
    @pytest.mark.parametrize("transform_y", (True, False))
    def test_value_transform_multiout_op(self, transform_x, transform_y, multiout_measurable_op):
        x, y = multiout_measurable_op(1, 2)
        x.name = "x"
        y.name = "y"
        x_vv = x.clone()
        y_vv = y.clone()

        transform_rewrite = TransformValuesRewrite(
            {
                x_vv: LogTransform() if transform_x else None,
                y_vv: ExpTransform() if transform_y else None,
            }
        )

        logp = conditional_logp({x: x_vv, y: y_vv}, extra_rewrites=transform_rewrite)
        logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

        x_vv_test = np.random.normal()
        y_vv_test = np.abs(np.random.normal())

        expected_logp = 0
        if not transform_x:
            expected_logp += x_vv_test + 1
        else:
            expected_logp += np.exp(x_vv_test) + 1 + x_vv_test
        # y logp
        if not transform_y:
            expected_logp += y_vv_test + 2
        else:
            expected_logp += np.log(y_vv_test) + 2 - np.log(y_vv_test)

        np.testing.assert_allclose(
            logp_combined.eval({x_vv: x_vv_test, y_vv: y_vv_test}), expected_logp
        )

    def test_transformed_rv_and_value(self):
        y_rv = pt.random.halfnormal(-1, 1, name="base_rv") + 1
        y_rv.name = "y"
        y_vv = y_rv.clone()

        transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

        logp = conditional_logp({y_rv: y_vv}, extra_rewrites=transform_rewrite)
        logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
        assert_no_rvs(logp_combined)
        logp_fn = pytensor.function([y_vv], logp_combined)

        y_test_val = -5

        np.testing.assert_allclose(
            logp_fn(y_test_val),
            sp.stats.halfnorm(0, 1).logpdf(np.exp(y_test_val)) + y_test_val,
        )

    @pytest.mark.filterwarnings("error")
    def test_mixture_transform(self):
        """Make sure that non-`RandomVariable` `MeasurableVariable`s can be transformed.

        This test is specific to `MixtureRV`, which is derived from an `OpFromGraph`.
        """

        I_rv = pt.random.bernoulli(0.5, name="I")
        Y_1_rv = pt.random.beta(100, 1, name="Y_1")
        Y_2_rv = pt.random.beta(1, 100, name="Y_2")

        # A `MixtureRV`, which is an `OpFromGraph` subclass, will replace this
        # `pt.stack` in the graph
        Y_rv = pt.stack([Y_1_rv, Y_2_rv])[I_rv]
        Y_rv.name = "Y"

        i_vv = I_rv.clone()
        i_vv.name = "i"
        y_vv = Y_rv.clone()
        y_vv.name = "y"

        logp_no_trans = conditional_logp(
            {Y_rv: y_vv, I_rv: i_vv},
        )
        logp_no_trans_comb = pt.sum([pt.sum(factor) for factor in logp_no_trans.values()])

        transform_rewrite = TransformValuesRewrite({y_vv: LogTransform()})

        logp_trans = conditional_logp(
            {Y_rv: y_vv, I_rv: i_vv},
            extra_rewrites=transform_rewrite,
            use_jacobian=False,
        )
        logp_trans_combined = pt.sum([pt.sum(factor) for factor in logp_trans.values()])

        # The untransformed graph should be the same as the transformed graph after
        # replacing the `Y_rv` value variable with a transformed version of itself
        logp_nt_fg = FunctionGraph(outputs=[logp_no_trans_comb], clone=False)
        y_trans = pt.exp(y_vv)
        y_trans.name = "y_log"
        logp_nt_fg.replace(y_vv, y_trans)
        logp_nt = logp_nt_fg.outputs[0]

        assert equal_computations([logp_nt], [logp_trans_combined])

    def test_scan_transform(self):
        """Test that Scan valued variables can be transformed"""

        init = pt.random.beta(1, 1, name="init")
        init_vv = init.clone()

        def scan_step(prev_innov):
            next_innov = pt.random.beta(prev_innov * 10, (1 - prev_innov) * 10)
            update = {next_innov.owner.inputs[0]: next_innov.owner.outputs[0]}
            return next_innov, update

        innov, _ = scan(
            fn=scan_step,
            outputs_info=[init],
            n_steps=4,
        )
        innov.name = "innov"
        innov_vv = innov.clone()

        tr = TransformValuesRewrite(
            {
                init_vv: LogOddsTransform(),
                innov_vv: LogOddsTransform(),
            }
        )
        logp = conditional_logp(
            {init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True
        )[innov_vv]
        logp_fn = pytensor.function([init_vv, innov_vv], logp, on_unused_input="ignore")

        # Create an unrolled scan graph as reference
        innov = []
        prev_innov = init
        for i in range(4):
            next_innov = pt.random.beta(prev_innov * 10, (1 - prev_innov) * 10, name=f"innov[i]")
            innov.append(next_innov)
            prev_innov = next_innov
        innov = pt.stack(innov)
        innov.name = "innov"

        tr = TransformValuesRewrite(
            {
                init_vv: LogOddsTransform(),
                innov_vv: LogOddsTransform(),
            }
        )
        ref_logp = conditional_logp(
            {init: init_vv, innov: innov_vv}, extra_rewrites=tr, use_jacobian=True
        )[innov_vv]
        ref_logp_fn = pytensor.function([init_vv, innov_vv], ref_logp, on_unused_input="ignore")

        test_point = {
            "init": np.array(-0.5),
            "innov": np.full((4,), -0.5),
        }
        np.testing.assert_allclose(logp_fn(**test_point), ref_logp_fn(**test_point))


class TestRVTransform:
    @pytest.mark.parametrize("ndim", (0, 1))
    def test_fallback_log_jac_det(self, ndim):
        """
        Test fallback log_jac_det in RVTransform produces correct the graph for a
        simple transformation: x**2 -> -log(2*x)
        """

        class SquareTransform(RVTransform):
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
        class SquareTransform(RVTransform):
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

    def test_invalid_interval_transform(self):
        x_rv = pt.random.normal(0, 1)
        x_vv = x_rv.clone()

        msg = "Both edges of IntervalTransform cannot be None"
        tr = IntervalTransform(lambda *inputs: (None, None))
        with pytest.raises(ValueError, match=msg):
            tr.forward(x_vv, *x_rv.owner.inputs)

        tr = IntervalTransform(lambda *inputs: (None, None))
        with pytest.raises(ValueError, match=msg):
            tr.backward(x_vv, *x_rv.owner.inputs)

        tr = IntervalTransform(lambda *inputs: (None, None))
        with pytest.raises(ValueError, match=msg):
            tr.log_jac_det(x_vv, *x_rv.owner.inputs)

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
            base_op=pt.random.multivariate_normal,
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
        x_rv = pt.sqrt(pt.random.chisquare(df=3, size=(4,)))
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


TRANSFORMATIONS = {
    "log1p": (pt.log1p, lambda x: pt.log(1 + x)),
    "softplus": (pt.softplus, lambda x: pt.log(1 + pt.exp(x))),
    "log1mexp": (pt.log1mexp, lambda x: pt.log(1 - pt.exp(x))),
    "log2": (pt.log2, lambda x: pt.log(x) / pt.log(2)),
    "log10": (pt.log10, lambda x: pt.log(x) / pt.log(10)),
    "exp2": (pt.exp2, lambda x: pt.exp(pt.log(2) * x)),
    "expm1": (pt.expm1, lambda x: pt.exp(x) - 1),
    "sigmoid": (pt.sigmoid, lambda x: 1 / (1 + pt.exp(-x))),
}


@pytest.mark.parametrize("transform", TRANSFORMATIONS.keys())
def test_special_log_exp_transforms(transform):
    base_rv = pt.random.normal(name="base_rv")
    vv = pt.scalar("vv")

    transform_func, ref_func = TRANSFORMATIONS[transform]
    transformed_rv = transform_func(base_rv)
    ref_transformed_rv = ref_func(base_rv)

    logp_test = logp(transformed_rv, vv)
    logp_ref = logp(ref_transformed_rv, vv)

    if transform in ["log2", "log10"]:
        # in the cases of log2 and log10 floating point inprecision causes failure
        # from equal_computations so evaluate logp and check all close instead
        vv_test = np.array(0.25)
        np.testing.assert_allclose(logp_ref.eval({vv: vv_test}), logp_test.eval({vv: vv_test}))
    else:
        assert equal_computations([logp_test], [logp_ref])


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


def test_discrete_rv_unary_transform_fails():
    y_rv = pt.exp(pt.random.poisson(1))
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({y_rv: y_rv.clone()})


def test_discrete_rv_multinary_transform_fails():
    y_rv = 5 + pt.random.poisson(1)
    with pytest.raises(RuntimeError, match="could not be derived"):
        conditional_logp({y_rv: y_rv.clone()})


@pytest.mark.xfail(reason="Check not implemented yet")
def test_invalid_broadcasted_transform_rv_fails():
    loc = pt.vector("loc")
    y_rv = loc + pt.random.normal(0, 1, size=1, name="base_rv")
    y_rv.name = "y"
    y_vv = y_rv.clone()

    # This logp derivation should fail or count only once the values that are broadcasted
    logprob = logp(y_rv, y_vv)
    assert logprob.eval({y_vv: [0, 0, 0, 0], loc: [0, 0, 0, 0]}).shape == ()
