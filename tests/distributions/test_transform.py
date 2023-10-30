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


import numpy as np
import pytensor
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from pytensor import tensor as pt
from pytensor.tensor.variable import TensorConstant

import pymc as pm
import pymc.distributions.transforms as tr

from pymc.distributions.transforms import (
    ArccoshTransform,
    ArcsinhTransform,
    ArctanhTransform,
    ChainTransform,
    CoshTransform,
    ErfcTransform,
    ErfcxTransform,
    ErfTransform,
    ExpTransform,
    IntervalTransform,
    LocTransform,
    LogTransform,
    ScaleTransform,
    SinhTransform,
    TanhTransform,
    Transform,
)
from pymc.logprob.basic import transformed_conditional_logp
from pymc.pytensorf import floatX, jacobian
from pymc.testing import (
    Circ,
    MultiSimplex,
    R,
    Rminusbig,
    Rplusbig,
    Simplex,
    SortedVector,
    Unit,
    UnitSortedVector,
    Vector,
)

# some transforms (stick breaking) require addition of small slack in order to be numerically
# stable. The minimal addable slack for float32 is higher thus we need to be less strict
tol = 1e-7 if pytensor.config.floatX == "float64" else 1e-5


def check_transform(transform, domain, constructor=pt.scalar, test=0, rv_var=None):
    x = constructor("x")
    x.tag.test_value = test
    if rv_var is None:
        rv_var = x
    rv_inputs = rv_var.owner.inputs if rv_var.owner else []
    # test forward and forward_val
    # FIXME: What's being tested here?  That the transformed graph can compile?
    forward_f = pytensor.function([x], transform.forward(x, *rv_inputs))
    # test transform identity
    z = transform.backward(transform.forward(x, *rv_inputs))
    assert z.type == x.type
    identity_f = pytensor.function([x], z, *rv_inputs)
    for val in domain.vals:
        assert_allclose(val, identity_f(val), atol=tol)


def check_vector_transform(transform, domain, rv_var=None):
    return check_transform(
        transform, domain, pt.vector, test=floatX(np.array([0, 0])), rv_var=rv_var
    )


def get_values(transform, domain=R, constructor=pt.scalar, test=0, rv_var=None):
    x = constructor("x")
    x.tag.test_value = test
    if rv_var is None:
        rv_var = x
    rv_inputs = rv_var.owner.inputs if rv_var.owner else []
    f = pytensor.function([x], transform.backward(x, *rv_inputs))
    return np.array([f(val) for val in domain.vals])


def check_jacobian_det(
    transform,
    domain,
    constructor=pt.scalar,
    test=0,
    make_comparable=None,
    elemwise=False,
    rv_var=None,
):
    y = constructor("y")
    y.tag.test_value = test

    if rv_var is None:
        rv_var = y

    rv_inputs = rv_var.owner.inputs if rv_var.owner else []

    x = transform.backward(y, *rv_inputs)
    # Assume non-injective transforms are symmetric around the origin
    if isinstance(x, tuple):
        x = x[-1]
    if make_comparable:
        x = make_comparable(x)

    if not elemwise:
        jac = pt.log(pt.nlinalg.det(jacobian(x, [y])))
    else:
        jac = pt.log(pt.abs(pt.diag(jacobian(x, [y]))))

    # ljd = log jacobian det
    actual_ljd = pytensor.function([y], jac)

    computed_ljd = pytensor.function(
        [y], pt.as_tensor_variable(transform.log_jac_det(y, *rv_inputs)), on_unused_input="ignore"
    )

    for yval in domain.vals:
        assert_allclose(actual_ljd(yval), computed_ljd(yval), rtol=tol)


class TestTransformBase:
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

        value = pt.vector("value", dtype="float64")
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


class TestInvalidTransform:
    def test_discrete_trafo(self):
        with pm.Model():
            with pytest.raises(ValueError) as err:
                pm.Binomial("a", n=5, p=0.5, transform="log")
            err.match("Transformations for discrete distributions")

    def test_univariate_transform_multivariate_dist_raises(self):
        with pm.Model() as m:
            pm.Dirichlet("x", [1, 1, 1], transform=tr.log)

        for jacobian in (True, False):
            with pytest.raises(
                NotImplementedError,
                match="Univariate transform LogTransform cannot be applied to multivariate",
            ):
                m.logp(jacobian=jacobian)

    def test_invalid_jacobian_broadcast_raises(self):
        class BuggyTransform(Transform):
            name = "buggy"

            def forward(self, value, *inputs):
                return value

            def backward(self, value, *inputs):
                return value

            def log_jac_det(self, value, *inputs):
                return pt.zeros_like(value.sum(-1, keepdims=True))

        buggy_transform = BuggyTransform()

        with pm.Model() as m:
            pm.Uniform("x", shape=(4, 3), transform=buggy_transform)

        for jacobian in (True, False):
            with pytest.raises(
                ValueError,
                match="are not allowed to broadcast together. There is a bug in the implementation of either one",
            ):
                m.logp(jacobian=jacobian)


class TestInterval:
    def test_lowerbound(self):
        trans = tr.Interval(0.0, None)
        check_transform(trans, Rplusbig)

        check_jacobian_det(trans, Rplusbig, elemwise=True)
        check_jacobian_det(trans, Vector(Rplusbig, 2), pt.vector, [0, 0], elemwise=True)

        vals = get_values(trans)
        assert_array_equal(vals > 0, True)

    def test_upperbound(self):
        trans = tr.Interval(None, 0.0)
        check_transform(trans, Rminusbig)

        check_jacobian_det(trans, Rminusbig, elemwise=True)
        check_jacobian_det(trans, Vector(Rminusbig, 2), pt.vector, [-1, -1], elemwise=True)

        vals = get_values(trans)
        assert_array_equal(vals < 0, True)

    def test_interval(self):
        for a, b in [(-4, 5.5), (0.1, 0.7), (-10, 4.3)]:
            domain = Unit * np.float64(b - a) + np.float64(a)

            trans = tr.Interval(a, b)
            check_transform(trans, domain)

            check_jacobian_det(trans, domain, elemwise=True)

            vals = get_values(trans)
            assert_array_equal(vals > a, True)
            assert_array_equal(vals < b, True)

    @pytest.mark.skipif(
        pytensor.config.floatX == "float32", reason="Test is designed for 64bit precision"
    )
    def test_interval_near_boundary(self):
        lb = -1.0
        ub = 1e-7
        x0 = np.nextafter(ub, lb)

        with pm.Model() as model:
            pm.Uniform("x", initval=x0, lower=lb, upper=ub)

        log_prob = model.point_logps()
        assert_allclose(list(log_prob.values()), floatX(np.array([-52.68])))

    def test_invalid_interval_helper(self):
        with pytest.raises(ValueError, match="Lower and upper interval bounds cannot both be None"):
            tr.Interval(None, None)

        with pytest.raises(ValueError, match="Interval bounds must be constant values"):
            tr.Interval(pt.constant(5) + 1, None)

        assert tr.Interval(pt.constant(5), None)

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


class TestSimplex:
    def test_simplex(self):
        check_vector_transform(tr.simplex, Simplex(2))
        check_vector_transform(tr.simplex, Simplex(4))

        check_transform(
            tr.simplex, MultiSimplex(3, 2), constructor=pt.matrix, test=floatX(np.zeros((2, 2)))
        )

    def test_simplex_bounds(self):
        vals = get_values(tr.simplex, Vector(R, 2), pt.vector, floatX(np.array([0, 0])))

        assert_allclose(vals.sum(axis=1), 1, tol)
        assert_array_equal(vals > 0, True)
        assert_array_equal(vals < 1, True)

        check_jacobian_det(
            tr.simplex, Vector(R, 2), pt.vector, floatX(np.array([0, 0])), lambda x: x[:-1]
        )

    def test_simplex_accuracy(self):
        val = floatX(np.array([-30]))
        x = pt.vector("x")
        x.tag.test_value = val
        identity_f = pytensor.function([x], tr.simplex.forward(tr.simplex.backward(x)))
        assert_allclose(val, identity_f(val), tol)


def test_sum_to_1():
    check_vector_transform(tr.sum_to_1, Simplex(2))
    check_vector_transform(tr.sum_to_1, Simplex(4))

    with pytest.warns(FutureWarning, match="ndim_supp argument is deprecated"):
        tr.SumTo1Transform(2)

    check_jacobian_det(
        tr.sum_to_1,
        Vector(Unit, 2),
        pt.vector,
        floatX(np.array([0, 0])),
        lambda x: x[:-1],
    )
    check_jacobian_det(
        tr.multivariate_sum_to_1,
        Vector(Unit, 2),
        pt.vector,
        floatX(np.array([0, 0])),
        lambda x: x[:-1],
    )


def test_log():
    check_transform(tr.log, Rplusbig)

    check_jacobian_det(tr.log, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log, Vector(Rplusbig, 2), pt.vector, [0, 0], elemwise=True)

    vals = get_values(tr.log)
    assert_array_equal(vals > 0, True)


@pytest.mark.skipif(
    pytensor.config.floatX == "float32", reason="Test is designed for 64bit precision"
)
def test_log_exp_m1():
    check_transform(tr.log_exp_m1, Rplusbig)

    check_jacobian_det(tr.log_exp_m1, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log_exp_m1, Vector(Rplusbig, 2), pt.vector, [0, 0], elemwise=True)

    vals = get_values(tr.log_exp_m1)
    assert_array_equal(vals > 0, True)


def test_logodds():
    check_transform(tr.logodds, Unit)

    check_jacobian_det(tr.logodds, Unit, elemwise=True)
    check_jacobian_det(tr.logodds, Vector(Unit, 2), pt.vector, [0.5, 0.5], elemwise=True)

    vals = get_values(tr.logodds)
    assert_array_equal(vals > 0, True)
    assert_array_equal(vals < 1, True)


def test_circular():
    trans = tr.circular
    check_transform(trans, Circ)

    check_jacobian_det(trans, Circ)

    vals = get_values(trans)
    assert_array_equal(vals > -np.pi, True)
    assert_array_equal(vals < np.pi, True)

    assert isinstance(trans.forward(1, None), TensorConstant)


def test_triangular_transform():
    with pm.Model() as m:
        x = pm.Triangular("x", lower=0, c=1, upper=2)

    transform = m.rvs_to_transforms[x]
    assert np.isclose(transform.backward(-np.inf, *x.owner.inputs).eval(), 0)
    assert np.isclose(transform.backward(np.inf, *x.owner.inputs).eval(), 2)


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
def test_check_jac_det(transform):
    check_jacobian_det(
        transform,
        Vector(Rplusbig, 2),
        pt.dvector,
        [0.1, 0.1],
        elemwise=True,
        rv_var=pt.random.normal(0.5, 1, name="base_rv"),
    )


def test_ordered():
    check_vector_transform(tr.ordered, SortedVector(6))

    with pytest.warns(FutureWarning, match="ndim_supp argument is deprecated"):
        tr.OrderedTransform(1)

    check_jacobian_det(
        tr.ordered, Vector(R, 2), pt.vector, floatX(np.array([0, 0])), elemwise=False
    )

    vals = get_values(tr.ordered, Vector(R, 3), pt.vector, floatX(np.zeros(3)))
    assert_array_equal(np.diff(vals) >= 0, True)


class TestChain:
    def test_chain_values(self):
        chain_tranf = tr.Chain([tr.logodds, tr.ordered])
        vals = get_values(chain_tranf, Vector(R, 5), pt.vector, floatX(np.zeros(5)))
        assert_array_equal(np.diff(vals) >= 0, True)

    def test_chain_vector_transform(self):
        chain_tranf = tr.Chain([tr.logodds, tr.ordered])
        check_vector_transform(chain_tranf, UnitSortedVector(3))

    @pytest.mark.xfail(reason="Fails due to precision issue. Values just close to expected.")
    def test_chain_jacob_det(self):
        chain_tranf = tr.Chain([tr.logodds, tr.ordered])
        check_jacobian_det(
            chain_tranf, Vector(R, 4), pt.vector, floatX(np.zeros(4)), elemwise=False
        )

    def test_chained_transform(self):
        loc = 5
        scale = 0.1

        ch = ChainTransform(
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
            rtol=1e-6,
        )

        x_val_backward = ch.backward(x_val_forward, *x.owner.inputs, scale, loc).eval()
        np.testing.assert_allclose(
            x_val_backward,
            x_val,
            rtol=1e-5,
        )

        log_jac_det = ch.log_jac_det(x_val_forward, *x.owner.inputs, scale, loc)
        np.testing.assert_allclose(
            pt.sum(log_jac_det).eval(),
            np.sum(-np.log(scale) - np.log(x_val_forward - loc)),
        )


class TestTransformedRVLogp:
    def build_model(self, distfam, params, size, transform, initval=None):
        if initval is not None:
            initval = pm.floatX(initval)
        with pm.Model() as m:
            distfam("x", size=size, transform=transform, initval=initval, **params)
        return m

    def check_transform_elementwise_logp(self, model, vector_transform=False):
        x = model.free_RVs[0]
        x_val_transf = model.rvs_to_values[x]
        transform = model.rvs_to_transforms[x]
        x_val_untransf = transform.backward(x_val_transf, *x.owner.inputs)

        point = model.initial_point(0)
        test_array_transf = floatX(np.random.randn(*point[x_val_transf.name].shape))
        test_array_untransf = x_val_untransf.eval({x_val_transf: test_array_transf})
        log_jac_det = transform.log_jac_det(x_val_transf, *x.owner.inputs)

        [transform_logp] = transformed_conditional_logp(
            (x,),
            rvs_to_values={x: x_val_transf},
            rvs_to_transforms={x: transform},
        )
        [untransform_logp] = transformed_conditional_logp(
            (x,),
            rvs_to_values={x: x_val_untransf},
            rvs_to_transforms={},
        )
        if vector_transform:
            assert transform_logp.ndim == (x.ndim - 1) == log_jac_det.ndim
        else:
            assert transform_logp.ndim == x.ndim == log_jac_det.ndim

        transform_logp_eval = transform_logp.eval({x_val_transf: test_array_transf})
        untransform_logp_eval = untransform_logp.eval({x_val_untransf: test_array_untransf})
        log_jac_det_eval = log_jac_det.eval({x_val_transf: test_array_transf})
        # Summing the log_jac_det separately from the untransform_logp ensures there is no broadcasting between terms
        assert_allclose(
            transform_logp_eval.sum(),
            untransform_logp_eval.sum() + log_jac_det_eval.sum(),
            rtol=tol,
        )

    def check_vectortransform_elementwise_logp(self, model):
        self.check_transform_elementwise_logp(model, vector_transform=True)

    @pytest.mark.parametrize(
        "sigma,size",
        [
            (2.5, 2),
            (5.0, (2, 3)),
            (np.ones(3) * 10.0, (4, 3)),
        ],
    )
    def test_half_normal(self, sigma, size):
        model = self.build_model(pm.HalfNormal, {"sigma": sigma}, size=size, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize("lam,size", [(2.5, 2), (5.0, (2, 3)), (np.ones(3), (4, 3))])
    def test_exponential(self, lam, size):
        model = self.build_model(pm.Exponential, {"lam": lam}, size=size, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,b,size",
        [
            (1.0, 1.0, 2),
            (0.5, 0.5, (2, 3)),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta(self, a, b, size):
        model = self.build_model(pm.Beta, {"alpha": a, "beta": b}, size=size, transform=tr.logodds)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower,upper,size",
        [
            (0.0, 1.0, 2),
            (0.5, 5.5, (2, 3)),
            (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3)),
        ],
    )
    def test_uniform(self, lower, upper, size):
        def transform_params(*inputs):
            _, _, _, lower, upper = inputs
            lower = pt.as_tensor_variable(lower) if lower is not None else None
            upper = pt.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(bounds_fn=transform_params)
        model = self.build_model(
            pm.Uniform, {"lower": lower, "upper": upper}, size=size, transform=interval
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower, c, upper, size",
        [
            (0.0, 1.0, 2.0, 2),
            (-10, 0, 200, (2, 3)),
            (floatX(np.zeros(3)), floatX(np.ones(3)), floatX(np.ones(3)), (4, 3)),
        ],
    )
    def test_triangular(self, lower, c, upper, size):
        def transform_params(*inputs):
            _, _, _, lower, _, upper = inputs
            lower = pt.as_tensor_variable(lower) if lower is not None else None
            upper = pt.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(bounds_fn=transform_params)
        model = self.build_model(
            pm.Triangular, {"lower": lower, "c": c, "upper": upper}, size=size, transform=interval
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "mu,kappa,size",
        [(0.0, 1.0, 2), (-0.5, 5.5, (2, 3)), (floatX(np.zeros(3)), floatX(np.ones(3)), (4, 3))],
    )
    def test_vonmises(self, mu, kappa, size):
        model = self.build_model(
            pm.VonMises, {"mu": mu, "kappa": kappa}, size=size, transform=tr.circular
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,size", [(np.ones(2), None), (np.ones((2, 3)) * 0.5, None), (np.ones(3), (4,))]
    )
    def test_dirichlet(self, a, size):
        model = self.build_model(pm.Dirichlet, {"a": a}, size=size, transform=tr.simplex)
        self.check_vectortransform_elementwise_logp(model)

    def test_normal_ordered(self):
        model = self.build_model(
            pm.Normal,
            {"mu": 0.0, "sigma": 1.0},
            size=3,
            initval=np.asarray([-1.0, 1.0, 4.0]),
            transform=tr.ordered,
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "sigma,size",
        [
            (2.5, (2,)),
            (np.ones(3), (4, 3)),
        ],
    )
    def test_half_normal_ordered(self, sigma, size):
        initval = np.sort(np.abs(np.random.randn(*size)))
        model = self.build_model(
            pm.HalfNormal,
            {"sigma": sigma},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize("lam,size", [(2.5, (2,)), (np.ones(3), (4, 3))])
    def test_exponential_ordered(self, lam, size):
        initval = np.sort(np.abs(np.random.randn(*size)))
        model = self.build_model(
            pm.Exponential,
            {"lam": lam},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,b,size",
        [
            (
                1.0,
                1.0,
                (2,),
            ),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta_ordered(self, a, b, size):
        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.Beta,
            {"alpha": a, "beta": b},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.logodds, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower,upper,size",
        [(0.0, 1.0, (2,)), (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3))],
    )
    def test_uniform_ordered(self, lower, upper, size):
        def transform_params(*inputs):
            _, _, _, lower, upper = inputs
            lower = pt.as_tensor_variable(lower) if lower is not None else None
            upper = pt.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(bounds_fn=transform_params)

        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            size=size,
            initval=initval,
            transform=tr.Chain([interval, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "mu,kappa,size", [(0.0, 1.0, (2,)), (floatX(np.zeros(3)), floatX(np.ones(3)), (4, 3))]
    )
    def test_vonmises_ordered(self, mu, kappa, size):
        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.VonMises,
            {"mu": mu, "kappa": kappa},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.circular, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower,upper,size,transform",
        [
            (0.0, 1.0, (2,), tr.simplex),
            (0.5, 5.5, (2, 3), tr.simplex),
            (
                floatX(np.zeros(3)),
                floatX(np.ones(3)),
                (4, 3),
                tr.Chain([tr.sum_to_1, tr.logodds]),
            ),
        ],
    )
    def test_uniform_other(self, lower, upper, size, transform):
        initval = np.ones(size) / size[-1]
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            size=size,
            initval=initval,
            transform=transform,
        )
        self.check_vectortransform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "mu,cov,size,shape",
        [
            (floatX(np.zeros(2)), floatX(np.diag(np.ones(2))), None, (2,)),
            (floatX(np.zeros(3)), floatX(np.diag(np.ones(3))), (4,), (4, 3)),
        ],
    )
    @pytest.mark.parametrize("transform", (tr.ordered, tr.sum_to_1))
    def test_mvnormal_transform(self, mu, cov, size, shape, transform):
        initval = np.sort(np.random.randn(*shape))
        model = self.build_model(
            pm.MvNormal,
            {"mu": mu, "cov": cov},
            size=size,
            initval=initval,
            transform=transform,
        )
        self.check_vectortransform_elementwise_logp(model)

    def test_transform_univariate_dist_logp_shape(self):
        with pm.Model() as m:
            pm.Uniform("x", shape=(4, 3), transform=tr.logodds)

        assert m.logp(jacobian=False, sum=False)[0].type.shape == (4, 3)
        assert m.logp(jacobian=True, sum=False)[0].type.shape == (4, 3)

        with pm.Model() as m:
            pm.Uniform("x", shape=(4, 3), transform=tr.ordered)

        assert m.logp(jacobian=False, sum=False)[0].type.shape == (4,)
        assert m.logp(jacobian=True, sum=False)[0].type.shape == (4,)


def test_deprecated_ndim_supp_transforms():
    with pytest.warns(FutureWarning, match="deprecated"):
        tr.OrderedTransform(ndim_supp=1)

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.univariate_ordered == tr.ordered

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.multivariate_ordered == tr.ordered

    with pytest.warns(FutureWarning, match="deprecated"):
        tr.SumTo1Transform(ndim_supp=1)

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.univariate_sum_to_1 == tr.sum_to_1

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.multivariate_sum_to_1 == tr.sum_to_1
