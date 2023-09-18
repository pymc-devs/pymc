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


from typing import Union

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.tensor.variable import TensorConstant

import pymc as pm
import pymc.distributions.transforms as tr

from pymc.logprob.basic import transformed_conditional_logp
from pymc.logprob.transforms import RVTransform
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
from tests.checks import close_to, close_to_logical

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
        close_to(val, identity_f(val), tol)


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
        np.testing.assert_allclose(actual_ljd(yval), computed_ljd(yval), rtol=tol)


def test_simplex():
    check_vector_transform(tr.simplex, Simplex(2))
    check_vector_transform(tr.simplex, Simplex(4))

    check_transform(
        tr.simplex, MultiSimplex(3, 2), constructor=pt.matrix, test=floatX(np.zeros((2, 2)))
    )


def test_simplex_bounds():
    vals = get_values(tr.simplex, Vector(R, 2), pt.vector, floatX(np.array([0, 0])))

    close_to(vals.sum(axis=1), 1, tol)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)

    check_jacobian_det(
        tr.simplex, Vector(R, 2), pt.vector, floatX(np.array([0, 0])), lambda x: x[:-1]
    )


def test_simplex_accuracy():
    val = floatX(np.array([-30]))
    x = pt.vector("x")
    x.tag.test_value = val
    identity_f = pytensor.function([x], tr.simplex.forward(x, tr.simplex.backward(x, x)))
    close_to(val, identity_f(val), tol)


def test_sum_to_1():
    check_vector_transform(tr.sum_to_1, Simplex(2))
    check_vector_transform(tr.sum_to_1, Simplex(4))

    with pytest.warns(FutureWarning, match="ndim_supp argument is deprecated"):
        tr.SumTo1(2)

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
    close_to_logical(vals > 0, True, tol)


@pytest.mark.skipif(
    pytensor.config.floatX == "float32", reason="Test is designed for 64bit precision"
)
def test_log_exp_m1():
    check_transform(tr.log_exp_m1, Rplusbig)

    check_jacobian_det(tr.log_exp_m1, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log_exp_m1, Vector(Rplusbig, 2), pt.vector, [0, 0], elemwise=True)

    vals = get_values(tr.log_exp_m1)
    close_to_logical(vals > 0, True, tol)


def test_logodds():
    check_transform(tr.logodds, Unit)

    check_jacobian_det(tr.logodds, Unit, elemwise=True)
    check_jacobian_det(tr.logodds, Vector(Unit, 2), pt.vector, [0.5, 0.5], elemwise=True)

    vals = get_values(tr.logodds)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)


def test_lowerbound():
    trans = tr.Interval(0.0, None)
    check_transform(trans, Rplusbig)

    check_jacobian_det(trans, Rplusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rplusbig, 2), pt.vector, [0, 0], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals > 0, True, tol)


def test_upperbound():
    trans = tr.Interval(None, 0.0)
    check_transform(trans, Rminusbig)

    check_jacobian_det(trans, Rminusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rminusbig, 2), pt.vector, [-1, -1], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals < 0, True, tol)


def test_interval():
    for a, b in [(-4, 5.5), (0.1, 0.7), (-10, 4.3)]:
        domain = Unit * np.float64(b - a) + np.float64(a)

        trans = tr.Interval(a, b)
        check_transform(trans, domain)

        check_jacobian_det(trans, domain, elemwise=True)

        vals = get_values(trans)
        close_to_logical(vals > a, True, tol)
        close_to_logical(vals < b, True, tol)


@pytest.mark.skipif(
    pytensor.config.floatX == "float32", reason="Test is designed for 64bit precision"
)
def test_interval_near_boundary():
    lb = -1.0
    ub = 1e-7
    x0 = np.nextafter(ub, lb)

    with pm.Model() as model:
        pm.Uniform("x", initval=x0, lower=lb, upper=ub)

    log_prob = model.point_logps()
    np.testing.assert_allclose(list(log_prob.values()), floatX(np.array([-52.68])))


def test_circular():
    trans = tr.circular
    check_transform(trans, Circ)

    check_jacobian_det(trans, Circ)

    vals = get_values(trans)
    close_to_logical(vals > -np.pi, True, tol)
    close_to_logical(vals < np.pi, True, tol)

    assert isinstance(trans.forward(1, None), TensorConstant)


def test_ordered():
    check_vector_transform(tr.ordered, SortedVector(6))

    with pytest.warns(FutureWarning, match="ndim_supp argument is deprecated"):
        tr.Ordered(1)

    check_jacobian_det(
        tr.ordered, Vector(R, 2), pt.vector, floatX(np.array([0, 0])), elemwise=False
    )

    vals = get_values(tr.ordered, Vector(R, 3), pt.vector, floatX(np.zeros(3)))
    close_to_logical(np.diff(vals) >= 0, True, tol)


def test_chain_values():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    vals = get_values(chain_tranf, Vector(R, 5), pt.vector, floatX(np.zeros(5)))
    close_to_logical(np.diff(vals) >= 0, True, tol)


def test_chain_vector_transform():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    check_vector_transform(chain_tranf, UnitSortedVector(3))


@pytest.mark.xfail(reason="Fails due to precision issue. Values just close to expected.")
def test_chain_jacob_det():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    check_jacobian_det(chain_tranf, Vector(R, 4), pt.vector, floatX(np.zeros(4)), elemwise=False)


class TestElementWiseLogp:
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
        np.testing.assert_allclose(
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


def test_triangular_transform():
    with pm.Model() as m:
        x = pm.Triangular("x", lower=0, c=1, upper=2)

    transform = m.rvs_to_transforms[x]
    assert np.isclose(transform.backward(-np.inf, *x.owner.inputs).eval(), 0)
    assert np.isclose(transform.backward(np.inf, *x.owner.inputs).eval(), 2)


def test_interval_transform_raises():
    with pytest.raises(ValueError, match="Lower and upper interval bounds cannot both be None"):
        tr.Interval(None, None)

    with pytest.raises(ValueError, match="Interval bounds must be constant values"):
        tr.Interval(pt.constant(5) + 1, None)

    assert tr.Interval(pt.constant(5), None)


def test_discrete_trafo():
    with pm.Model():
        with pytest.raises(ValueError) as err:
            pm.Binomial("a", n=5, p=0.5, transform="log")
        err.match("Transformations for discrete distributions")


def test_transform_univariate_dist_logp_shape():
    with pm.Model() as m:
        pm.Uniform("x", shape=(4, 3), transform=tr.logodds)

    assert m.logp(jacobian=False, sum=False)[0].type.shape == (4, 3)
    assert m.logp(jacobian=True, sum=False)[0].type.shape == (4, 3)

    with pm.Model() as m:
        pm.Uniform("x", shape=(4, 3), transform=tr.ordered)

    assert m.logp(jacobian=False, sum=False)[0].type.shape == (4,)
    assert m.logp(jacobian=True, sum=False)[0].type.shape == (4,)


def test_univariate_transform_multivariate_dist_raises():
    with pm.Model() as m:
        pm.Dirichlet("x", [1, 1, 1], transform=tr.log)

    for jacobian in (True, False):
        with pytest.raises(
            NotImplementedError,
            match="Univariate transform LogTransform cannot be applied to multivariate",
        ):
            m.logp(jacobian=jacobian)


def test_invalid_jacobian_broadcast_raises():
    class BuggyTransform(RVTransform):
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


def test_deprecated_ndim_supp_transforms():
    with pytest.warns(FutureWarning, match="deprecated"):
        tr.Ordered(ndim_supp=1)

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.univariate_ordered == tr.ordered

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.multivariate_ordered == tr.ordered

    with pytest.warns(FutureWarning, match="deprecated"):
        tr.SumTo1(ndim_supp=1)

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.univariate_sum_to_1 == tr.sum_to_1

    with pytest.warns(FutureWarning, match="deprecated"):
        assert tr.multivariate_sum_to_1 == tr.sum_to_1
