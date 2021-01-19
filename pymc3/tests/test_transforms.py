#   Copyright 2020 The PyMC Developers
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
import pytest
import theano
import theano.tensor as tt

import pymc3 as pm
import pymc3.distributions.transforms as tr

from pymc3.tests.checks import close_to, close_to_logical
from pymc3.tests.helpers import SeededTest
from pymc3.tests.test_distributions import (
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
from pymc3.theanof import jacobian

# some transforms (stick breaking) require additon of small slack in order to be numerically
# stable. The minimal addable slack for float32 is higher thus we need to be less strict
tol = 1e-7 if theano.config.floatX == "float64" else 1e-6


def check_transform(transform, domain, constructor=tt.dscalar, test=0):
    x = constructor("x")
    x.tag.test_value = test
    # test forward and forward_val
    forward_f = theano.function([x], transform.forward(x))
    # test transform identity
    identity_f = theano.function([x], transform.backward(transform.forward(x)))
    for val in domain.vals:
        close_to(val, identity_f(val), tol)
        close_to(transform.forward_val(val), forward_f(val), tol)


def check_vector_transform(transform, domain):
    return check_transform(transform, domain, tt.dvector, test=np.array([0, 0]))


def get_values(transform, domain=R, constructor=tt.dscalar, test=0):
    x = constructor("x")
    x.tag.test_value = test
    f = theano.function([x], transform.backward(x))
    return np.array([f(val) for val in domain.vals])


def check_jacobian_det(
    transform, domain, constructor=tt.dscalar, test=0, make_comparable=None, elemwise=False
):
    y = constructor("y")
    y.tag.test_value = test

    x = transform.backward(y)
    if make_comparable:
        x = make_comparable(x)

    if not elemwise:
        jac = tt.log(tt.nlinalg.det(jacobian(x, [y])))
    else:
        jac = tt.log(tt.abs_(tt.diag(jacobian(x, [y]))))

    # ljd = log jacobian det
    actual_ljd = theano.function([y], jac)

    computed_ljd = theano.function(
        [y], tt.as_tensor_variable(transform.jacobian_det(y)), on_unused_input="ignore"
    )

    for yval in domain.vals:
        close_to(actual_ljd(yval), computed_ljd(yval), tol)


def test_stickbreaking():
    with pytest.warns(
        DeprecationWarning, match="The argument `eps` is deprecated and will not be used."
    ):
        tr.StickBreaking(eps=1e-9)
    check_vector_transform(tr.stick_breaking, Simplex(2))
    check_vector_transform(tr.stick_breaking, Simplex(4))

    check_transform(
        tr.stick_breaking, MultiSimplex(3, 2), constructor=tt.dmatrix, test=np.zeros((2, 2))
    )


def test_stickbreaking_bounds():
    vals = get_values(tr.stick_breaking, Vector(R, 2), tt.dvector, np.array([0, 0]))

    close_to(vals.sum(axis=1), 1, tol)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)

    check_jacobian_det(
        tr.stick_breaking, Vector(R, 2), tt.dvector, np.array([0, 0]), lambda x: x[:-1]
    )


def test_stickbreaking_accuracy():
    val = np.array([-30])
    x = tt.dvector("x")
    x.tag.test_value = val
    identity_f = theano.function([x], tr.stick_breaking.forward(tr.stick_breaking.backward(x)))
    close_to(val, identity_f(val), tol)


def test_sum_to_1():
    check_vector_transform(tr.sum_to_1, Simplex(2))
    check_vector_transform(tr.sum_to_1, Simplex(4))

    check_jacobian_det(tr.sum_to_1, Vector(Unit, 2), tt.dvector, np.array([0, 0]), lambda x: x[:-1])


def test_log():
    check_transform(tr.log, Rplusbig)

    check_jacobian_det(tr.log, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log, Vector(Rplusbig, 2), tt.dvector, [0, 0], elemwise=True)

    vals = get_values(tr.log)
    close_to_logical(vals > 0, True, tol)


def test_log_exp_m1():
    check_transform(tr.log_exp_m1, Rplusbig)

    check_jacobian_det(tr.log_exp_m1, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log_exp_m1, Vector(Rplusbig, 2), tt.dvector, [0, 0], elemwise=True)

    vals = get_values(tr.log_exp_m1)
    close_to_logical(vals > 0, True, tol)


def test_logodds():
    check_transform(tr.logodds, Unit)

    check_jacobian_det(tr.logodds, Unit, elemwise=True)
    check_jacobian_det(tr.logodds, Vector(Unit, 2), tt.dvector, [0.5, 0.5], elemwise=True)

    vals = get_values(tr.logodds)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)


def test_lowerbound():
    trans = tr.lowerbound(0.0)
    check_transform(trans, Rplusbig)

    check_jacobian_det(trans, Rplusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rplusbig, 2), tt.dvector, [0, 0], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals > 0, True, tol)


def test_upperbound():
    trans = tr.upperbound(0.0)
    check_transform(trans, Rminusbig)

    check_jacobian_det(trans, Rminusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rminusbig, 2), tt.dvector, [-1, -1], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals < 0, True, tol)


def test_interval():
    for a, b in [(-4, 5.5), (0.1, 0.7), (-10, 4.3)]:
        domain = Unit * np.float64(b - a) + np.float64(a)
        trans = tr.interval(a, b)
        check_transform(trans, domain)

        check_jacobian_det(trans, domain, elemwise=True)

        vals = get_values(trans)
        close_to_logical(vals > a, True, tol)
        close_to_logical(vals < b, True, tol)


@pytest.mark.skipif(theano.config.floatX == "float32", reason="Test fails on 32 bit")
def test_interval_near_boundary():
    lb = -1.0
    ub = 1e-7
    x0 = np.nextafter(ub, lb)

    with pm.Model() as model:
        pm.Uniform("x", testval=x0, lower=lb, upper=ub)

    log_prob = model.check_test_point()
    np.testing.assert_allclose(log_prob.values, np.array([-52.68]))


def test_circular():
    trans = tr.circular
    check_transform(trans, Circ)

    check_jacobian_det(trans, Circ)

    vals = get_values(trans)
    close_to_logical(vals > -np.pi, True, tol)
    close_to_logical(vals < np.pi, True, tol)

    assert isinstance(trans.forward(1), tt.TensorConstant)


def test_ordered():
    check_vector_transform(tr.ordered, SortedVector(6))

    check_jacobian_det(tr.ordered, Vector(R, 2), tt.dvector, np.array([0, 0]), elemwise=False)

    vals = get_values(tr.ordered, Vector(R, 3), tt.dvector, np.zeros(3))
    close_to_logical(np.diff(vals) >= 0, True, tol)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
def test_chain():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    check_vector_transform(chain_tranf, UnitSortedVector(3))

    check_jacobian_det(chain_tranf, Vector(R, 4), tt.dvector, np.zeros(4), elemwise=False)

    vals = get_values(chain_tranf, Vector(R, 5), tt.dvector, np.zeros(5))
    close_to_logical(np.diff(vals) >= 0, True, tol)


class TestElementWiseLogp(SeededTest):
    def build_model(self, distfam, params, shape, transform, testval=None):
        if testval is not None:
            testval = pm.floatX(testval)
        with pm.Model() as m:
            distfam("x", shape=shape, transform=transform, testval=testval, **params)
        return m

    def check_transform_elementwise_logp(self, model):
        x0 = model.deterministics[0]
        x = model.free_RVs[0]
        assert x.ndim == x.logp_elemwiset.ndim

        pt = model.test_point
        array = np.random.randn(*pt[x.name].shape)
        pt[x.name] = array
        dist = x.distribution
        logp_nojac = x0.distribution.logp(dist.transform_used.backward(array))
        jacob_det = dist.transform_used.jacobian_det(theano.shared(array))
        assert x.logp_elemwiset.ndim == jacob_det.ndim

        elementwiselogp = logp_nojac + jacob_det

        close_to(x.logp_elemwise(pt), elementwiselogp.eval(), tol)

    def check_vectortransform_elementwise_logp(self, model, vect_opt=0):
        x0 = model.deterministics[0]
        x = model.free_RVs[0]
        assert (x.ndim - 1) == x.logp_elemwiset.ndim

        pt = model.test_point
        array = np.random.randn(*pt[x.name].shape)
        pt[x.name] = array
        dist = x.distribution
        logp_nojac = x0.distribution.logp(dist.transform_used.backward(array))
        jacob_det = dist.transform_used.jacobian_det(theano.shared(array))
        assert x.logp_elemwiset.ndim == jacob_det.ndim

        if vect_opt == 0:
            # the original distribution is univariate
            elementwiselogp = logp_nojac.sum(axis=-1) + jacob_det
        else:
            elementwiselogp = logp_nojac + jacob_det
        # Hack to get relative tolerance
        a = x.logp_elemwise(pt)
        b = elementwiselogp.eval()
        close_to(a, b, np.abs(0.5 * (a + b) * tol))

    @pytest.mark.parametrize(
        "sd,shape",
        [
            (2.5, 2),
            (5.0, (2, 3)),
            (np.ones(3) * 10.0, (4, 3)),
        ],
    )
    def test_half_normal(self, sd, shape):
        model = self.build_model(pm.HalfNormal, {"sd": sd}, shape=shape, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize("lam,shape", [(2.5, 2), (5.0, (2, 3)), (np.ones(3), (4, 3))])
    def test_exponential(self, lam, shape):
        model = self.build_model(pm.Exponential, {"lam": lam}, shape=shape, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,b,shape",
        [
            (1.0, 1.0, 2),
            (0.5, 0.5, (2, 3)),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta(self, a, b, shape):
        model = self.build_model(
            pm.Beta, {"alpha": a, "beta": b}, shape=shape, transform=tr.logodds
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower,upper,shape",
        [
            (0.0, 1.0, 2),
            (0.5, 5.5, (2, 3)),
            (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3)),
        ],
    )
    def test_uniform(self, lower, upper, shape):
        interval = tr.Interval(lower, upper)
        model = self.build_model(
            pm.Uniform, {"lower": lower, "upper": upper}, shape=shape, transform=interval
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "mu,kappa,shape", [(0.0, 1.0, 2), (-0.5, 5.5, (2, 3)), (np.zeros(3), np.ones(3), (4, 3))]
    )
    def test_vonmises(self, mu, kappa, shape):
        model = self.build_model(
            pm.VonMises, {"mu": mu, "kappa": kappa}, shape=shape, transform=tr.circular
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,shape", [(np.ones(2), 2), (np.ones((2, 3)) * 0.5, (2, 3)), (np.ones(3), (4, 3))]
    )
    def test_dirichlet(self, a, shape):
        model = self.build_model(pm.Dirichlet, {"a": a}, shape=shape, transform=tr.stick_breaking)
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)

    def test_normal_ordered(self):
        model = self.build_model(
            pm.Normal,
            {"mu": 0.0, "sd": 1.0},
            shape=3,
            testval=np.asarray([-1.0, 1.0, 4.0]),
            transform=tr.ordered,
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "sd,shape",
        [
            (2.5, (2,)),
            (np.ones(3), (4, 3)),
        ],
    )
    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_half_normal_ordered(self, sd, shape):
        testval = np.sort(np.abs(np.random.randn(*shape)))
        model = self.build_model(
            pm.HalfNormal,
            {"sd": sd},
            shape=shape,
            testval=testval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize("lam,shape", [(2.5, (2,)), (np.ones(3), (4, 3))])
    def test_exponential_ordered(self, lam, shape):
        testval = np.sort(np.abs(np.random.randn(*shape)))
        model = self.build_model(
            pm.Exponential,
            {"lam": lam},
            shape=shape,
            testval=testval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "a,b,shape",
        [
            (1.0, 1.0, (2,)),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta_ordered(self, a, b, shape):
        testval = np.sort(np.abs(np.random.rand(*shape)))
        model = self.build_model(
            pm.Beta,
            {"alpha": a, "beta": b},
            shape=shape,
            testval=testval,
            transform=tr.Chain([tr.logodds, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "lower,upper,shape",
        [(0.0, 1.0, (2,)), (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3))],
    )
    def test_uniform_ordered(self, lower, upper, shape):
        interval = tr.Interval(lower, upper)
        testval = np.sort(np.abs(np.random.rand(*shape)))
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            shape=shape,
            testval=testval,
            transform=tr.Chain([interval, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "mu,kappa,shape", [(0.0, 1.0, (2,)), (np.zeros(3), np.ones(3), (4, 3))]
    )
    def test_vonmises_ordered(self, mu, kappa, shape):
        testval = np.sort(np.abs(np.random.rand(*shape)))
        model = self.build_model(
            pm.VonMises,
            {"mu": mu, "kappa": kappa},
            shape=shape,
            testval=testval,
            transform=tr.Chain([tr.circular, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "lower,upper,shape,transform",
        [
            (0.0, 1.0, (2,), tr.stick_breaking),
            (0.5, 5.5, (2, 3), tr.stick_breaking),
            (np.zeros(3), np.ones(3), (4, 3), tr.Chain([tr.sum_to_1, tr.logodds])),
        ],
    )
    def test_uniform_other(self, lower, upper, shape, transform):
        testval = np.ones(shape) / shape[-1]
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            shape=shape,
            testval=testval,
            transform=transform,
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "mu,cov,shape",
        [
            (np.zeros(2), np.diag(np.ones(2)), (2,)),
            (np.zeros(3), np.diag(np.ones(3)), (4, 3)),
        ],
    )
    def test_mvnormal_ordered(self, mu, cov, shape):
        testval = np.sort(np.random.randn(*shape))
        model = self.build_model(
            pm.MvNormal, {"mu": mu, "cov": cov}, shape=shape, testval=testval, transform=tr.ordered
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)
