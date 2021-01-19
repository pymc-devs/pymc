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

import functools
import io
import operator

import numpy as np
import pytest
import theano
import theano.tensor as tt

import pymc3 as pm
import pymc3.memoize
import pymc3.util

from pymc3.tests import models
from pymc3.tests.helpers import not_raises
from pymc3.theanof import intX
from pymc3.variational import flows, opvi
from pymc3.variational.approximations import (
    Empirical,
    EmpiricalGroup,
    FullRank,
    FullRankGroup,
    MeanField,
    MeanFieldGroup,
    NormalizingFlow,
    NormalizingFlowGroup,
)
from pymc3.variational.inference import ADVI, ASVGD, NFVI, SVGD, FullRankADVI, fit
from pymc3.variational.opvi import Approximation, Group

pytestmark = pytest.mark.usefixtures("strict_float32", "seeded_test")


@pytest.mark.parametrize("diff", ["relative", "absolute"])
@pytest.mark.parametrize("ord", [1, 2, np.inf])
def test_callbacks_convergence(diff, ord):
    cb = pm.variational.callbacks.CheckParametersConvergence(every=1, diff=diff, ord=ord)

    class _approx:
        params = (theano.shared(np.asarray([1, 2, 3])),)

    approx = _approx()

    with pytest.raises(StopIteration):
        cb(approx, None, 1)
        cb(approx, None, 10)


def test_tracker_callback():
    import time

    tracker = pm.callbacks.Tracker(
        ints=lambda *t: t[-1],
        ints2=lambda ap, h, j: j,
        time=time.time,
    )
    for i in range(10):
        tracker(None, None, i)
    assert "time" in tracker.hist
    assert "ints" in tracker.hist
    assert "ints2" in tracker.hist
    assert len(tracker["ints"]) == len(tracker["ints2"]) == len(tracker["time"]) == 10
    assert tracker["ints"] == tracker["ints2"] == list(range(10))
    tracker = pm.callbacks.Tracker(bad=lambda t: t)  # bad signature
    with pytest.raises(TypeError):
        tracker(None, None, 1)


@pytest.fixture(scope="module")
def three_var_model():
    with pm.Model() as model:
        pm.HalfNormal("one", shape=(10, 2), total_size=100)
        pm.Normal("two", shape=(10,))
        pm.Normal("three", shape=(10, 1, 2))
    return model


@pytest.mark.parametrize(
    ["raises", "grouping"],
    [
        (not_raises(), {MeanFieldGroup: None}),
        (not_raises(), {FullRankGroup: None, MeanFieldGroup: ["one"]}),
        (
            not_raises(),
            {MeanFieldGroup: ["one"], FullRankGroup: ["two"], NormalizingFlowGroup: ["three"]},
        ),
        (
            pytest.raises(TypeError, match="Found duplicates"),
            {
                MeanFieldGroup: ["one"],
                FullRankGroup: ["two", "one"],
                NormalizingFlowGroup: ["three"],
            },
        ),
        (
            pytest.raises(TypeError, match="No approximation is specified"),
            {MeanFieldGroup: ["one", "two"]},
        ),
        (not_raises(), {MeanFieldGroup: ["one"], FullRankGroup: ["two", "three"]}),
    ],
)
def test_init_groups(three_var_model, raises, grouping):
    with raises, three_var_model:
        approxes, groups = zip(*grouping.items())
        groups = [
            list(map(functools.partial(getattr, three_var_model), g)) if g is not None else None
            for g in groups
        ]
        inited_groups = [a(group=g) for a, g in zip(approxes, groups)]
        approx = Approximation(inited_groups)
        for ig, g in zip(inited_groups, groups):
            if g is None:
                pass
            else:
                assert {pm.util.get_transformed(z) for z in g} == set(ig.group)
        else:
            assert approx.ndim == three_var_model.ndim


@pytest.fixture(
    params=[
        ({}, {MeanFieldGroup: (None, {})}),
        ({}, {FullRankGroup: (None, {}), MeanFieldGroup: (["one"], {})}),
        (
            {},
            {
                MeanFieldGroup: (["one"], {}),
                FullRankGroup: (["two"], {}),
                NormalizingFlowGroup: (["three"], {"flow": "scale-hh*2-planar-radial-loc"}),
            },
        ),
        ({}, {MeanFieldGroup: (["one"], {}), FullRankGroup: (["two", "three"], {})}),
        ({}, {MeanFieldGroup: (["one"], {}), EmpiricalGroup: (["two", "three"], {"size": 100})}),
    ],
    ids=lambda t: ", ".join("{}: {}".format(k.__name__, v[0]) for k, v in t[1].items()),
)
def three_var_groups(request, three_var_model):
    kw, grouping = request.param
    approxes, groups = zip(*grouping.items())
    groups, gkwargs = zip(*groups)
    groups = [
        list(map(functools.partial(getattr, three_var_model), g)) if g is not None else None
        for g in groups
    ]
    inited_groups = [
        a(group=g, model=three_var_model, **gk) for a, g, gk in zip(approxes, groups, gkwargs)
    ]
    return inited_groups


@pytest.fixture
def three_var_approx(three_var_model, three_var_groups):
    approx = Approximation(three_var_groups, model=three_var_model)
    return approx


@pytest.fixture
def three_var_approx_single_group_mf(three_var_model):
    return MeanField(model=three_var_model)


@pytest.fixture
def test_sample_simple(three_var_approx, request):
    backend, name = request.param
    trace = three_var_approx.sample(100, name=name)
    assert set(trace.varnames) == {"one", "one_log__", "three", "two"}
    assert len(trace) == 100
    assert trace[0]["one"].shape == (10, 2)
    assert trace[0]["two"].shape == (10,)
    assert trace[0]["three"].shape == (10, 1, 2)


@pytest.fixture
def aevb_initial():
    return theano.shared(np.random.rand(3, 7).astype("float32"))


@pytest.fixture(
    params=[
        (MeanFieldGroup, {}),
        (FullRankGroup, {}),
        (NormalizingFlowGroup, {"flow": "scale"}),
        (NormalizingFlowGroup, {"flow": "loc"}),
        (NormalizingFlowGroup, {"flow": "hh"}),
        (NormalizingFlowGroup, {"flow": "planar"}),
        (NormalizingFlowGroup, {"flow": "radial"}),
        (NormalizingFlowGroup, {"flow": "radial-loc"}),
    ],
    ids=lambda t: "{c}: {d}".format(c=t[0].__name__, d=t[1]),
)
def parametric_grouped_approxes(request):
    return request.param


@pytest.fixture
def three_var_aevb_groups(parametric_grouped_approxes, three_var_model, aevb_initial):
    dsize = np.prod(pymc3.util.get_transformed(three_var_model.one).dshape[1:])
    cls, kw = parametric_grouped_approxes
    spec = cls.get_param_spec_for(d=dsize, **kw)
    params = dict()
    for k, v in spec.items():
        if isinstance(k, int):
            params[k] = dict()
            for k_i, v_i in v.items():
                params[k][k_i] = aevb_initial.dot(np.random.rand(7, *v_i).astype("float32"))
        else:
            params[k] = aevb_initial.dot(np.random.rand(7, *v).astype("float32"))
    aevb_g = cls([three_var_model.one], params=params, model=three_var_model, local=True)
    return [aevb_g, MeanFieldGroup(None, model=three_var_model)]


@pytest.fixture
def three_var_aevb_approx(three_var_model, three_var_aevb_groups):
    approx = Approximation(three_var_aevb_groups, model=three_var_model)
    return approx


def test_sample_aevb(three_var_aevb_approx, aevb_initial):
    pm.KLqp(three_var_aevb_approx).fit(
        1, more_replacements={aevb_initial: np.zeros_like(aevb_initial.get_value())[:1]}
    )
    aevb_initial.set_value(np.random.rand(7, 7).astype("float32"))
    trace = three_var_aevb_approx.sample(500)
    assert set(trace.varnames) == {"one", "one_log__", "two", "three"}
    assert len(trace) == 500
    assert trace[0]["one"].shape == (7, 2)
    assert trace[0]["two"].shape == (10,)
    assert trace[0]["three"].shape == (10, 1, 2)

    aevb_initial.set_value(np.random.rand(13, 7).astype("float32"))
    trace = three_var_aevb_approx.sample(500)
    assert set(trace.varnames) == {"one", "one_log__", "two", "three"}
    assert len(trace) == 500
    assert trace[0]["one"].shape == (13, 2)
    assert trace[0]["two"].shape == (10,)
    assert trace[0]["three"].shape == (10, 1, 2)


def test_replacements_in_sample_node_aevb(three_var_aevb_approx, aevb_initial):
    inp = tt.matrix(dtype="float32")
    three_var_aevb_approx.sample_node(
        three_var_aevb_approx.model.one, 2, more_replacements={aevb_initial: inp}
    ).eval({inp: np.random.rand(7, 7).astype("float32")})

    three_var_aevb_approx.sample_node(
        three_var_aevb_approx.model.one, None, more_replacements={aevb_initial: inp}
    ).eval({inp: np.random.rand(7, 7).astype("float32")})


def test_vae():
    minibatch_size = 10
    data = pm.floatX(np.random.rand(100))
    x_mini = pm.Minibatch(data, minibatch_size)
    x_inp = tt.vector()
    x_inp.tag.test_value = data[:minibatch_size]

    ae = theano.shared(pm.floatX([0.1, 0.1]))
    be = theano.shared(pm.floatX(1.0))

    ad = theano.shared(pm.floatX(1.0))
    bd = theano.shared(pm.floatX(1.0))

    enc = x_inp.dimshuffle(0, "x") * ae.dimshuffle("x", 0) + be
    mu, rho = enc[:, 0], enc[:, 1]

    with pm.Model():
        # Hidden variables
        zs = pm.Normal("zs", mu=0, sigma=1, shape=minibatch_size)
        dec = zs * ad + bd
        # Observation model
        pm.Normal("xs_", mu=dec, sigma=0.1, observed=x_inp)

        pm.fit(
            1,
            local_rv={zs: dict(mu=mu, rho=rho)},
            more_replacements={x_inp: x_mini},
            more_obj_params=[ae, be, ad, bd],
        )


def test_logq_mini_1_sample_1_var(parametric_grouped_approxes, three_var_model):
    cls, kw = parametric_grouped_approxes
    approx = cls([three_var_model.one], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 1, 0)
    logq.eval()


def test_logq_mini_2_sample_2_var(parametric_grouped_approxes, three_var_model):
    cls, kw = parametric_grouped_approxes
    approx = cls([three_var_model.one, three_var_model.two], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 2, 0)
    logq.eval()


def test_logq_mini_sample_aevb(three_var_aevb_groups):
    approx = three_var_aevb_groups[0]
    logq, symbolic_logq = approx.set_size_and_deterministic(
        [approx.logq, approx.symbolic_logq], 3, 0
    )
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (3,)


def test_logq_aevb(three_var_aevb_approx):
    approx = three_var_aevb_approx
    logq, symbolic_logq = approx.set_size_and_deterministic(
        [approx.logq, approx.symbolic_logq], 1, 0
    )
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (1,)

    logq, symbolic_logq = approx.set_size_and_deterministic(
        [approx.logq, approx.symbolic_logq], 2, 0
    )
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (2,)


def test_logq_globals(three_var_approx):
    if not three_var_approx.has_logq:
        pytest.skip("%s does not implement logq" % three_var_approx)
    approx = three_var_approx
    logq, symbolic_logq = approx.set_size_and_deterministic(
        [approx.logq, approx.symbolic_logq], 1, 0
    )
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (1,)

    logq, symbolic_logq = approx.set_size_and_deterministic(
        [approx.logq, approx.symbolic_logq], 2, 0
    )
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (2,)


@pytest.mark.parametrize(
    "raises, vfam, type_, kw",
    [
        (not_raises(), "mean_field", MeanFieldGroup, {}),
        (not_raises(), "mf", MeanFieldGroup, {}),
        (not_raises(), "full_rank", FullRankGroup, {}),
        (not_raises(), "fr", FullRankGroup, {}),
        (not_raises(), "FR", FullRankGroup, {}),
        (not_raises(), "loc", NormalizingFlowGroup, {}),
        (not_raises(), "scale", NormalizingFlowGroup, {}),
        (not_raises(), "hh", NormalizingFlowGroup, {}),
        (not_raises(), "planar", NormalizingFlowGroup, {}),
        (not_raises(), "radial", NormalizingFlowGroup, {}),
        (not_raises(), "scale-loc", NormalizingFlowGroup, {}),
        (
            pytest.raises(ValueError, match="Need `trace` or `size`"),
            "empirical",
            EmpiricalGroup,
            {},
        ),
        (not_raises(), "empirical", EmpiricalGroup, {"size": 100}),
    ],
)
def test_group_api_vfam(three_var_model, raises, vfam, type_, kw):
    with three_var_model, raises:
        g = Group([three_var_model.one], vfam, **kw)
        assert isinstance(g, type_)
        assert not hasattr(g, "_kwargs")
        if isinstance(g, NormalizingFlowGroup):
            assert isinstance(g.flow, pm.flows.AbstractFlow)
            assert g.flow.formula == vfam


@pytest.mark.parametrize(
    "raises, params, type_, kw, formula",
    [
        (
            not_raises(),
            dict(mu=np.ones((10, 2), "float32"), rho=np.ones((10, 2), "float32")),
            MeanFieldGroup,
            {},
            None,
        ),
        (
            not_raises(),
            dict(
                mu=np.ones((10, 2), "float32"),
                L_tril=np.ones(
                    FullRankGroup.get_param_spec_for(d=np.prod((10, 2)))["L_tril"], "float32"
                ),
            ),
            FullRankGroup,
            {},
            None,
        ),
        (not_raises(), {0: dict(loc=np.ones((10, 2), "float32"))}, NormalizingFlowGroup, {}, "loc"),
        (
            not_raises(),
            {0: dict(rho=np.ones((10, 2), "float32"))},
            NormalizingFlowGroup,
            {},
            "scale",
        ),
        (
            not_raises(),
            {
                0: dict(
                    v=np.ones((10, 2), "float32"),
                )
            },
            NormalizingFlowGroup,
            {},
            "hh",
        ),
        (
            not_raises(),
            {0: dict(u=np.ones((10, 2), "float32"), w=np.ones((10, 2), "float32"), b=1.0)},
            NormalizingFlowGroup,
            {},
            "planar",
        ),
        (
            not_raises(),
            {0: dict(z_ref=np.ones((10, 2), "float32"), a=1.0, b=1.0)},
            NormalizingFlowGroup,
            {},
            "radial",
        ),
        (
            not_raises(),
            {0: dict(rho=np.ones((10, 2), "float32")), 1: dict(loc=np.ones((10, 2), "float32"))},
            NormalizingFlowGroup,
            {},
            "scale-loc",
        ),
        (not_raises(), dict(histogram=np.ones((20, 10, 2), "float32")), EmpiricalGroup, {}, None),
    ],
)
def test_group_api_params(three_var_model, raises, params, type_, kw, formula):
    with three_var_model, raises:
        g = Group([three_var_model.one], params=params, **kw)
        assert isinstance(g, type_)
        if isinstance(g, NormalizingFlowGroup):
            assert g.flow.formula == formula
        if g.has_logq:
            # should work as well
            logq = g.logq
            logq = g.set_size_and_deterministic(logq, 1, 0)
            logq.eval()


@pytest.mark.parametrize(
    "gcls, approx, kw",
    [
        (MeanFieldGroup, MeanField, {}),
        (FullRankGroup, FullRank, {}),
        (EmpiricalGroup, Empirical, {"size": 100}),
        (NormalizingFlowGroup, NormalizingFlow, {"flow": "loc"}),
        (NormalizingFlowGroup, NormalizingFlow, {"flow": "scale-loc-scale"}),
        (NormalizingFlowGroup, NormalizingFlow, {}),
    ],
)
def test_single_group_shortcuts(three_var_model, approx, kw, gcls):
    with three_var_model:
        a = approx(**kw)
    assert isinstance(a, Approximation)
    assert len(a.groups) == 1
    assert isinstance(a.groups[0], gcls)
    if isinstance(a, NormalizingFlow):
        assert a.flow.formula == kw.get("flow", NormalizingFlowGroup.default_flow)


def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])

    post_mu = np.array([1.88], dtype=theano.config.floatX)
    post_sigma = np.array([1], dtype=theano.config.floatX)
    # Create a model for test
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu0, sigma=sigma)
        pm.Normal("y", mu=mu, sigma=1, observed=y_obs)

    # Create variational gradient tensor
    mean_field = MeanField(model=model)
    with theano.config.change_flags(compute_test_value="off"):
        elbo = -pm.operators.KL(mean_field)()(10000)

    mean_field.shared_params["mu"].set_value(post_mu)
    mean_field.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

    f = theano.function([], elbo)
    elbo_mc = f()

    # Exact value
    elbo_true = -0.5 * (
        3
        + 3 * post_mu ** 2
        - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu
        + y_obs[0] ** 2
        + y_obs[1] ** 2
        + mu0 ** 2
        + 3 * np.log(2 * np.pi)
    ) + 0.5 * (np.log(2 * np.pi) + 1)
    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


@pytest.mark.parametrize("aux_total_size", range(2, 10, 3))
def test_scale_cost_to_minibatch_works(aux_total_size):
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
    beta = len(y_obs) / float(aux_total_size)

    # TODO: theano_config
    # with pm.Model(theano_config=dict(floatX='float64')):
    # did not not work as expected
    # there were some numeric problems, so float64 is forced
    with theano.config.change_flags(floatX="float64", warn_float64="ignore"):

        assert theano.config.floatX == "float64"
        assert theano.config.warn_float64 == "ignore"

        post_mu = np.array([1.88], dtype=theano.config.floatX)
        post_sigma = np.array([1], dtype=theano.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            assert mean_field_1.scale_cost_to_minibatch
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with theano.config.change_flags(compute_test_value="off"):
                elbo_via_total_size_scaled = -pm.operators.KL(mean_field_1)()(10000)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_2 = MeanField()
            assert mean_field_1.scale_cost_to_minibatch
            mean_field_2.scale_cost_to_minibatch = False
            assert not mean_field_2.scale_cost_to_minibatch
            mean_field_2.shared_params["mu"].set_value(post_mu)
            mean_field_2.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

        with theano.config.change_flags(compute_test_value="off"):
            elbo_via_total_size_unscaled = -pm.operators.KL(mean_field_2)()(10000)

        np.testing.assert_allclose(
            elbo_via_total_size_unscaled.eval(),
            elbo_via_total_size_scaled.eval() * pm.floatX(1 / beta),
            rtol=0.02,
            atol=1e-1,
        )


@pytest.mark.parametrize("aux_total_size", range(2, 10, 3))
def test_elbo_beta_kl(aux_total_size):
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
    beta = len(y_obs) / float(aux_total_size)

    with theano.config.change_flags(floatX="float64", warn_float64="ignore"):

        post_mu = np.array([1.88], dtype=theano.config.floatX)
        post_sigma = np.array([1], dtype=theano.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            mean_field_1.scale_cost_to_minibatch = True
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with theano.config.change_flags(compute_test_value="off"):
                elbo_via_total_size_scaled = -pm.operators.KL(mean_field_1)()(10000)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs)
            # Create variational gradient tensor
            mean_field_3 = MeanField()
            mean_field_3.shared_params["mu"].set_value(post_mu)
            mean_field_3.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with theano.config.change_flags(compute_test_value="off"):
                elbo_via_beta_kl = -pm.operators.KL(mean_field_3, beta=beta)()(10000)

        np.testing.assert_allclose(
            elbo_via_total_size_scaled.eval(), elbo_via_beta_kl.eval(), rtol=0, atol=1e-1
        )


@pytest.fixture(scope="module", params=[True, False], ids=["mini", "full"])
def use_minibatch(request):
    return request.param


@pytest.fixture
def simple_model_data(use_minibatch):
    n = 1000
    sigma0 = 2.0
    mu0 = 4.0
    sigma = 3.0
    mu = -5.0

    data = sigma * np.random.randn(n) + mu
    d = n / sigma ** 2 + 1 / sigma0 ** 2
    mu_post = (n * np.mean(data) / sigma ** 2 + mu0 / sigma0 ** 2) / d
    if use_minibatch:
        data = pm.Minibatch(data)
    return dict(
        n=n,
        data=data,
        mu_post=mu_post,
        d=d,
        mu0=mu0,
        sigma0=sigma0,
        sigma=sigma,
    )


@pytest.fixture
def simple_model(simple_model_data):
    with pm.Model() as model:
        mu_ = pm.Normal(
            "mu", mu=simple_model_data["mu0"], sigma=simple_model_data["sigma0"], testval=0
        )
        pm.Normal(
            "x",
            mu=mu_,
            sigma=simple_model_data["sigma"],
            observed=simple_model_data["data"],
            total_size=simple_model_data["n"],
        )
    return model


@pytest.fixture(
    scope="module",
    params=[
        dict(cls=NFVI, init=dict(flow="scale-loc")),
        dict(cls=ADVI, init=dict()),
        dict(cls=FullRankADVI, init=dict()),
        dict(cls=SVGD, init=dict(n_particles=500, jitter=1)),
        dict(cls=ASVGD, init=dict(temperature=1.0)),
    ],
    ids=["NFVI=scale-loc", "ADVI", "FullRankADVI", "SVGD", "ASVGD"],
)
def inference_spec(request):
    cls = request.param["cls"]
    init = request.param["init"]

    def init_(**kw):
        k = init.copy()
        k.update(kw)
        return cls(**k)

    init_.cls = cls
    return init_


@pytest.fixture(scope="function")
def inference(inference_spec, simple_model):
    with simple_model:
        return inference_spec()


@pytest.fixture(scope="function")
def fit_kwargs(inference, use_minibatch):
    _select = {
        (ADVI, "full"): dict(obj_optimizer=pm.adagrad_window(learning_rate=0.02, n_win=50), n=5000),
        (ADVI, "mini"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50), n=12000
        ),
        (NFVI, "full"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50), n=12000
        ),
        (NFVI, "mini"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50), n=12000
        ),
        (FullRankADVI, "full"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.007, n_win=50), n=6000
        ),
        (FullRankADVI, "mini"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.007, n_win=50), n=12000
        ),
        (SVGD, "full"): dict(obj_optimizer=pm.adagrad_window(learning_rate=0.075, n_win=7), n=300),
        (SVGD, "mini"): dict(obj_optimizer=pm.adagrad_window(learning_rate=0.075, n_win=7), n=300),
        (ASVGD, "full"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=10), n=500, obj_n_mc=300
        ),
        (ASVGD, "mini"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.07, n_win=10), n=500, obj_n_mc=300
        ),
    }
    if use_minibatch:
        key = "mini"
        # backward compat for PR#3071
        inference.approx.scale_cost_to_minibatch = False
    else:
        key = "full"
    return _select[(type(inference), key)]


def test_fit_oo(inference, fit_kwargs, simple_model_data):
    trace = inference.fit(**fit_kwargs).sample(10000)
    mu_post = simple_model_data["mu_post"]
    d = simple_model_data["d"]
    np.testing.assert_allclose(np.mean(trace["mu"]), mu_post, rtol=0.05)
    np.testing.assert_allclose(np.std(trace["mu"]), np.sqrt(1.0 / d), rtol=0.2)


def test_profile(inference):
    inference.run_profiling(n=100).summary()


def test_remove_scan_op():
    with pm.Model():
        pm.Normal("n", 0, 1)
        inference = ADVI()
        buff = io.StringIO()
        inference.run_profiling(n=10).summary(buff)
        assert "theano.scan.op.Scan" not in buff.getvalue()
        buff.close()


def test_clear_cache():
    import pickle

    pymc3.memoize.clear_cache()
    assert all(len(c) == 0 for c in pymc3.memoize.CACHE_REGISTRY)
    with pm.Model():
        pm.Normal("n", 0, 1)
        inference = ADVI()
        inference.fit(n=10)
        assert any(len(c) != 0 for c in inference.approx._cache.values())
        pymc3.memoize.clear_cache(inference.approx)
        # should not be cleared at this call
        assert all(len(c) == 0 for c in inference.approx._cache.values())
        new_a = pickle.loads(pickle.dumps(inference.approx))
        assert not hasattr(new_a, "_cache")
        inference_new = pm.KLqp(new_a)
        inference_new.fit(n=10)
        assert any(len(c) != 0 for c in inference_new.approx._cache.values())
        pymc3.memoize.clear_cache(inference_new.approx)
        assert all(len(c) == 0 for c in inference_new.approx._cache.values())


@pytest.fixture(scope="module")
def another_simple_model():
    _model = models.simple_model()[1]
    with _model:
        pm.Potential("pot", tt.ones((10, 10)))
    return _model


@pytest.fixture(
    params=[
        dict(name="advi", kw=dict(start={})),
        dict(name="fullrank_advi", kw=dict(start={})),
        dict(name="svgd", kw=dict(start={})),
    ],
    ids=lambda d: d["name"],
)
def fit_method_with_object(request, another_simple_model):
    _select = dict(advi=ADVI, fullrank_advi=FullRankADVI, svgd=SVGD)
    with another_simple_model:
        return _select[request.param["name"]](**request.param["kw"])


@pytest.mark.parametrize(
    ["method", "kwargs", "error"],
    [
        ("undefined", dict(), KeyError),
        (1, dict(), TypeError),
        ("advi", dict(total_grad_norm_constraint=10), None),
        ("fullrank_advi", dict(), None),
        ("svgd", dict(total_grad_norm_constraint=10), None),
        ("svgd", dict(start={}), None),
        # start argument is not allowed for ASVGD
        ("asvgd", dict(start={}, total_grad_norm_constraint=10), TypeError),
        ("asvgd", dict(total_grad_norm_constraint=10), None),
        ("nfvi", dict(start={}), None),
        ("nfvi=scale-loc", dict(start={}), None),
        ("nfvi=bad-formula", dict(start={}), KeyError),
    ],
)
def test_fit_fn_text(method, kwargs, error, another_simple_model):
    with another_simple_model:
        if error is not None:
            with pytest.raises(error):
                fit(10, method=method, **kwargs)
        else:
            fit(10, method=method, **kwargs)


@pytest.fixture(scope="module")
def aevb_model():
    with pm.Model() as model:
        pm.HalfNormal("x", shape=(2,), total_size=5)
        pm.Normal("y", shape=(2,))
    x = model.x
    y = model.y
    mu = theano.shared(x.init_value)
    rho = theano.shared(np.zeros_like(x.init_value))
    return {"model": model, "y": y, "x": x, "replace": dict(mu=mu, rho=rho)}


def test_aevb(inference_spec, aevb_model):
    # add to inference that supports aevb
    x = aevb_model["x"]
    y = aevb_model["y"]
    model = aevb_model["model"]
    replace = aevb_model["replace"]
    with model:
        try:
            inference = inference_spec(
                local_rv={x: {"mu": replace["mu"] * 5, "rho": replace["rho"]}}
            )
            approx = inference.fit(3, obj_n_mc=2, more_obj_params=list(replace.values()))
            approx.sample(10)
            approx.sample_node(y, more_replacements={x: np.asarray([1, 1], dtype=x.dtype)}).eval()
        except pm.opvi.AEVBInferenceError:
            pytest.skip("Does not support AEVB")


def test_rowwise_approx(three_var_model, parametric_grouped_approxes):
    # add to inference that supports aevb
    cls, kw = parametric_grouped_approxes
    with three_var_model:
        try:
            approx = Approximation(
                [cls([three_var_model.one], rowwise=True, **kw), Group(None, vfam="mf")]
            )
            inference = pm.KLqp(approx)
            approx = inference.fit(3, obj_n_mc=2)
            approx.sample(10)
            approx.sample_node(three_var_model.one).eval()
        except pm.opvi.BatchedGroupError:
            pytest.skip("Does not support rowwise grouping")


def test_pickle_approx(three_var_approx):
    import pickle

    dump = pickle.dumps(three_var_approx)
    new = pickle.loads(dump)
    assert new.sample(1)


def test_pickle_single_group(three_var_approx_single_group_mf):
    import pickle

    dump = pickle.dumps(three_var_approx_single_group_mf)
    new = pickle.loads(dump)
    assert new.sample(1)


def test_pickle_approx_aevb(three_var_aevb_approx):
    import pickle

    dump = pickle.dumps(three_var_aevb_approx)
    new = pickle.loads(dump)
    assert new.sample(1000)


@pytest.fixture(scope="module")
def binomial_model():
    n_samples = 100
    xs = intX(np.random.binomial(n=1, p=0.2, size=n_samples))
    with pm.Model() as model:
        p = pm.Beta("p", alpha=1, beta=1)
        pm.Binomial("xs", n=1, p=p, observed=xs)
    return model


@pytest.fixture(scope="module")
def binomial_model_inference(binomial_model, inference_spec):
    with binomial_model:
        return inference_spec()


def test_replacements(binomial_model_inference):
    d = tt.bscalar()
    d.tag.test_value = 1
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p ** 3
    p_s = approx.sample_node(p_t)
    if theano.config.compute_test_value != "off":
        assert p_s.tag.test_value.shape == p_t.tag.test_value.shape
    sampled = [p_s.eval() for _ in range(100)]
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic

    p_d = approx.sample_node(p_t, deterministic=True)
    sampled = [p_d.eval() for _ in range(100)]
    assert all(map(operator.eq, sampled[1:], sampled[:-1]))  # deterministic

    p_r = approx.sample_node(p_t, deterministic=d)
    sampled = [p_r.eval({d: 1}) for _ in range(100)]
    assert all(map(operator.eq, sampled[1:], sampled[:-1]))  # deterministic
    sampled = [p_r.eval({d: 0}) for _ in range(100)]
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic


def test_sample_replacements(binomial_model_inference):
    i = tt.iscalar()
    i.tag.test_value = 1
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p ** 3
    p_s = approx.sample_node(p_t, size=100)
    if theano.config.compute_test_value != "off":
        assert p_s.tag.test_value.shape == (100,) + p_t.tag.test_value.shape
    sampled = p_s.eval()
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic
    assert sampled.shape[0] == 100

    p_d = approx.sample_node(p_t, size=i)
    sampled = p_d.eval({i: 100})
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # deterministic
    assert sampled.shape[0] == 100
    sampled = p_d.eval({i: 101})
    assert sampled.shape[0] == 101


def test_discrete_not_allowed():
    mu_true = np.array([-2, 0, 2])
    z_true = np.random.randint(len(mu_true), size=100)
    y = np.random.normal(mu_true[z_true], np.ones_like(z_true))

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=10, shape=3)
        z = pm.Categorical("z", p=tt.ones(3) / 3, shape=len(y))
        pm.Normal("y_obs", mu=mu[z], sigma=1.0, observed=y)
        with pytest.raises(opvi.ParametrizationError):
            pm.fit(n=1)  # fails


def test_var_replacement():
    X_mean = pm.floatX(np.linspace(0, 10, 10))
    y = pm.floatX(np.random.normal(X_mean * 4, 0.05))
    with pm.Model():
        inp = pm.Normal("X", X_mean, shape=X_mean.shape)
        coef = pm.Normal("b", 4.0)
        mean = inp * coef
        pm.Normal("y", mean, 0.1, observed=y)
        advi = pm.fit(100)
        assert advi.sample_node(mean).eval().shape == (10,)
        x_new = pm.floatX(np.linspace(0, 10, 11))
        assert advi.sample_node(mean, more_replacements={inp: x_new}).eval().shape == (11,)


def test_empirical_from_trace(another_simple_model):
    with another_simple_model:
        step = pm.Metropolis()
        trace = pm.sample(100, step=step, chains=1, tune=0)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 100
        trace = pm.sample(100, step=step, chains=4, tune=0)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 400


@pytest.fixture(
    params=[
        dict(cls=flows.PlanarFlow, init=dict(jitter=0.1)),
        dict(cls=flows.RadialFlow, init=dict(jitter=0.1)),
        dict(cls=flows.ScaleFlow, init=dict(jitter=0.1)),
        dict(cls=flows.LocFlow, init=dict(jitter=0.1)),
        dict(cls=flows.HouseholderFlow, init=dict(jitter=0.1)),
    ],
    ids=lambda d: d["cls"].__name__,
)
def flow_spec(request):
    cls = request.param["cls"]
    init = request.param["init"]

    def init_(**kw):
        k = init.copy()
        k.update(kw)
        return cls(**k)

    init_.cls = cls
    return init_


def test_flow_det(flow_spec):
    z0 = tt.arange(0, 20).astype("float32")
    flow = flow_spec(dim=20, z0=z0.dimshuffle("x", 0))
    with theano.config.change_flags(compute_test_value="off"):
        z1 = flow.forward.flatten()
        J = tt.jacobian(z1, z0)
        logJdet = tt.log(tt.abs_(tt.nlinalg.det(J)))
        det = flow.logdet[0]
    np.testing.assert_allclose(logJdet.eval(), det.eval(), atol=0.0001)


def test_flow_det_local(flow_spec):
    z0 = tt.arange(0, 12).astype("float32")
    spec = flow_spec.cls.get_param_spec_for(d=12)
    params = dict()
    for k, shp in spec.items():
        params[k] = np.random.randn(1, *shp).astype("float32")
    flow = flow_spec(dim=12, z0=z0.reshape((1, 1, 12)), **params)
    assert flow.batched
    with theano.config.change_flags(compute_test_value="off"):
        z1 = flow.forward.flatten()
        J = tt.jacobian(z1, z0)
        logJdet = tt.log(tt.abs_(tt.nlinalg.det(J)))
        det = flow.logdet[0]
    np.testing.assert_allclose(logJdet.eval(), det.eval(), atol=0.0001)


def test_flows_collect_chain():
    initial = tt.ones((3, 2))
    flow1 = flows.PlanarFlow(dim=2, z0=initial)
    flow2 = flows.PlanarFlow(dim=2, z0=flow1)
    assert len(flow2.params) == 3
    assert len(flow2.all_params) == 6
    np.testing.assert_allclose(flow1.logdet.eval() + flow2.logdet.eval(), flow2.sum_logdets.eval())


@pytest.mark.parametrize(
    "formula,length,order",
    [
        ("planar", 1, [flows.PlanarFlow]),
        ("planar*2", 2, [flows.PlanarFlow] * 2),
        ("planar-planar", 2, [flows.PlanarFlow] * 2),
        ("planar-planar*2", 3, [flows.PlanarFlow] * 3),
        ("hh-planar*2", 3, [flows.HouseholderFlow] + [flows.PlanarFlow] * 2),
    ],
)
def test_flow_formula(formula, length, order):
    spec = flows.Formula(formula)
    flows_list = spec.flows
    assert len(flows_list) == length
    if order is not None:
        assert flows_list == order
    spec(dim=2, jitter=1)(tt.ones((3, 2))).eval()  # should work
