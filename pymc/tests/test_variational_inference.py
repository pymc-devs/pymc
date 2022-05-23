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

import aesara
import aesara.tensor as at
import numpy as np
import pytest

import pymc as pm

from pymc.aesaraf import intX
from pymc.tests import models
from pymc.variational import opvi
from pymc.variational.approximations import Empirical, MeanField
from pymc.variational.inference import ADVI, SVGD, FullRankADVI
from pymc.variational.opvi import NotImplementedInference

pytestmark = pytest.mark.usefixtures("strict_float32", "seeded_test")


def ignore_not_implemented_inference(func):
    @functools.wraps(func)
    def new_test(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedInference:
            pytest.xfail("NotImplementedInference")

    return new_test


@pytest.mark.parametrize("diff", ["relative", "absolute"])
@pytest.mark.parametrize("ord", [1, 2, np.inf])
def test_callbacks_convergence(diff, ord):
    cb = pm.variational.callbacks.CheckParametersConvergence(every=1, diff=diff, ord=ord)

    class _approx:
        params = (aesara.shared(np.asarray([1, 2, 3])),)

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
        pm.HalfNormal("one", size=(10, 2), total_size=100)
        pm.Normal("two", size=(10,))
        pm.Normal("three", size=(10, 1, 2))
    return model


@pytest.fixture
def three_var_approx_single_group_mf(three_var_model):
    return MeanField(model=three_var_model)


@pytest.fixture
def test_sample_simple(three_var_approx, request):
    backend, name = request.param
    trace = three_var_approx.sample(100, name=name, return_inferencedata=False)
    assert set(trace.varnames) == {"one", "one_log__", "three", "two"}
    assert len(trace) == 100
    assert trace[0]["one"].shape == (10, 2)
    assert trace[0]["two"].shape == (10,)
    assert trace[0]["three"].shape == (10, 1, 2)


@pytest.fixture
def aevb_initial():
    return aesara.shared(np.random.rand(3, 7).astype("float32"))


@ignore_not_implemented_inference
def test_vae():
    minibatch_size = 10
    data = pm.floatX(np.random.rand(100))
    x_mini = pm.Minibatch(data, minibatch_size)
    x_inp = at.vector()
    x_inp.tag.test_value = data[:minibatch_size]

    ae = aesara.shared(pm.floatX([0.1, 0.1]))
    be = aesara.shared(pm.floatX(1.0))

    ad = aesara.shared(pm.floatX(1.0))
    bd = aesara.shared(pm.floatX(1.0))

    enc = x_inp.dimshuffle(0, "x") * ae.dimshuffle("x", 0) + be
    mu, rho = enc[:, 0], enc[:, 1]

    with pm.Model():
        # Hidden variables
        zs = pm.Normal("zs", mu=0, sigma=1, size=minibatch_size)
        dec = zs * ad + bd
        # Observation model
        pm.Normal("xs_", mu=dec, sigma=0.1, observed=x_inp)

        pm.fit(
            1,
            local_rv={zs: dict(mu=mu, rho=rho)},
            more_replacements={x_inp: x_mini},
            more_obj_params=[ae, be, ad, bd],
        )


def test_elbo():
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])

    post_mu = np.array([1.88], dtype=aesara.config.floatX)
    post_sigma = np.array([1], dtype=aesara.config.floatX)
    # Create a model for test
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu0, sigma=sigma)
        pm.Normal("y", mu=mu, sigma=1, observed=y_obs)

    # Create variational gradient tensor
    mean_field = MeanField(model=model)
    with aesara.config.change_flags(compute_test_value="off"):
        elbo = -pm.operators.KL(mean_field)()(10000)

    mean_field.shared_params["mu"].set_value(post_mu)
    mean_field.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

    f = aesara.function([], elbo)
    elbo_mc = f()

    # Exact value
    elbo_true = -0.5 * (
        3
        + 3 * post_mu**2
        - 2 * (y_obs[0] + y_obs[1] + mu0) * post_mu
        + y_obs[0] ** 2
        + y_obs[1] ** 2
        + mu0**2
        + 3 * np.log(2 * np.pi)
    ) + 0.5 * (np.log(2 * np.pi) + 1)
    np.testing.assert_allclose(elbo_mc, elbo_true, rtol=0, atol=1e-1)


@pytest.mark.parametrize("aux_total_size", range(2, 10, 3))
def test_scale_cost_to_minibatch_works(aux_total_size):
    mu0 = 1.5
    sigma = 1.0
    y_obs = np.array([1.6, 1.4])
    beta = len(y_obs) / float(aux_total_size)

    # TODO: aesara_config
    # with pm.Model(aesara_config=dict(floatX='float64')):
    # did not not work as expected
    # there were some numeric problems, so float64 is forced
    with aesara.config.change_flags(floatX="float64", warn_float64="ignore"):

        assert aesara.config.floatX == "float64"
        assert aesara.config.warn_float64 == "ignore"

        post_mu = np.array([1.88], dtype=aesara.config.floatX)
        post_sigma = np.array([1], dtype=aesara.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            assert mean_field_1.scale_cost_to_minibatch
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with aesara.config.change_flags(compute_test_value="off"):
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

        with aesara.config.change_flags(compute_test_value="off"):
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

    with aesara.config.change_flags(floatX="float64", warn_float64="ignore"):

        post_mu = np.array([1.88], dtype=aesara.config.floatX)
        post_sigma = np.array([1], dtype=aesara.config.floatX)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs, total_size=aux_total_size)
            # Create variational gradient tensor
            mean_field_1 = MeanField()
            mean_field_1.scale_cost_to_minibatch = True
            mean_field_1.shared_params["mu"].set_value(post_mu)
            mean_field_1.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with aesara.config.change_flags(compute_test_value="off"):
                elbo_via_total_size_scaled = -pm.operators.KL(mean_field_1)()(10000)

        with pm.Model():
            mu = pm.Normal("mu", mu=mu0, sigma=sigma)
            pm.Normal("y", mu=mu, sigma=1, observed=y_obs)
            # Create variational gradient tensor
            mean_field_3 = MeanField()
            mean_field_3.shared_params["mu"].set_value(post_mu)
            mean_field_3.shared_params["rho"].set_value(np.log(np.exp(post_sigma) - 1))

            with aesara.config.change_flags(compute_test_value="off"):
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
    d = n / sigma**2 + 1 / sigma0**2
    mu_post = (n * np.mean(data) / sigma**2 + mu0 / sigma0**2) / d
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
            "mu", mu=simple_model_data["mu0"], sigma=simple_model_data["sigma0"], initval=0
        )
        pm.Normal(
            "x",
            mu=mu_,
            sigma=simple_model_data["sigma"],
            observed=simple_model_data["data"],
            total_size=simple_model_data["n"],
        )
    return model


def test_remove_scan_op():
    with pm.Model():
        pm.Normal("n", 0, 1)
        inference = ADVI()
        buff = io.StringIO()
        inference.run_profiling(n=10).summary(buff)
        assert "aesara.scan.op.Scan" not in buff.getvalue()
        buff.close()


def test_clear_cache():
    import cloudpickle

    with pm.Model():
        pm.Normal("n", 0, 1)
        inference = ADVI()
        inference.fit(n=10)
        assert any(len(c) != 0 for c in inference.approx._cache.values())
        inference.approx._cache.clear()
        # should not be cleared at this call
        assert all(len(c) == 0 for c in inference.approx._cache.values())
        new_a = cloudpickle.loads(cloudpickle.dumps(inference.approx))
        assert not hasattr(new_a, "_cache")
        inference_new = pm.KLqp(new_a)
        inference_new.fit(n=10)
        assert any(len(c) != 0 for c in inference_new.approx._cache.values())
        inference_new.approx._cache.clear()
        assert all(len(c) == 0 for c in inference_new.approx._cache.values())


@pytest.fixture(scope="module")
def another_simple_model():
    _model = models.simple_model()[1]
    with _model:
        pm.Potential("pot", at.ones((10, 10)))
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


@pytest.fixture(scope="module")
def aevb_model():
    with pm.Model() as model:
        pm.HalfNormal("x", size=(2,), total_size=5)
        pm.Normal("y", size=(2,))
    x = model.x
    y = model.y
    xr = model.initial_point(0)[model.rvs_to_values[x].name]
    mu = aesara.shared(xr)
    rho = aesara.shared(np.zeros_like(xr))
    return {"model": model, "y": y, "x": x, "replace": dict(mu=mu, rho=rho)}


def test_pickle_single_group(three_var_approx_single_group_mf):
    import cloudpickle

    dump = cloudpickle.dumps(three_var_approx_single_group_mf)
    new = cloudpickle.loads(dump)
    assert new.sample(1)


@pytest.fixture(scope="module")
def binomial_model():
    n_samples = 100
    xs = intX(np.random.binomial(n=1, p=0.2, size=n_samples))
    with pm.Model() as model:
        p = pm.Beta("p", alpha=1, beta=1)
        pm.Binomial("xs", n=1, p=p, observed=xs)
    return model


def test_discrete_not_allowed():
    mu_true = np.array([-2, 0, 2])
    z_true = np.random.randint(len(mu_true), size=100)
    y = np.random.normal(mu_true[z_true], np.ones_like(z_true))

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=10, size=3)
        z = pm.Categorical("z", p=at.ones(3) / 3, size=len(y))
        pm.Normal("y_obs", mu=mu[z], sigma=1.0, observed=y)
        with pytest.raises(opvi.ParametrizationError):
            pm.fit(n=1)  # fails


def test_var_replacement():
    X_mean = pm.floatX(np.linspace(0, 10, 10))
    y = pm.floatX(np.random.normal(X_mean * 4, 0.05))
    with pm.Model():
        inp = pm.Normal("X", X_mean, size=X_mean.shape)
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
        trace = pm.sample(100, step=step, chains=1, tune=0, return_inferencedata=False)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 100
        trace = pm.sample(100, step=step, chains=4, tune=0, return_inferencedata=False)
        emp = Empirical(trace)
        assert emp.histogram.shape[0].eval() == 400


@pytest.mark.parametrize("score", [True, False])
def test_fit_with_nans(score):
    X_mean = pm.floatX(np.linspace(0, 10, 10))
    y = pm.floatX(np.random.normal(X_mean * 4, 0.05))
    with pm.Model():
        inp = pm.Normal("X", X_mean, size=X_mean.shape)
        coef = pm.Normal("b", 4.0)
        mean = inp * coef
        pm.Normal("y", mean, 0.1, observed=y)
        with pytest.raises(FloatingPointError) as e:
            advi = pm.fit(100, score=score, obj_optimizer=pm.adam(learning_rate=float("nan")))
