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

import io
import operator
import warnings

from contextlib import nullcontext

import cloudpickle
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import pymc as pm
import pymc.variational.opvi as opvi

from pymc.pytensorf import intX
from pymc.variational.inference import ADVI, ASVGD, SVGD, FullRankADVI
from pymc.variational.opvi import NotImplementedInference
from tests import models

pytestmark = pytest.mark.usefixtures("strict_float32", "seeded_test", "fail_on_warning")


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
        data = pm.Minibatch(data, batch_size=128)
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


@pytest.fixture(
    scope="module",
    params=[
        dict(cls=ADVI, init=dict()),
        dict(cls=FullRankADVI, init=dict()),
        dict(cls=SVGD, init=dict(n_particles=500, jitter=1)),
        dict(cls=ASVGD, init=dict(temperature=1.0)),
    ],
    ids=["ADVI", "FullRankADVI", "SVGD", "ASVGD"],
)
def inference_spec(request):
    cls = request.param["cls"]
    init = request.param["init"]

    def init_(**kw):
        k = init.copy()
        k.update(kw)
        if cls == ASVGD:
            with pytest.warns(UserWarning, match="experimental inference Operator"):
                return cls(**k)
        else:
            return cls(**k)

    init_.cls = cls
    return init_


@pytest.fixture(scope="function")
def inference(inference_spec, simple_model):
    with simple_model:
        return inference_spec(random_seed=42)


@pytest.fixture(scope="function")
def fit_kwargs(inference, use_minibatch):
    _select = {
        (ADVI, "full"): dict(obj_optimizer=pm.adagrad_window(learning_rate=0.02, n_win=50), n=5000),
        (ADVI, "mini"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.01, n_win=50), n=12000
        ),
        (FullRankADVI, "full"): dict(
            obj_optimizer=pm.adagrad_window(learning_rate=0.015, n_win=50), n=6000
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
    if (type(inference), key) in {(SVGD, "mini"), (ASVGD, "mini")}:
        pytest.skip("Not Implemented Inference")
    return _select[(type(inference), key)]


def test_fit_oo(inference, fit_kwargs, simple_model_data):
    # Minibatch data can't be extracted into the `observed_data` group in the final InferenceData
    if getattr(simple_model_data["data"], "name", "").startswith("minibatch"):
        warn_ctxt = pytest.warns(
            UserWarning, match="Could not extract data from symbolic observation"
        )
    else:
        warn_ctxt = nullcontext()

    with warn_ctxt:
        trace = inference.fit(**fit_kwargs).sample(10000)
    mu_post = simple_model_data["mu_post"]
    d = simple_model_data["d"]
    np.testing.assert_allclose(np.mean(trace.posterior["mu"]), mu_post, rtol=0.05)
    np.testing.assert_allclose(np.std(trace.posterior["mu"]), np.sqrt(1.0 / d), rtol=0.2)


def test_fit_start(inference_spec, simple_model):
    mu_init = 17
    mu_sigma_init = 13

    with simple_model:
        if type(inference_spec()) == ASVGD:
            # ASVGD doesn't support the start argument
            return
        elif type(inference_spec()) == ADVI:
            has_start_sigma = True
        else:
            has_start_sigma = False

    kw = {"start": {"mu": mu_init}}
    if has_start_sigma:
        kw.update({"start_sigma": {"mu": mu_sigma_init}})
    with simple_model:
        inference = inference_spec(**kw)

    # Minibatch data can't be extracted into the `observed_data` group in the final InferenceData
    [observed_value] = [simple_model.rvs_to_values[obs] for obs in simple_model.observed_RVs]
    if observed_value.name.startswith("minibatch"):
        warn_ctxt = pytest.warns(
            UserWarning, match="Could not extract data from symbolic observation"
        )
    else:
        warn_ctxt = nullcontext()

    warning_raised = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            trace = inference.fit(n=0).sample(10000)
        except NotImplementedInference as e:
            pytest.skip(str(e))

    if len(w) > 0:
        for item in w:
            if issubclass(
                item.category, UserWarning
            ) and "Could not extract data from symbolic observation" in str(item.message):
                warning_raised = True
    if observed_value.name.startswith("minibatch"):
        assert warning_raised
    else:
        assert not warning_raised
    np.testing.assert_allclose(np.mean(trace.posterior["mu"]), mu_init, rtol=0.05)
    if has_start_sigma:
        np.testing.assert_allclose(np.std(trace.posterior["mu"]), mu_sigma_init, rtol=0.05)


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
        ("nfvi=bad-formula", dict(start={}), KeyError),
    ],
)
def test_fit_fn_text(method, kwargs, error):
    with models.another_simple_model():
        if method == "asvgd":
            with pytest.warns(UserWarning, match="experimental inference Operator"):
                if error is not None:
                    with pytest.raises(error):
                        pm.fit(10, method=method, **kwargs)
                else:
                    pm.fit(10, method=method, **kwargs)
        else:
            if error is not None:
                with pytest.raises(error):
                    pm.fit(10, method=method, **kwargs)
            else:
                pm.fit(10, method=method, **kwargs)


def test_profile(inference):
    if type(inference) in {SVGD, ASVGD}:
        pytest.skip("Not Implemented Inference")
    inference.run_profiling(n=100).summary()


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


@pytest.mark.xfail("pytensor.config.warn_float64 == 'raise'", reason="too strict float32")
def test_replacements(binomial_model_inference):
    d = pytensor.shared(1)
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p**3
    p_s = approx.sample_node(p_t)
    assert not any(
        isinstance(n.owner.op, pytensor.tensor.random.basic.BetaRV)
        for n in pytensor.graph.ancestors([p_s])
        if n.owner
    ), "p should be replaced"
    if pytensor.config.compute_test_value != "off":
        assert p_s.tag.test_value.shape == p_t.tag.test_value.shape
    sampled = [pm.draw(p_s) for _ in range(100)]
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic
    p_z = approx.sample_node(p_t, deterministic=False, size=10)
    assert p_z.shape.eval() == (10,)
    try:
        p_z = approx.sample_node(p_t, deterministic=True, size=10)
        assert p_z.shape.eval() == (10,)
    except opvi.NotImplementedInference:
        pass

    try:
        p_d = approx.sample_node(p_t, deterministic=True)
        sampled = [pm.draw(p_d) for _ in range(100)]
        assert all(map(operator.eq, sampled[1:], sampled[:-1]))  # deterministic
    except opvi.NotImplementedInference:
        pass

    p_r = approx.sample_node(p_t, deterministic=d)
    d.set_value(1)
    sampled = [pm.draw(p_r) for _ in range(100)]
    assert all(map(operator.eq, sampled[1:], sampled[:-1]))  # deterministic
    d.set_value(0)
    sampled = [pm.draw(p_r) for _ in range(100)]
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic


def test_sample_replacements(binomial_model_inference):
    i = pt.iscalar()
    i.tag.test_value = 1
    approx = binomial_model_inference.approx
    p = approx.model.p
    p_t = p**3
    p_s = approx.sample_node(p_t, size=100)
    if pytensor.config.compute_test_value != "off":
        assert p_s.tag.test_value.shape == (100, *p_t.tag.test_value.shape)
    sampled = p_s.eval()
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # stochastic
    assert sampled.shape[0] == 100

    p_d = approx.sample_node(p_t, size=i)
    sampled = p_d.eval({i: 100})
    assert any(map(operator.ne, sampled[1:], sampled[:-1]))  # deterministic
    assert sampled.shape[0] == 100
    sampled = p_d.eval({i: 101})
    assert sampled.shape[0] == 101


def test_remove_scan_op():
    with pm.Model():
        pm.Normal("n", 0, 1)
        inference = ADVI()
        buff = io.StringIO()
        inference.run_profiling(n=10).summary(buff)
        assert "pytensor.scan.op.Scan" not in buff.getvalue()
        buff.close()


def test_var_replacement():
    X_mean = pm.floatX(np.linspace(0, 10, 10))
    y = pm.floatX(np.random.normal(X_mean * 4, 0.05))
    inp_size = pytensor.shared(np.array(10, dtype="int64"), name="inp_size")
    with pm.Model():
        inp = pm.Normal("X", X_mean, size=(inp_size,))
        coef = pm.Normal("b", 4.0)
        mean = inp * coef
        pm.Normal("y", mean, 0.1, shape=inp.shape, observed=y)
        advi = pm.fit(100)
        assert advi.sample_node(mean).eval().shape == (10,)

        inp_size.set_value(11)
        x_new = pm.floatX(np.linspace(0, 10, 11))
        assert advi.sample_node(mean, more_replacements={inp: x_new}).eval().shape == (11,)


def test_clear_cache():
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


def test_fit_data(inference, fit_kwargs, simple_model_data):
    fitted = inference.fit(**fit_kwargs)
    mu_post = simple_model_data["mu_post"]
    d = simple_model_data["d"]
    np.testing.assert_allclose(fitted.mean_data["mu"].values, mu_post, rtol=0.05)
    np.testing.assert_allclose(fitted.std_data["mu"], np.sqrt(1.0 / d), rtol=0.2)


@pytest.fixture
def hierarchical_model_data():
    group_coords = {
        "group_d1": np.arange(3),
        "group_d2": np.arange(7),
    }
    group_shape = tuple(len(d) for d in group_coords.values())
    data_coords = {"data_d": np.arange(11), **group_coords}

    data_shape = tuple(len(d) for d in data_coords.values())

    mu = -5.0

    sigma_group_mu = 3
    group_mu = sigma_group_mu * np.random.randn(*group_shape)

    sigma = 3.0

    data = sigma * np.random.randn(*data_shape) + group_mu + mu

    return dict(
        group_coords=group_coords,
        group_shape=group_shape,
        data_coords=data_coords,
        data_shape=data_shape,
        mu=mu,
        sigma_group_mu=sigma_group_mu,
        sigma=sigma,
        group_mu=group_mu,
        data=data,
    )


@pytest.fixture
def hierarchical_model(hierarchical_model_data):
    with pm.Model(coords=hierarchical_model_data["data_coords"]) as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma_group_mu = pm.HalfNormal("sigma_group_mu", sigma=5)

        group_mu = pm.Normal(
            "group_mu",
            mu=0,
            sigma=sigma_group_mu,
            dims=list(hierarchical_model_data["group_coords"].keys()),
        )

        sigma = pm.HalfNormal("sigma", sigma=3)

        pm.Normal(
            "data",
            mu=(mu + group_mu),
            sigma=sigma,
            observed=hierarchical_model_data["data"],
        )
    return model


def test_fit_data_coords(hierarchical_model, hierarchical_model_data):
    with hierarchical_model:
        fitted = pm.fit(1)

    for data in [fitted.mean_data, fitted.std_data]:
        assert set(data.keys()) == {"sigma_group_mu_log__", "sigma_log__", "group_mu", "mu"}
        assert data["group_mu"].shape == hierarchical_model_data["group_shape"]
        assert list(data["group_mu"].coords.keys()) == list(
            hierarchical_model_data["group_coords"].keys()
        )
        assert data["mu"].shape == tuple()
