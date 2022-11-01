#   Copyright 2022 The PyMC Developers
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
import re

import aesara
import aesara.tensor as at
import numpy as np
import pytest

from aesara import Mode
from aesara.compile import SharedVariable

import pymc as pm

from pymc.backends.base import MultiTrace
from pymc.sampling_utils import (
    _get_seeds_per_chain,
    compile_forward_sampling_function,
    get_vars_in_point_list,
)
from pymc.tests.helpers import SeededTest


class TestDraw(SeededTest):
    def test_univariate(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws = pm.draw(x)
        assert x_draws.shape == ()

        (x_draws,) = pm.draw([x])
        assert x_draws.shape == ()

        x_draws = pm.draw(x, draws=10)
        assert x_draws.shape == (10,)

        (x_draws,) = pm.draw([x], draws=10)
        assert x_draws.shape == (10,)

    def test_multivariate(self):
        with pm.Model():
            mln = pm.Multinomial("mln", n=5, p=np.array([0.25, 0.25, 0.25, 0.25]))

        mln_draws = pm.draw(mln, draws=1)
        assert mln_draws.shape == (4,)

        (mln_draws,) = pm.draw([mln], draws=1)
        assert mln_draws.shape == (4,)

        mln_draws = pm.draw(mln, draws=10)
        assert mln_draws.shape == (10, 4)

        (mln_draws,) = pm.draw([mln], draws=10)
        assert mln_draws.shape == (10, 4)

    def test_multiple_variables(self):
        with pm.Model():
            x = pm.Normal("x")
            y = pm.Normal("y", shape=10)
            z = pm.Uniform("z", shape=5)
            w = pm.Dirichlet("w", a=[1, 1, 1])

        num_draws = 100
        draws = pm.draw((x, y, z, w), draws=num_draws)
        assert draws[0].shape == (num_draws,)
        assert draws[1].shape == (num_draws, 10)
        assert draws[2].shape == (num_draws, 5)
        assert draws[3].shape == (num_draws, 3)

    def test_draw_different_samples(self):
        with pm.Model():
            x = pm.Normal("x")

        x_draws_1 = pm.draw(x, 100)
        x_draws_2 = pm.draw(x, 100)
        assert not np.all(np.isclose(x_draws_1, x_draws_2))

    def test_draw_aesara_function_kwargs(self):
        sharedvar = aesara.shared(0)
        x = pm.DiracDelta.dist(0.0)
        y = x + sharedvar
        draws = pm.draw(
            y,
            draws=5,
            mode=Mode("py"),
            updates={sharedvar: sharedvar + 1},
        )
        assert np.all(draws == np.arange(5))


class TestCompileForwardSampler:
    @staticmethod
    def get_function_roots(function):
        return [
            var
            for var in aesara.graph.basic.graph_inputs(function.maker.fgraph.outputs)
            if var.name
        ]

    @staticmethod
    def get_function_inputs(function):
        return {i for i in function.maker.fgraph.inputs if not isinstance(i, SharedVariable)}

    def test_linear_model(self):
        with pm.Model() as model:
            x = pm.MutableData("x", np.linspace(0, 1, 10))
            y = pm.MutableData("y", np.ones(10))

            alpha = pm.Normal("alpha", 0, 0.1)
            beta = pm.Normal("beta", 0, 0.1)
            mu = pm.Deterministic("mu", alpha + beta * x)
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=x.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            [obs],
            vars_in_trace=[alpha, beta, sigma, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"alpha", "beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "alpha", "beta", "sigma"}

        with pm.Model() as model:
            x = pm.ConstantData("x", np.linspace(0, 1, 10))
            y = pm.MutableData("y", np.ones(10))

            alpha = pm.Normal("alpha", 0, 0.1)
            beta = pm.Normal("beta", 0, 0.1)
            mu = pm.Deterministic("mu", alpha + beta * x)
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=x.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            [obs],
            vars_in_trace=[alpha, beta, sigma, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"alpha", "beta", "sigma", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma"}

    def test_nested_observed_model(self):
        with pm.Model() as model:
            p = pm.ConstantData("p", np.array([0.25, 0.5, 0.25]))
            x = pm.MutableData("x", np.zeros(10))
            y = pm.MutableData("y", np.ones(10))

            category = pm.Categorical("category", p, observed=x)
            beta = pm.Normal("beta", 0, 0.1, size=p.shape)
            mu = pm.Deterministic("mu", beta[category])
            sigma = pm.HalfNormal("sigma", 0.1)
            obs = pm.Normal("obs", mu, sigma, observed=y, shape=mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {category, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"x", "p", "beta", "sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[beta, mu, sigma],
            basic_rvs=model.basic_RVs,
            givens_dict={category: np.zeros(10, dtype=category.dtype)},
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"beta", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "x",
            "p",
            "category",
            "beta",
            "sigma",
        }

    def test_volatile_parameters(self):
        with pm.Model() as model:
            y = pm.MutableData("y", np.ones(10))
            mu = pm.Normal("mu", 0, 1)
            nested_mu = pm.Normal("nested_mu", mu, 1, size=10)
            sigma = pm.HalfNormal("sigma", 1)
            obs = pm.Normal("obs", nested_mu, sigma, observed=y, shape=nested_mu.shape)

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[nested_mu, sigma],  # mu isn't in the trace and will be deemed volatile
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mu, nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"sigma"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=model.observed_RVs,
            vars_in_trace=[mu, nested_mu, sigma],
            basic_rvs=model.basic_RVs,
            givens_dict={
                mu: np.array(1.0)
            },  # mu will be considered volatile because it's in givens
        )
        assert volatile_rvs == {nested_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma"}

    def test_mixture(self):
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(3), size=(5, 3))

            mu = pm.Normal("mu", mu=np.arange(3), sigma=1)

            components = pm.Normal.dist(mu=mu, sigma=1, size=w.shape)
            mix_mu = pm.Mixture("mix_mu", w=w, comp_dists=components)
            obs = pm.Normal("obs", mix_mu, 1, observed=np.ones((5, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu", "mix_mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mix_mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu, w],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"w", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"w", "mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mix_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {w, mix_mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

    def test_censored(self):
        with pm.Model() as model:
            latent_mu = pm.Normal("latent_mu", mu=np.arange(3), sigma=1)
            mu = pm.Censored("mu", pm.Normal.dist(mu=latent_mu, sigma=1), lower=-1, upper=1)
            obs = pm.Normal("obs", mu, 1, observed=np.ones((10, 3)))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[latent_mu, mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"latent_mu", "mu"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[mu],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {latent_mu, mu, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_lkj_cholesky_cov(self):
        with pm.Model() as model:
            mu = np.zeros(3)
            sd_dist = pm.Exponential.dist(1.0, size=3)
            chol, corr, stds = pm.LKJCholeskyCov(  # pylint: disable=unpacking-non-sequence
                "chol_packed", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            chol_packed = model["chol_packed"]
            chol = pm.Deterministic("chol", chol)
            obs = pm.MvNormal("obs", mu=mu, chol=chol, observed=np.zeros(3))

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed, chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed", "chol"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol_packed],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {obs}
        assert {i.name for i in self.get_function_inputs(f)} == {"chol_packed"}
        assert {i.name for i in self.get_function_roots(f)} == {"chol_packed"}

        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[chol],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {chol_packed, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_non_random_model_variable(self):
        with pm.Model() as model:
            # A user may register non-pure RandomVariables that can nevertheless be
            # sampled, as long as a custom logprob is dispatched or Aeppl can infer
            # its logprob (which is the case for `clip`)
            y = at.clip(pm.Normal.dist(), -1, 1)
            y = model.register_rv(y, name="y")
            y_abs = pm.Deterministic("y_abs", at.abs(y))
            obs = pm.Normal("obs", y_abs, observed=np.zeros(10))

        # y_abs should be resampled even if in the trace, because the source y is missing
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[obs],
            vars_in_trace=[y_abs],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, obs}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == set()

    def test_mutable_coords_volatile(self):
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=1, scale=0.2, size=(10, 3))
        with pm.Model() as model:
            model.add_coord("name", ["A", "B", "C"], mutable=True)
            model.add_coord("obs", list(range(10, 20)), mutable=True)
            offsets = pm.MutableData("offsets", rng.normal(0, 1, size=(10,)))
            a = pm.Normal("a", mu=0, sigma=1, dims=["name"])
            b = pm.Normal("b", mu=offsets, sigma=1)
            mu = pm.Deterministic("mu", a + b[..., None], dims=["obs", "name"])
            sigma = pm.HalfNormal("sigma", sigma=1, dims=["name"])

            data = pm.MutableData(
                "y_obs",
                data,
                dims=["obs", "name"],
            )
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=data, dims=["obs", "name"])

        # When no constant_data and constant_coords, all the dependent nodes will be volatile and
        # resampled
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
        )
        assert volatile_rvs == {y, a, b, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == set()
        assert {i.name for i in self.get_function_roots(f)} == {"name", "obs", "offsets"}

        # When the constant data has the same values as the shared data, offsets wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
        )
        assert volatile_rvs == {y, a, sigma}
        assert {i.name for i in self.get_function_inputs(f)} == {"b"}
        assert {i.name for i in self.get_function_roots(f)} == {"b", "name", "obs"}

        # When we declare constant_coords, the shared variables with matching names wont be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }

        # When we have both constant_data and constant_coords, only y will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value()},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "b", "mu", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {"mu", "sigma", "name", "obs"}

        # When constant_data has different values than the shared variable, then
        # offsets will be volatile
        f, volatile_rvs = compile_forward_sampling_function(
            outputs=[y],
            vars_in_trace=[a, b, mu, sigma],
            basic_rvs=model.basic_RVs,
            constant_data={"offsets": offsets.get_value() + 1},
            constant_coords={"name", "obs"},
        )
        assert volatile_rvs == {y, b}
        assert {i.name for i in self.get_function_inputs(f)} == {"a", "sigma"}
        assert {i.name for i in self.get_function_roots(f)} == {
            "a",
            "sigma",
            "name",
            "obs",
            "offsets",
        }


def test_get_seeds_per_chain():
    ret = _get_seeds_per_chain(None, chains=1)
    assert len(ret) == 1 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(None, chains=2)
    assert len(ret) == 2 and isinstance(ret[0], int)

    ret = _get_seeds_per_chain(5, chains=1)
    assert ret == (5,)

    ret = _get_seeds_per_chain(5, chains=3)
    assert len(ret) == 3 and isinstance(ret[0], int) and not any(r == 5 for r in ret)

    rng = np.random.default_rng(123)
    expected_ret = rng.integers(2**30, dtype=np.int64, size=1)
    rng = np.random.default_rng(123)
    ret = _get_seeds_per_chain(rng, chains=1)
    assert ret == expected_ret

    rng = np.random.RandomState(456)
    expected_ret = rng.randint(2**30, dtype=np.int64, size=2)
    rng = np.random.RandomState(456)
    ret = _get_seeds_per_chain(rng, chains=2)
    assert np.all(ret == expected_ret)

    for expected_ret in ([0, 1, 2], (0, 1, 2, 3), np.arange(5)):
        ret = _get_seeds_per_chain(expected_ret, chains=len(expected_ret))
        assert ret is expected_ret

        with pytest.raises(ValueError, match="does not match the number of chains"):
            _get_seeds_per_chain(expected_ret, chains=len(expected_ret) + 1)

    with pytest.raises(ValueError, match=re.escape("The `seeds` must be array-like")):
        _get_seeds_per_chain({1: 1, 2: 2}, 2)


def test_get_vars_in_point_list():
    with pm.Model() as modelA:
        pm.Normal("a", 0, 1)
        pm.Normal("b", 0, 1)
    with pm.Model() as modelB:
        a = pm.Normal("a", 0, 1)
        pm.Normal("c", 0, 1)

    point_list = [{"a": 0, "b": 0}]
    vars_in_trace = get_vars_in_point_list(point_list, modelB)
    assert set(vars_in_trace) == {a}

    strace = pm.backends.NDArray(model=modelB, vars=modelA.free_RVs)
    strace.setup(1, 1)
    strace.values = point_list[0]
    strace.draw_idx = 1
    trace = MultiTrace([strace])
    vars_in_trace = get_vars_in_point_list(trace, modelB)
    assert set(vars_in_trace) == {a}
