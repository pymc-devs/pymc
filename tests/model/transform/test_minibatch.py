#   Copyright 2025 - present The PyMC Developers
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

from pymc.data import Data, Minibatch
from pymc.distributions import HalfNormal, Normal
from pymc.model.core import Model
from pymc.model.transform.minibatch import minibatch_model, remove_minibatch
from pymc.variational.minibatch_rv import MinibatchRandomVariable


def test_minibatch_model():
    data_size = 100
    n_features = 4

    obs_data_np = np.random.normal(size=(data_size,))
    X_data_np = np.random.normal(size=(data_size, n_features))

    with Model(coords={"feature": range(n_features), "data_dim": range(data_size)}) as m:
        obs_data = Data("obs_data", obs_data_np, dims=["data_dim"])
        X_data = Data("X_data", X_data_np, dims=["data_dim", "feature"])
        beta = Normal("beta", mu=np.pi, dims="feature")

        mu = X_data @ beta
        y = Normal("y", mu=mu, sigma=1, observed=obs_data, dims="data_dim")

    with Model(coords={"feature": range(n_features), "data_dim": range(data_size)}) as ref_m:
        obs_data = Data("obs_data", obs_data_np, dims=["data_dim"])
        X_data = Data("X_data", X_data_np, dims=["data_dim", "feature"])
        minibatch_obs_data, minibatch_X_data = Minibatch(obs_data, X_data, batch_size=10)
        beta = Normal("beta", mu=np.pi, dims="feature")
        mu = minibatch_X_data @ beta
        y = Normal(
            "y",
            mu=mu,
            sigma=1,
            observed=minibatch_obs_data,
            dims="data_dim",
            total_size=(obs_data.shape[0], ...),
        )

    mb = minibatch_model(m, batch_size=10)
    mb_logp_fn = mb.compile_logp(random_seed=42)
    ref_mb_logp_fn = ref_m.compile_logp(random_seed=42)
    ip = mb.initial_point()

    mb_res1 = mb_logp_fn(ip)
    ref_mb_res1 = ref_mb_logp_fn(ip)
    np.testing.assert_allclose(mb_res1, ref_mb_res1)
    mb_res2 = mb_logp_fn(ip)
    # Minibatch should give different results on each call
    assert mb_res1 != mb_res2
    ref_mb_res2 = ref_mb_logp_fn(ip)
    np.testing.assert_allclose(mb_res2, ref_mb_res2)


def test_remove_minibatch():
    data_size = 100
    n_features = 5
    batch_size = 10
    with Model(coords={"d": range(n_features)}) as mb:
        X_data = Data("X_data", np.random.normal(size=(data_size, n_features)))
        obs_data = Data("obs_data", [1, 2, 3, 4, 5])
        minibatch_X_data, minibatch_obs_data = Minibatch(X_data, obs_data, batch_size=batch_size)

        beta = Normal("beta", dims=("d",))
        mu = minibatch_X_data @ beta
        sigma = HalfNormal("sigma")
        y = Normal("y", mu=mu, sigma=sigma, observed=minibatch_obs_data, total_size=X_data.shape[0])

    m = remove_minibatch(mb)
    assert isinstance(mb.y.owner.op, MinibatchRandomVariable)
    assert tuple(mb.y.shape).eval() == (batch_size,)
    assert isinstance(m.y.owner.op, Normal)
    assert tuple(m.y.shape.eval()) == (data_size,)
    assert mb.coords == m.coords
    assert mb.dim_lengths["d"].eval() == m.dim_lengths["d"].eval()


@pytest.mark.parametrize("static_shape", (True, False))
def test_minibatch_transform_roundtrip(static_shape):
    data_size = 100
    n_features = 4
    with Model(coords={"feature": range(n_features), "data_dim": range(data_size)}) as m:
        obs_data = Data(
            "obs_data",
            np.random.normal(size=(data_size,)),
            dims=["data_dim"],
            shape=(data_size if static_shape else None,),
        )
        X_data = Data(
            "X_data",
            np.random.normal(size=(data_size, n_features)),
            dims=["data_dim", "feature"],
            shape=(data_size if static_shape else None, n_features),
        )
        beta = Normal("beta", mu=np.pi, dims="feature")

        mu = X_data @ beta
        y = Normal("y", mu=mu, sigma=1, observed=obs_data, dims="data_dim")

    m_again = remove_minibatch(minibatch_model(m, batch_size=10))
    m_again_logp_fn = m_again.compile_logp(random_seed=42)
    m_logp_fn = m_again.compile_logp(random_seed=42)
    ip = m_again.initial_point()
    m_again_res = m_again_logp_fn(ip)
    m_res = m_logp_fn(ip)
    np.testing.assert_allclose(m_again_res, m_res)
    # Check that repeated calls give the same result (no more minibatching)
    np.testing.assert_allclose(m_again_res, m_again_logp_fn(ip))
