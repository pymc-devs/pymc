import numpy as np

import pymc3 as pm


def test_split_node():
    split_node = pm.distributions.tree.SplitNode(index=5, idx_split_variable=2, split_value=3.0)
    assert split_node.index == 5
    assert split_node.idx_split_variable == 2
    assert split_node.split_value == 3.0
    assert split_node.depth == 2
    assert split_node.get_idx_parent_node() == 2
    assert split_node.get_idx_left_child() == 11
    assert split_node.get_idx_right_child() == 12


def test_leaf_node():
    leaf_node = pm.distributions.tree.LeafNode(index=5, value=3.14, idx_data_points=[1, 2, 3])
    assert leaf_node.index == 5
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert leaf_node.value == 3.14
    assert leaf_node.get_idx_parent_node() == 2
    assert leaf_node.get_idx_left_child() == 11
    assert leaf_node.get_idx_right_child() == 12


def test_model():
    X = np.linspace(7, 15, 100)
    Y = np.sin(np.random.normal(X, 0.2)) + 3
    X = X[:, None]

    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.BART("mu", X, Y, m=50)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample()
        mean = idata.posterior["mu"].stack(samples=("chain", "draw")).mean("samples")

    np.testing.assert_allclose(mean, Y, 0.5)

    Y = np.repeat([0, 1], 50)
    with pm.Model() as model:
        mu_ = pm.BART("mu_", X, Y, m=50)
        mu = pm.Deterministic("mu", pm.math.invlogit(mu_))
        y = pm.Bernoulli("y", mu, observed=Y)
        idata = pm.sample()
        mean = idata.posterior["mu"].stack(samples=("chain", "draw")).mean("samples")

    np.testing.assert_allclose(mean, Y, atol=0.5)


def test_bart_vi():
    X = np.random.normal(0, 1, size=(3, 250)).T
    Y = np.random.normal(0, 1, size=250)
    X[:, 0] = np.random.normal(Y, 0.1)

    with pm.Model() as model:
        mu = pm.BART("mu", X, Y, m=10)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(random_seed=3415, chains=1)
        var_imp = (
            idata.sample_stats["variable_inclusion"]
            .stack(samples=("chain", "draw"))
            .mean("samples")
        )
        var_imp /= var_imp.sum()
        assert var_imp[0] > var_imp[1:].sum()
        np.testing.assert_almost_equal(var_imp.sum(), 1)
