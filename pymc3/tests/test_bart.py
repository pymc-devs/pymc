import numpy as np

import pymc3 as pm


def test_split_node():
    split_node = pm.distributions.bart.SplitNode(index=5, idx_split_variable=2, split_value=3.0)
    assert split_node.index == 5
    assert split_node.idx_split_variable == 2
    assert split_node.split_value == 3.0
    assert split_node.depth == 2
    assert split_node.get_idx_parent_node() == 2
    assert split_node.get_idx_left_child() == 11
    assert split_node.get_idx_right_child() == 12


def test_leaf_node():
    leaf_node = pm.distributions.bart.LeafNode(index=5, value=3.14, idx_data_points=[1, 2, 3])
    assert leaf_node.index == 5
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert leaf_node.value == 3.14
    assert leaf_node.get_idx_parent_node() == 2
    assert leaf_node.get_idx_left_child() == 11
    assert leaf_node.get_idx_right_child() == 12


def test_bart():
    X = np.random.sample(size=(100, 4))
    Y = np.random.sample(size=(100))

    with pm.Model():
        b = pm.BART("b", X=X, Y=Y)

    bart = b.distribution
    assert bart.num_observations == 100
    assert bart.num_variates == 4
    assert len(bart.trees) == bart.m

    assert np.array_equal(bart.trees[0][0].idx_data_points, np.array(range(100), dtype="int32"))


def test_available_splitting_rules():
    X = np.array([[1.0, 2.0, 3.0, np.nan], [2.0, 2.0, 3.0, 99.9], [3.0, 4.0, 3.0, -3.3]])
    Y = np.random.sample(size=(3))
    with pm.Model():
        b = pm.BART("b", X=X, Y=Y)

    bart = b.distribution

    idx_split_variable = 0
    idx_data_points = np.array(range(bart.num_observations), dtype="int32")
    available_splitting_rules = bart.get_available_splitting_rules(
        idx_data_points, idx_split_variable
    )
    assert available_splitting_rules.size == 2
    np.testing.assert_almost_equal(available_splitting_rules, np.array([1.0, 2.0]), 1)

    idx_split_variable = 1
    idx_data_points = np.array(range(bart.num_observations), dtype="int32")
    available_splitting_rules = bart.get_available_splitting_rules(
        idx_data_points, idx_split_variable
    )
    assert available_splitting_rules.size == 1
    np.testing.assert_almost_equal(
        available_splitting_rules,
        np.array(
            [
                2.0,
            ]
        ),
        1,
    )

    idx_split_variable = 2
    idx_data_points = np.array(range(bart.num_observations), dtype="int32")
    available_splitting_rules = bart.get_available_splitting_rules(
        idx_data_points, idx_split_variable
    )
    assert available_splitting_rules.size == 0
    np.testing.assert_almost_equal(available_splitting_rules, np.array([]))

    idx_split_variable = 3
    idx_data_points = np.array(range(bart.num_observations), dtype="int32")
    available_splitting_rules = bart.get_available_splitting_rules(
        idx_data_points, idx_split_variable
    )
    assert available_splitting_rules.size == 1
    np.testing.assert_almost_equal(available_splitting_rules, np.array([-3.3]), 1)


def test_model():
    np.random.seed(212480)
    X = np.linspace(7, 15, 100)
    Y = np.sin(np.random.normal(X, 0.2)) + 3
    X = X[:, None]

    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 1)
        mu = pm.BART("mu", X, Y, m=50)
        y = pm.Normal("y", mu, sigma, observed=Y)
        trace = pm.sample(1000, random_seed=212480, return_inferencedata=False)

    np.testing.assert_allclose(trace[mu].mean(0), Y, 0.5)

    Y = np.repeat([0, 1], 50)
    with pm.Model() as model:
        mu = pm.BART("mu", X, Y, m=50, inv_link="logistic")
        y = pm.Bernoulli("y", mu, observed=Y)
        trace = pm.sample(1000, random_seed=212480, return_inferencedata=False)

    np.testing.assert_allclose(trace[mu].mean(0), Y, atol=0.5)
