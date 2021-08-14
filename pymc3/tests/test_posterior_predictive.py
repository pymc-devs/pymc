import numpy as np

import pymc3 as pm

from pymc3.backends.ndarray import point_list_to_multitrace
from pymc3.distributions.posterior_predictive import _TraceDict


def test_translate_point_list():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
        mt = point_list_to_multitrace([model.test_point], model)
        assert isinstance(mt, pm.backends.base.MultiTrace)
        assert {"mu"} == set(mt.varnames)
        assert len(mt) == 1


def test_build_TraceDict():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
        trace = pm.sample(chains=2, draws=500, return_inferencedata=False)
        dict = _TraceDict(multi_trace=trace)
        assert isinstance(dict, _TraceDict)
        assert len(dict) == 1000
        np.testing.assert_array_equal(trace["mu"], dict["mu"])
        assert set(trace.varnames) == set(dict.varnames) == {"mu"}


def test_build_TraceDict_point_list():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=np.array([0.5, 0.2]))
        dict = _TraceDict(point_list=[model.test_point])
        assert set(dict.varnames) == {"mu"}
        assert len(dict) == 1
        assert len(dict["mu"]) == 1
        assert dict["mu"][0] == 0.0


def test_fast_sample_posterior_predictive_shape_assertions():
    """
    This test checks the shape assertions in pm.fast_sample_posterior_predictive.
    Originally reported - https://github.com/pymc-devs/pymc3/issues/4778
    """
    with pm.Model():
        p = pm.Beta("p", 2, 2)
        trace = pm.sample(
            tune=30, draws=50, chains=1, return_inferencedata=True, compute_convergence_checks=False
        )

    with pm.Model() as m_forward:
        p = pm.Beta("p", 2, 2)
        b2 = pm.Binomial("b2", n=1, p=p)
        b3 = pm.Binomial("b3", n=1, p=p * b2)

    with m_forward:
        trace_forward = pm.fast_sample_posterior_predictive(trace, var_names=["p", "b2", "b3"])

    for free_rv in trace_forward.values():
        assert free_rv.shape[0] == 50
