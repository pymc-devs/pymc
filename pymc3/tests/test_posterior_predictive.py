import pymc3 as pm
import pytest

from pymc3.backends.ndarray import point_list_to_multitrace

def test_translate_point_list():
   with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        a = pm.Normal("a", mu=mu, sigma=1, observed=0.0)
        mt = point_list_to_multitrace([model.test_point], model)
        assert isinstance(mt, pm.backends.base.MultiTrace)
        assert set(["mu"]) == set(mt.varnames)
        assert len(mt) == 1
