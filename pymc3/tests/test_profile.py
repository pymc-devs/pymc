import pymc3 as pm
from .models import simple_model


def test_profile_model():
    start, model, _ = simple_model()

    assert model.profile(model.logpt).fct_call_time > 0


def test_profile_variable():
    start, model, _ = simple_model()

    assert model.profile(model.vars[0].logpt).fct_call_time > 0


def test_profile_count():
    start, model, _ = simple_model()

    assert model.profile(model.logpt, n=1005).fct_callcount == 1005
