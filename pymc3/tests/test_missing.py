from pymc3 import Model, Normal
from numpy import ma
import numpy
import pandas as pd


def test_missing():
    data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
    with Model() as model:
        x = Normal('x', 1, 1)
        Normal('y', x, 1, observed=data)

    y_missing, = model.missing_values
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)


def test_missing_pandas():
    data = pd.DataFrame([1, 2, numpy.nan, 4, numpy.nan])
    with Model() as model:
        x = Normal('x', 1, 1)
        Normal('y', x, 1, observed=data)

    y_missing, = model.missing_values
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)
