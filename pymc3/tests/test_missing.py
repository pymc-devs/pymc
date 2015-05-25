from pymc3 import Model, Normal, Metropolis, MvNormal
from numpy import ma

def test_missing():
    data = ma.masked_values([1,2,-1,4,-1], value=-1)
    with Model() as model:
        x = Normal('x', 1, 1)
        y = Normal('y', x, 1, observed=data)

    y_missing, = model.missing_values 
    assert y_missing.tag.test_value.shape == (2,)

    model.logp(model.test_point)


