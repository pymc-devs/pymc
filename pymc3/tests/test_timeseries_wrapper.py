import aesara
import aesara.tensor as aet
import numpy as np
import pytest
import statsmodels.api as sm

from pymc3.timeseries.wrapper import Loglike, Score


@pytest.fixture()
def mock_model():
    n_obs = 100
    np.random.seed(7)

    # simulate random walk
    mock_timeseries = np.random.normal(size=n_obs).cumsum()

    # model initialization
    mock_model = sm.tsa.statespace.SARIMAX(mock_timeseries, order=(1, 0, 1))

    return mock_model


def test_loglike(mock_model):
    """
    Test that the LogLike class produces the same log-likelihood as
    produced by statsmodels.
    """
    inputs = aet.dvector()
    func_Loglike = aesara.function([inputs], Loglike(mock_model)(inputs))

    fit = mock_model.fit(disp=False)
    theta = np.array([fit._params_ar[0], fit._params_ma[0], fit._params_variance[0]])

    actual = func_Loglike(theta)
    expected = mock_model.loglike(theta)
    assert np.allclose(actual, expected)


def test_score(mock_model):
    """
    Test that the same derivatives are produced by the Score class as
    are produced by statsmodels.
    """
    inputs = aet.dvector()
    func_Score = aesara.function([inputs], Score(mock_model)(inputs))

    fit = mock_model.fit(disp=False)
    theta = np.array([fit._params_ar[0], fit._params_ma[0], fit._params_variance[0]])

    actual = func_Score(theta)
    expected = mock_model.score(theta)
    assert np.allclose(actual, expected)


# todo add test for gradient
