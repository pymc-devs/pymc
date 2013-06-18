from pymc.tuning import scaling
from import models


a = np.array([-10, -.01, 0, 10, 1e300, -inf, inf])
def test_adjust_precision():
    a1 = adjust_precision(a)

    assert all(a1 > 0 && a1 < 1e200)

def test_guess_scaling():

    model = models.non_normal(n = 5)

    assert all(a1 > 0 && a1 < 1e200)
