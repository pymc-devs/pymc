import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.compile.mode import get_default_mode

from aeppl.dists import dirac_delta


def test_dirac_delta():
    fn = aesara.function(
        [], dirac_delta(at.as_tensor(1)), mode=get_default_mode().excluding("useless")
    )
    with pytest.warns(UserWarning, match=".*DiracDelta.*"):
        assert np.array_equal(fn(), 1)
