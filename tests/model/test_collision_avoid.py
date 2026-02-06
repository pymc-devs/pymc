import pymc as pm
import pytest


def test_coord_name_conflicts_with_variable_name():
    with pytest.raises(ValueError, match="conflicts"):
        with pm.Model(coords={"a": [0, 1]}):
            pm.Data("a", [5, 10])


def test_variable_name_conflicts_with_coord_name():
    with pytest.raises(ValueError, match="conflicts"):
        with pm.Model() as m:
            pm.Data("a", [5, 10])
            m.add_coord("a", [0, 1])
