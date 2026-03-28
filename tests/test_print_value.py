# tests/test_printing.py

import pymc as pm
import pytensor.tensor as pt
from pytensor.printing import Print
from pymc.printing import print_value


def test_print_value_is_passthrough():
    """print_value should not change the variable's value."""
    x = pt.vector("x")
    x_printed = print_value(x, name="test_x")

    import pytensor
    import numpy as np

    f = pytensor.function([x], x_printed)
    result = f([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_print_value_default_name():
    """print_value should use var.name if no name is given."""
    x = pt.vector("my_var")
    x_printed = print_value(x)
    # The Print op's message should match the variable name
    assert x_printed.owner.op.message == "my_var"


def test_print_value_custom_name():
    """print_value should use the custom name when provided."""
    x = pt.vector("x")
    x_printed = print_value(x, name="custom_label")
    assert x_printed.owner.op.message == "custom_label"


def test_print_value_accessible_from_pm():
    """print_value should be accessible as pm.print_value."""
    assert hasattr(pm, "print_value")
