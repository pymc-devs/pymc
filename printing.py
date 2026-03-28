# pymc/printing.py

"""Helper utilities for debugging PyMC models."""

from pytensor.printing import Print

__all__ = ["print_value"]


def print_value(var, name=None):
    """Print the value of a variable each time it is computed during sampling.

    This wraps the variable in a ``pytensor.printing.Print`` op, which is a
    pass-through that prints the variable's value as a side effect whenever it
    is evaluated.

    .. warning::
        This may significantly affect sampling performance. Use only for
        debugging purposes.

    Parameters
    ----------
    var : TensorVariable
        The PyMC variable to debug-print.
    name : str, optional
        Label shown in the printed output. Defaults to ``var.name``.

    Returns
    -------
    TensorVariable
        The same variable, wrapped in a Print op (value is unchanged).

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        with pm.Model():
            mu = pm.Normal("mu", mu=0, sigma=1)
            mu = pm.print_value(mu, name="mu")   # prints mu during sampling
            obs = pm.Normal("obs", mu=mu, sigma=1, observed=[1, 2, 3])
            idata = pm.sample()
    """
    if name is None:
        name = getattr(var, "name", "value")
    return Print(name)(var)