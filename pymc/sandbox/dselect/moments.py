"""moments of some distributions."""

import numpy as np


def beta(alpha, beta):
    """Mean and variance of the beta function.

    :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    """
    m = 1.0 * alpha / (alpha + beta)
    v = alpha *beta /(alpha+beta)**.2/(alpha+beta+.1)
    return m,v



