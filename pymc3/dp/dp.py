#   Copyright 2021 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import pymc3 as pm

from pymc3.distributions.distribution import Distribution

__all__ = []


class Base(Distribution):
    R"""
    Base class
    """

    def __init__(self, alpha=1, base_dist=pm.Normal):

        if not (isinstance(base_dist, Distribution)):
            raise TypeError(
                f"Supplied base_dist must be a " "Distribution but got {type(base_dist)}" "instead."
            )
        self.alpha = alpha
        self.base_dist = base_dist

    def __add__(self, other):
        return self.__class__(self.alpha, self.base_dist)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError


class DirichletProcess(Base):
    pass
