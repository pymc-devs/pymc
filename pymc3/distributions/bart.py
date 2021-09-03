#   Copyright 2020 The PyMC Developers
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

import numpy as np

from aesara.tensor.random.op import RandomVariable, default_shape_from_params

from pymc3.distributions.distribution import NoDistribution

__all__ = ["BART"]


class BARTRV(RandomVariable):
    """
    Base class for BART
    """

    name = "BART"
    ndim_supp = 1
    ndims_params = [2, 1, 0, 0, 0, 1]
    dtype = "floatX"
    _print_name = ("BART", "\\operatorname{BART}")
    all_trees = None

    def _shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        return default_shape_from_params(self.ndim_supp, dist_params, rep_param_idx, param_shapes)

    @classmethod
    def rng_fn(cls, rng=np.random.default_rng(), *args, **kwargs):
        size = kwargs.pop("size", None)
        X_new = kwargs.pop("X_new", None)
        all_trees = cls.all_trees
        if all_trees:

            if size is None:
                size = ()
            elif isinstance(size, int):
                size = [size]

            flatten_size = 1
            for s in size:
                flatten_size *= s

            idx = rng.randint(len(all_trees), size=flatten_size)

            if X_new is None:
                pred = np.zeros((flatten_size, all_trees[0][0].num_observations))
                for ind, p in enumerate(pred):
                    for tree in all_trees[idx[ind]]:
                        p += tree.predict_output()
            else:
                pred = np.zeros((flatten_size, X_new.shape[0]))
                for ind, p in enumerate(pred):
                    for tree in all_trees[idx[ind]]:
                        p += np.array([tree.predict_out_of_sample(x) for x in X_new])
            return pred.reshape((*size, -1))
        else:
            return np.full_like(cls.Y, cls.Y.mean())


bart = BARTRV()


class BART(NoDistribution):
    """
    Bayesian Additive Regression Tree distribution.

    Distribution representing a sum over trees

    Parameters
    ----------
    X : array-like
        The covariate matrix.
    Y : array-like
        The response vector.
    m : int
        Number of trees
    alpha : float
        Control the prior probability over the depth of the trees. Even when it can takes values in
        the interval (0, 1), it is recommended to be in the interval (0, 0.5].
    k : float
        Scale parameter for the values of the leaf nodes. Defaults to 2. Recomended to be between 1
        and 3.
    split_prior : array-like
        Each element of split_prior should be in the [0, 1] interval and the elements should sum to
        1. Otherwise they will be normalized.
        Defaults to None, i.e. all covariates have the same prior probability to be selected.
    """

    def __new__(
        cls,
        name,
        X,
        Y,
        m=50,
        alpha=0.25,
        k=2,
        split_prior=None,
        **kwargs,
    ):

        cls.all_trees = []

        bart_op = type(
            f"BART_{name}",
            (BARTRV,),
            dict(
                name="BART",
                all_trees=cls.all_trees,
                inplace=False,
                initval=Y.mean(),
                X=X,
                Y=Y,
                m=m,
                alpha=alpha,
                k=k,
                split_prior=split_prior,
            ),
        )()

        NoDistribution.register(BARTRV)

        cls.rv_op = bart_op
        params = [X, Y, m, alpha, k]
        return super().__new__(cls, name, *params, **kwargs)

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)
