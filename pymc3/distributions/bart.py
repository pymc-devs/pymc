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
    name = "bart"
    ndim_supp = 1
    ndims_params = [2, 1, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("BART", "\\operatorname{BART}")
    all_trees = None

    def _shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        return default_shape_from_params(self.ndim_supp, dist_params, rep_param_idx, param_shapes)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    @classmethod
    def rng_fn(cls, X_new=None, *args, **kwargs):
        # in old version X_new could be passed by the user to get out-of-sample predictions

        all_trees = cls.all_trees
        if all_trees:
            pred = 0
            idx = np.random.randint(len(all_trees))
            trees = all_trees[idx]
            if X_new is None:
                for tree in trees:
                    pred += tree.predict_output()
            else:
                for tree in trees:
                    pred += np.array([tree.predict_out_of_sample(x) for x in X_new])
            return pred
            # XXX check why I did not make this random
            # pred = np.zeros((len(trees), num_observations))
            # for draw, trees_to_sum in enumerate(trees):
            #    new_Y = np.zeros(num_observations)
            #    for tree in trees_to_sum:
            #        new_Y += [tree.predict_out_of_sample(x) for x in X_new]
            #    pred[draw] = new_Y
            # return pred
        else:
            return np.full_like(cls.Y, cls.Y.mean())


bart = BARTRV()


class BART(NoDistribution):
    """Improper flat prior over the positive reals."""

    def __new__(
        cls,
        name,
        X,
        Y,
        m=50,
        alpha=0.25,
        k=2,
        ndim_supp=1,
        ndims_params=[2, 1, 0, 0, 0],
        dtype="floatX",
        **kwargs,
    ):

        if X.ndim != 2:
            raise ValueError("The design matrix X must have two dimensions")

        if Y.ndim != 1:
            raise ValueError("The response matrix Y must have one dimension")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "The design matrix X and the response matrix Y must have the same number of elements"
            )
        if not isinstance(m, int):
            raise ValueError("The number of trees m type must be int")
        if m < 1:
            raise ValueError("The number of trees m must be greater than zero")

        if alpha <= 0 or 1 <= alpha:
            raise ValueError(
                "The value for the alpha parameter for the tree structure "
                "must be in the interval (0, 1)"
            )

        cls.all_trees = []
        bart_op = type(
            f"BART_{name}",
            (BARTRV,),
            dict(
                name="BART",
                all_trees=cls.all_trees,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                dtype=dtype,
                inplace=False,
                X=X,
                Y=Y,
                m=m,
                alpha=alpha,
                k=k,
            ),
        )()

        NoDistribution.register(BARTRV)

        cls.rv_op = bart_op
        params = [X, Y, m, alpha, k]
        return super().__new__(cls, name, *params, **kwargs)

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)
