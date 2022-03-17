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

import aesara
import numpy as np

from aesara import tensor as at

from pymc.distributions.dist_math import rho2sigma
from pymc.util import WithMemoization
from pymc.variational import opvi
from pymc.variational.opvi import collect_shared_to_list, node_property

__all__ = ["Formula", "PlanarFlow", "HouseholderFlow", "RadialFlow", "LocFlow", "ScaleFlow"]


class Formula:
    """
    Helpful class to use string like formulas with
    __call__ syntax similar to Flow.__init__

    Parameters
    ----------
    formula: str
        string representing normalizing flow
        e.g. 'planar', 'planar*4', 'planar*4-radial*3', 'planar-radial-planar'
        Yet simple pattern is supported:

            1. dash separated flow identifiers
            2. star for replication after flow identifier

    Methods
    -------
    __call__(z0, dim, jitter) - initializes and links all flows returning the last one
    """

    def __init__(self, formula):
        identifiers = formula.lower().replace(" ", "").split("-")
        self.formula = "-".join(identifiers)
        identifiers = [idf.split("*") for idf in identifiers]
        self.flows = []

        for tup in identifiers:
            if len(tup) == 1:
                self.flows.append(flow_for_short_name(tup[0]))
            elif len(tup) == 2:
                self.flows.extend([flow_for_short_name(tup[0])] * int(tup[1]))
            else:
                raise ValueError("Wrong format: %s" % formula)
        if len(self.flows) == 0:
            raise ValueError("No flows in formula")

    def __call__(self, z0=None, dim=None, jitter=0.001, params=None, batch_size=None):
        if len(self.flows) == 0:
            raise ValueError("No flows in formula")
        if params is None:
            params = dict()
        flow = z0
        for i, flow_cls in enumerate(self.flows):
            flow = flow_cls(
                dim=dim, jitter=jitter, z0=flow, batch_size=batch_size, **params.get(i, {})
            )
        return flow

    def __reduce__(self):
        return self.__class__, self.formula

    def __latex__(self):
        return r"Formula{\mathcal{N}(0, 1) -> %s}" % self.formula

    __repr__ = _latex_repr_ = __latex__

    def get_param_spec_for(self, **kwargs):
        res = dict()
        for i, cls in enumerate(self.flows):
            res[i] = cls.get_param_spec_for(**kwargs)
        return res


def seems_like_formula(formula):
    try:
        Formula(formula)
        return True
    except (ValueError, KeyError):
        return False


def seems_like_flow_params(params):
    if set(range(len(params))) == set(params):
        for p in params.values():
            try:
                flow_for_params(p)
            except KeyError:
                return False
        else:
            return True
    else:
        return False


class AbstractFlow(WithMemoization):
    shared_params = None
    __param_spec__ = dict()
    short_name = ""
    __param_registry = dict()
    __name_registry = dict()

    @classmethod
    def register(cls, sbcls):
        assert (
            frozenset(sbcls.__param_spec__) not in cls.__param_registry
        ), "Duplicate __param_spec__"
        cls.__param_registry[frozenset(sbcls.__param_spec__)] = sbcls
        assert sbcls.short_name not in cls.__name_registry, "Duplicate short_name"
        cls.__name_registry[sbcls.short_name] = sbcls
        return sbcls

    @classmethod
    def flow_for_params(cls, params):
        if frozenset(params) not in cls.__param_registry:
            raise KeyError(
                "No such flow for the following params: {!r}, "
                "only the following are supported\n\n{}".format(params, cls.__param_registry)
            )
        return cls.__param_registry[frozenset(params)]

    @classmethod
    def flow_for_short_name(cls, name):
        if name.lower() not in cls.__name_registry:
            raise KeyError(
                "No such flow: {!r}, "
                "only the following are supported\n\n{}".format(name, cls.__name_registry)
            )
        return cls.__name_registry[name.lower()]

    def __init__(self, z0=None, dim=None, jitter=0.001, batch_size=None, local=False):
        self.local = local
        self.batch_size = batch_size
        self.__jitter = jitter
        if isinstance(z0, AbstractFlow):
            parent = z0
            dim = parent.dim
            z0 = parent.forward
        else:
            parent = None
        if dim is not None:
            self.dim = dim
        else:
            raise ValueError(
                "Cannot infer dimension of flow, " "please provide dim or Flow instance as z0"
            )
        if z0 is None:
            self.z0 = at.matrix()  # type: TensorVariable
        else:
            self.z0 = at.as_tensor(z0)
        self.parent = parent

    def add_param(self, user=None, name=None, ref=0.0, dtype="floatX"):
        if dtype == "floatX":
            dtype = aesara.config.floatX
        spec = self.__param_spec__[name]
        shape = tuple(eval(s, {"d": self.dim}) for s in spec)
        if user is None:
            if self.local:
                raise opvi.LocalGroupError("Need parameters for local group flow")
            if self.batched:
                if self.batch_size is None:
                    raise opvi.BatchedGroupError("Need batch size to infer parameter shape")
                shape = (self.batch_size,) + shape
            return aesara.shared(
                np.asarray(np.random.normal(size=shape) * self.__jitter + ref).astype(dtype),
                name=name,
            )

        else:
            if self.batched:
                if self.local or self.batch_size is None:
                    shape = (-1,) + shape
                else:
                    shape = (self.batch_size,) + shape
            return at.as_tensor(user).reshape(shape)

    @property
    def params(self):
        return collect_shared_to_list(self.shared_params)

    @property
    def all_params(self):
        params = self.params  # type: list
        current = self
        while not current.isroot:
            current = current.parent
            params.extend(current.params)
        return params

    @property
    @aesara.config.change_flags(compute_test_value="off")
    def sum_logdets(self):
        dets = [self.logdet]
        current = self
        while not current.isroot:
            current = current.parent
            dets.append(current.logdet)
        return at.add(*dets)

    @node_property
    def forward(self):
        raise NotImplementedError

    @node_property
    def logdet(self):
        raise NotImplementedError

    @aesara.config.change_flags(compute_test_value="off")
    def forward_pass(self, z0):
        ret = aesara.clone_replace(self.forward, {self.root.z0: z0})
        return ret

    __call__ = forward_pass

    @property
    def root(self):
        current = self
        while not current.isroot:
            current = current.parent
        return current

    @property
    def formula(self):
        f = self.short_name
        current = self
        while not current.isroot:
            current = current.parent
            f = current.short_name + "-" + f
        return f

    @property
    def isroot(self):
        return self.parent is None

    @property
    def batched(self):
        return self.z0.ndim == 3

    @classmethod
    def get_param_spec_for(cls, **kwargs):
        res = dict()
        for name, fshape in cls.__param_spec__.items():
            res[name] = tuple(eval(s, kwargs) for s in fshape)
        return res

    def __repr__(self):
        return "Flow{%s}" % self.short_name

    def __str__(self):
        return self.short_name


flow_for_params = AbstractFlow.flow_for_params
flow_for_short_name = AbstractFlow.flow_for_short_name


class FlowFn:
    @staticmethod
    def fn(*args):
        raise NotImplementedError

    @staticmethod
    def inv(*args):
        raise NotImplementedError

    @staticmethod
    def deriv(*args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.fn(*args)


class LinearFlow(AbstractFlow):
    __param_spec__ = dict(u=("d",), w=("d",), b=())

    @aesara.config.change_flags(compute_test_value="off")
    def __init__(self, h, u=None, w=None, b=None, **kwargs):
        self.h = h
        super().__init__(**kwargs)
        u = self.add_param(u, "u")
        w = self.add_param(w, "w")
        b = self.add_param(b, "b")
        self.shared_params = dict(u=u, w=w, b=b)
        self.u_, self.w_ = self.make_uw(self.u, self.w)

    u = property(lambda self: self.shared_params["u"])
    w = property(lambda self: self.shared_params["w"])
    b = property(lambda self: self.shared_params["b"])

    def make_uw(self, u, w):
        raise NotImplementedError("Need to implement valid U, W transform")

    @node_property
    def forward(self):
        z = self.z0  # sxd
        u = self.u_  # d
        w = self.w_  # d
        b = self.b  # .
        h = self.h  # f
        # h(sxd \dot d + .)  = s
        if not self.batched:
            hwz = h(z.dot(w) + b)  # s
            # sxd + (s \outer d) = sxd
            z1 = z + at.outer(hwz, u)  # sxd
            return z1
        else:
            z = z.swapaxes(0, 1)
            # z bxsxd
            # u bxd
            # w bxd
            b = b.dimshuffle(0, "x")
            # b bx-
            hwz = h(at.batched_dot(z, w) + b)  # bxs
            # bxsxd + (bxsx- * bx-xd) = bxsxd
            hwz = hwz.dimshuffle(0, 1, "x")  # bxsx-
            u = u.dimshuffle(0, "x", 1)  # bx-xd
            z1 = z + hwz * u  # bxsxd
            return z1.swapaxes(0, 1)  # sxbxd

    @node_property
    def logdet(self):
        z = self.z0  # sxd
        u = self.u_  # d
        w = self.w_  # d
        b = self.b  # .
        deriv = self.h.deriv  # f'
        if not self.batched:
            # f'(sxd \dot d + .) * -xd = sxd
            phi = deriv(z.dot(w) + b).dimshuffle(0, "x") * w.dimshuffle("x", 0)
            # \abs(. + sxd \dot d) = s
            det = at.abs_(1.0 + phi.dot(u))
            return at.log(det)
        else:
            z = z.swapaxes(0, 1)
            b = b.dimshuffle(0, "x")
            # z bxsxd
            # u bxd
            # w bxd
            # b bx-x-
            # f'(bxsxd \bdot bxd + bx-x-) * bx-xd = bxsxd
            phi = deriv(at.batched_dot(z, w) + b).dimshuffle(0, 1, "x") * w.dimshuffle(0, "x", 1)
            # \abs(. + bxsxd \bdot bxd) = bxs
            det = at.abs_(1.0 + at.batched_dot(phi, u))  # bxs
            return at.log(det).sum(0)  # s


class Tanh(FlowFn):
    fn = at.tanh
    inv = at.arctanh

    @staticmethod
    def deriv(*args):
        (x,) = args
        return 1.0 - at.tanh(x) ** 2


@AbstractFlow.register
class PlanarFlow(LinearFlow):
    short_name = "planar"

    def __init__(self, **kwargs):
        super().__init__(h=Tanh(), **kwargs)

    def make_uw(self, u, w):
        if not self.batched:
            # u_: d
            # w_: d
            wu = u.dot(w)  # .
            mwu = -1.0 + at.softplus(wu)  # .
            # d + (. - .) * d / .
            u_h = u + (mwu - wu) * w / ((w**2).sum() + 1e-10)
            return u_h, w
        else:
            # u_: bxd
            # w_: bxd
            wu = (u * w).sum(-1, keepdims=True)  # bx-
            mwu = -1.0 + at.softplus(wu)  # bx-
            # bxd + (bx- - bx-) * bxd / bx- = bxd
            u_h = u + (mwu - wu) * w / ((w**2).sum(-1, keepdims=True) + 1e-10)
            return u_h, w


class ReferencePointFlow(AbstractFlow):
    __param_spec__ = dict(a=(), b=(), z_ref=("d",))

    @aesara.config.change_flags(compute_test_value="off")
    def __init__(self, h, a=None, b=None, z_ref=None, **kwargs):
        super().__init__(**kwargs)
        a = self.add_param(a, "a")
        b = self.add_param(b, "b")
        if hasattr(self.z0, "tag") and hasattr(self.z0.tag, "test_value"):
            z_ref = self.add_param(
                z_ref, "z_ref", ref=self.z0.tag.test_value[0], dtype=self.z0.dtype
            )
        else:
            z_ref = self.add_param(z_ref, "z_ref", dtype=self.z0.dtype)
        self.h = h
        self.shared_params = dict(a=a, b=b, z_ref=z_ref)
        self.a_, self.b_ = self.make_ab(self.a, self.b)

    a = property(lambda self: self.shared_params["a"])
    b = property(lambda self: self.shared_params["b"])
    z_ref = property(lambda self: self.shared_params["z_ref"])

    def make_ab(self, a, b):
        raise NotImplementedError("Need to specify how to get a, b")

    @node_property
    def forward(self):
        a = self.a_  # .
        b = self.b_  # .
        z_ref = self.z_ref  # d
        z = self.z0  # sxd
        h = self.h  # h(a, r)
        if self.batched:
            # a bx-x-
            # b bx-x-
            # z bxsxd
            # z_ref bx-xd
            z = z.swapaxes(0, 1)
            a = a.dimshuffle(0, "x", "x")
            b = b.dimshuffle(0, "x", "x")
            z_ref = z_ref.dimshuffle(0, "x", 1)
        r = (z - z_ref).norm(2, axis=-1, keepdims=True)  # sx- (bxsx-)
        # global: sxd + . * h(., sx-) * (sxd - sxd) = sxd
        # local: bxsxd + b * h(b, bxsx-) * (bxsxd - bxsxd) = bxsxd
        z1 = z + b * h(a, r) * (z - z_ref)
        if self.batched:
            z1 = z1.swapaxes(0, 1)
        return z1

    @node_property
    def logdet(self):
        d = float(self.dim)
        a = self.a_  # .
        b = self.b_  # .
        z_ref = self.z_ref  # d
        z = self.z0  # sxd
        h = self.h  # h(a, r)
        deriv = self.h.deriv  # h'(a, r)
        if self.batched:
            z = z.swapaxes(0, 1)
            a = a.dimshuffle(0, "x", "x")
            b = b.dimshuffle(0, "x", "x")
            z_ref = z_ref.dimshuffle(0, "x", 1)
            # a bx-x-
            # b bx-x-
            # z bxsxd
            # z_ref bx-xd
        r = (z - z_ref).norm(2, axis=-1, keepdims=True)  # s
        har = h(a, r)
        dar = deriv(a, r)
        logdet = at.log((1.0 + b * har) ** (d - 1.0) * (1.0 + b * har + b * dar * r))
        if self.batched:
            return logdet.sum([0, -1])
        else:
            return logdet.sum(-1)


class Radial(FlowFn):
    @staticmethod
    def fn(*args):
        a, r = args
        return 1.0 / (a + r)

    @staticmethod
    def inv(*args):
        a, y = args
        return 1.0 / y - a

    @staticmethod
    def deriv(*args):
        a, r = args
        return -1.0 / (a + r) ** 2


@AbstractFlow.register
class RadialFlow(ReferencePointFlow):
    short_name = "radial"

    def __init__(self, **kwargs):
        super().__init__(Radial(), **kwargs)

    def make_ab(self, a, b):
        a = at.exp(a)
        b = -a + at.softplus(b)
        return a, b


@AbstractFlow.register
class LocFlow(AbstractFlow):
    __param_spec__ = dict(loc=("d",))
    short_name = "loc"

    def __init__(self, loc=None, **kwargs):
        super().__init__(**kwargs)
        loc = self.add_param(loc, "loc")
        self.shared_params = dict(loc=loc)

    loc = property(lambda self: self.shared_params["loc"])

    @node_property
    def forward(self):
        loc = self.loc  # (bx)d
        z = self.z0  # sx(bx)d
        return z + loc

    @node_property
    def logdet(self):
        return at.zeros((self.z0.shape[0],))


@AbstractFlow.register
class ScaleFlow(AbstractFlow):
    __param_spec__ = dict(rho=("d",))
    short_name = "scale"

    @aesara.config.change_flags(compute_test_value="off")
    def __init__(self, rho=None, **kwargs):
        super().__init__(**kwargs)
        rho = self.add_param(rho, "rho")
        self.scale = rho2sigma(rho)
        self.shared_params = dict(rho=rho)

    log_scale = property(lambda self: self.shared_params["log_scale"])

    @node_property
    def forward(self):
        scale = self.scale  # (bx)d
        z = self.z0  # sx(bx)d
        return z * scale

    @node_property
    def logdet(self):
        return at.repeat(at.sum(at.log(self.scale)), self.z0.shape[0])


@AbstractFlow.register
class HouseholderFlow(AbstractFlow):
    __param_spec__ = dict(v=("d",))
    short_name = "hh"

    @aesara.config.change_flags(compute_test_value="raise")
    def __init__(self, v=None, **kwargs):
        super().__init__(**kwargs)
        v = self.add_param(v, "v")
        self.shared_params = dict(v=v)
        if self.batched:
            vv = v.dimshuffle(0, 1, "x") * v.dimshuffle(0, "x", 1)
            I = at.eye(self.dim).dimshuffle("x", 0, 1)
            vvn = (1e-10 + (v**2).sum(-1)).dimshuffle(0, "x", "x")
        else:
            vv = at.outer(v, v)
            I = at.eye(self.dim)
            vvn = (v**2).sum(-1) + 1e-10
        self.H = I - 2.0 * vv / vvn

    @node_property
    def forward(self):
        z = self.z0  # sxd
        H = self.H  # dxd
        if self.batched:
            return at.batched_dot(z.swapaxes(0, 1), H).swapaxes(0, 1)
        else:
            return z.dot(H)

    @node_property
    def logdet(self):
        return at.zeros((self.z0.shape[0],))
