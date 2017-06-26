import warnings
import collections

import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from pymc3.theanof import change_flags
from .opvi import node_property, cast_to_list

__all__ = [
]


class AbstractFlow(object):
    shared_params = None

    def __init__(self, z0=None, dim=None, parent=None):
        if dim is not None and parent is not None:
            if dim != parent.dim:
                raise ValueError('Provided `dim` and `parent.dim` are not equal')
        if parent is not None:
            self.dim = parent.dim
        elif dim is not None:
            self.dim = dim
        else:
            raise ValueError('Need `parent` or `dim` to infer dimension of flow')
        if z0 is not None and parent is not None:
            if z0 is not parent.forward:
                raise ValueError('Provided `z0` and `parent.forward` are not equal')
        elif z0 is not None:
            self.z0 = z0
        elif parent is not None:
            self.z0 = parent.forward
        else:
            self.z0 = tt.matrix()
            self.z0.tag.test_value = np.random.rand(
                2, dim
            ).astype(self.z0.dtype)
        self.parent = parent
        self._initialize(self.dim)

    @property
    def params(self):
        return cast_to_list(self.shared_params)

    @property
    def all_params(self):
        params = self.params  # type: list
        current = self
        while not current.isroot:
            current = current.parent
            params.extend(current.params)
        return params

    @property
    def all_dets(self):
        dets = list()
        dets.append(self.det)
        current = self
        while not current.isroot:
            current = current.parent
            dets.append(current.det)
        return tt.add(*dets)

    def _initialize(self, dim):
        pass

    @node_property
    def forward(self):
        raise NotImplementedError

    @node_property
    def det(self):
        raise NotImplementedError

    @change_flags(compute_test_value='off')
    def forward_apply(self, z0):
        ret = theano.clone(self.forward, {self.root.z0: z0})
        ret.tag.test_value = np.random.normal(
            size=z0.tag.test_value.shape
        ).astype(self.z0.dtype)
        return ret

    @property
    def root(self):
        current = self
        while not current.isroot:
            current = current.parent
        return current

    @property
    def isroot(self):
        return self.parent is None


def link_flows(flows, z0=None):
    """Link flows in given order, optionally override
    starting `z0` with new one. This operation can be
    performed only once as `owner` attributes are set
    on symbolic variables

    Parameters
    ----------
    flows : list[AbstractFlow]
    z0 : matrix

    Returns
    -------
    list[AbstractFlow]
    """
    view_op = theano.compile.view_op
    if z0 is not None:
        theano.Apply(view_op, [z0], [flows[0].z0])
    for f0, f1 in zip(flows[:-1], flows[1:]):
        if f0.dim != f1.dim:
            raise ValueError('Flows have different dims')
        theano.Apply(view_op, [f0.forward], [f1.z0])
        f1.parent = f0
    return flows

FlowFn = collections.namedtuple('FlowFn', 'fn,inv,deriv')
FlowFn.__call__ = lambda self, x: self.fn(x)


class LinearFlow(AbstractFlow):
    def __init__(self, h, z0=None, dim=None, parent=None):
        self.h = h
        super(LinearFlow, self).__init__(dim=dim, z0=z0, parent=parent)

    def _initialize(self, dim):
        super(LinearFlow, self)._initialize(dim)
        _u = theano.shared(pm.floatX(np.random.randn(dim, 1)))
        _w = theano.shared(pm.floatX(np.random.randn(dim, 1)))
        b = theano.shared(pm.floatX(np.random.randn()))
        self.shared_params = dict(_u=_u, _w=_w, b=b)
        self.u, self.w = self.make_uw(self._u, self._w)
        self.u = tt.patternbroadcast(self.u, (False, True))
        self.w = tt.patternbroadcast(self.w, (False, True))

    _u = property(lambda self: self.shared_params['_u'])
    _w = property(lambda self: self.shared_params['_w'])
    b = property(lambda self: self.shared_params['b'])

    def make_uw(self, u, w):
        warnings.warn('flow can be not revertible', stacklevel=3)
        return u, w

    @node_property
    def forward(self):
        z = self.z0  # sxd
        u = self.u   # dx1
        w = self.w   # dx1
        b = self.b   # ()
        h = self.h   # f
        # h(sxd \dot dx1 + .)  = sx1
        hwz = h(z.dot(w) + b)  # sx1
        # sx1 + (sx1 * 1xd) = sxd
        z1 = z + hwz * u.T     # sxd
        return z1

    @node_property
    def det(self):
        z = self.z0  # sxd
        u = self.u   # dx1
        w = self.w   # dx1
        b = self.b   # ()
        deriv = self.h.deriv  # f'
        # h^-1(sxd \dot dx1 + .) * 1xd   = sxd
        phi = deriv(z.dot(w) + b) * w.T  # sxd
        # \abs(. + sxd \dot dx1) = sx1
        det = tt.abs_(1. + phi.dot(u))
        return det.flatten()  # s

Tanh = FlowFn(tt.tanh, tt.arctanh, lambda x: 1. - tt.tanh(x) ** 2)


class PlanarFlow(LinearFlow):
    def __init__(self, **kwargs):
        super(PlanarFlow, self).__init__(h=Tanh, **kwargs)

    def make_uw(self, u, w):
        # u : dx1
        # w : dx1
        # --> reparametrize
        # u' : dx1
        # w : dx1
        wu = w.T.dot(u).reshape(())  # .
        mwu = -1. + tt.log1p(tt.exp(wu))  # .
        # dx1 + (1x1 - 1x1) * dx1 / .
        u_h = u+(mwu-wu)*w/(w**2).sum()
        return u_h, w
