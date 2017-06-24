import warnings
import collections

import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from .opvi import node_property

__all__ = [
]


class AbstractFlow(object):
    shared_params = None

    def __init__(self, z0=None, dim=None, parent=None):
        if dim is None and parent is not None:
            if dim != parent.dim:
                raise ValueError('Provided `dim` and `parent.dim` are not equal')
        if parent is not None:
            self.dim = parent.dim
        elif dim is not None:
            self.dim = dim
        else:
            raise ValueError('Need `parent` or `dim` to infer dimension of flow')
        if z0 is None and parent is not None:
            if z0 is not parent.forward:
                raise ValueError('Provided `z0` and `parent.forward` are not equal')
        elif z0 is not None:
            self.z0 = z0
        elif parent is not None:
            self.z0 = parent.forward
        else:
            raise ValueError('Need `parent` or `z0` to set input of flow')
        self.parent = parent
        self._initialize(dim)

    def _initialize(self, dim):
        self.z0.tag.test_value = np.random.rand(
            2, dim
        ).astype(self.z0.dtype)

    @node_property
    def forward(self):
        raise NotImplementedError

    @node_property
    def det(self):
        raise NotImplementedError

    def forward_apply(self, z0):
        return theano.clone(self.forward, {self.root.z0: z0})

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
        theano.Apply(view_op, [f0.forward], [f1.z0])
        f1.parent = f0
    return flows

FlowFn = collections.namedtuple('FlowFn', 'h,h_inv,h_deriv')


class LinearFlow(AbstractFlow):
    def __init__(self, h, z0=None, dim=None, parent=None):
        self.h, self.h_inv, self.h_deriv = h
        super(LinearFlow, self).__init__(dim=dim, z0=z0, parent=parent)

    def _initialize(self, dim):
        super(LinearFlow, self)._initialize(dim)
        _u = theano.shared(pm.floatX(np.random.randn(dim, 1) * 0.01))
        _w = theano.shared(pm.floatX(np.random.randn(dim, 1) * 0.01))
        b = theano.shared(pm.floatX(0))
        self.shared_params = dict(_u=_u, _w=_w, b=b)
        self.u, self.w = self.make_uw(self._u, self._w)

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
        # sx1 + 1xs = sxd
        z1 = z + u.T * hwz     # sxd
        return z1

    @node_property
    def det(self):
        z = self.z0  # sxd
        u = self.u   # dx1
        w = self.w   # dx1
        b = self.b   # ()
        h_inv = self.h_inv  # f^-1
        # h^-1(sxd \dot dx1 + .) * 1xd   = sxd
        phi = h_inv(z.dot(w) + b) * w.T  # sxd
        # \abs(. + sxd \dot dx1) = sx1
        det = tt.abs_(1. + phi.dot(u))
        return det

Tanh = FlowFn(tt.tanh, tt.arctanh, lambda x: 1. - tt.tanh(x) ** 2)


class PlanarFlow(LinearFlow):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__(dim, Tanh)

    def make_uw(self, u, w):
        # u : dx1
        # w : dx1
        # --> reparametrize
        # u' : dx1
        # w : dx1
        wu = w.T.dot(u)
        mwu = -1. + tt.log1p(tt.exp(wu))
        u_h = u+(mwu-wu)*w/(w**2).sum()
        return u_h, w
