import collections

import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from pymc3.theanof import change_flags
from .opvi import node_property, collect_shared_to_list

__all__ = [
    'Formula',
    'link_flows',
    'PlanarFlow',
    'LocFlow',
    'ScaleFlow'
]


class Formula(object):
    """
    Helpful class to use string like formulas with
    __call__ syntax similar to Flow.__init__

    Parameters
    ----------
    formula : str
        string representing normalizing flow
        e.g. 'planar', 'planar*4', 'planar*4-radial*3', 'planar-radial-planar'
        Yet simple pattern is supported:

            1. dash separated flow identifiers
            2. star for replication after flow identifier

    Methods
    -------
    __call__(z0, dim) - initializes and links all flows returning the last one
    """

    def __init__(self, formula):
        _select = dict(
            planar=PlanarFlow,
            radial=RadialFlow,
            hh=HouseholderFlow,
            loc=LocFlow,
            scale=ScaleFlow,
        )
        self.formula = formula
        _formula = formula.lower().replace(' ', '')
        identifiers = _formula.split('-')
        identifiers = [idf.split('*') for idf in identifiers]
        self.flows = []

        for tup in identifiers:
            if len(tup) == 1:
                self.flows.append(_select[tup[0]])
            elif len(tup) == 2:
                self.flows.extend([_select[tup[0]]]*int(tup[1]))
            else:
                raise ValueError('Wrong format: %s' % formula)

    def __call__(self, z0=None, dim=None):
        _flows = [flow(dim=dim) for flow in self.flows]
        return link_flows(_flows, z0)[-1]

    def __reduce__(self):
        return self.__class__, self.formula


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
        if isinstance(z0, AbstractFlow):
            z0 = z0.forward
        theano.Apply(view_op, [z0], [flows[0].z0])
    for f0, f1 in zip(flows[:-1], flows[1:]):
        if f0.dim != f1.dim:
            raise ValueError('Flows have different dims')
        theano.Apply(view_op, [f0.forward.astype(f1.z0.dtype)], [f1.z0])
        f1.parent = f0
    return flows


class AbstractFlow(object):
    shared_params = None

    @change_flags(compute_test_value='raise')
    def __init__(self, z0=None, dim=None):
        if isinstance(z0, AbstractFlow):
            parent = z0
            dim = parent.dim
            z0 = parent.forward
        else:
            parent = None
        if dim is not None:
            self.dim = dim
        else:
            raise ValueError('Cannot infer dimension of flow, '
                             'please provide dim or Flow instance as z0')
        if z0 is None:
            self.z0 = tt.matrix()  # type: tt.TensorVariable
            self.z0.tag.test_value = np.random.rand(
                2, dim
            ).astype(self.z0.dtype)
        else:
            self.z0 = z0
            if not hasattr(z0.tag, 'test_value'):
                self.z0.tag.test_value = np.random.rand(
                    2, dim
                ).astype(self.z0.dtype)
        self.parent = parent

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
    def sum_logdets(self):
        dets = [self.logdet]
        current = self
        while not current.isroot:
            current = current.parent
            dets.append(current.logdet)
        return tt.add(*dets)

    @node_property
    def forward(self):
        raise NotImplementedError

    @node_property
    def logdet(self):
        raise NotImplementedError

    @change_flags(compute_test_value='off')
    def forward_pass(self, z0):
        ret = theano.clone(self.forward, {self.root.z0: z0})
        try:
            ret.tag.test_value = np.random.normal(
                size=z0.tag.test_value.shape
            ).astype(self.z0.dtype)
        except AttributeError:
            ret.tag.test_value = self.root.z0.tag.test_value
        return ret

    __call__ = forward_pass

    @property
    def root(self):
        current = self
        while not current.isroot:
            current = current.parent
        return current

    @property
    def isroot(self):
        return self.parent is None


FlowFn = collections.namedtuple('FlowFn', 'fn,inv,deriv')
FlowFn.__call__ = lambda self, *args: self.fn(*args)


class LinearFlow(AbstractFlow):
    @change_flags(compute_test_value='raise')
    def __init__(self, h, z0=None, dim=None, u=None, w=None, b=None):
        self.h = h
        super(LinearFlow, self).__init__(dim=dim, z0=z0)
        if u is None:
            _u = theano.shared(pm.floatX(np.random.randn(dim)))
        else:
            _u = u
        if w is None:
            _w = theano.shared(pm.floatX(np.random.randn(dim)))
        else:
            _w = w
        if b is None:
            b = theano.shared(pm.floatX(np.random.randn()))
        self.shared_params = dict(_u=_u, _w=_w, b=b)
        self.u, self.w = self.make_uw(self._u, self._w)
        self.u = self.u.dimshuffle(0, 'x')
        self.w = self.w.dimshuffle(0, 'x')

    _u = property(lambda self: self.shared_params['_u'])
    _w = property(lambda self: self.shared_params['_w'])
    b = property(lambda self: self.shared_params['b'])

    def make_uw(self, u, w):
        raise NotImplementedError('Need to implement valid U, W transform')

    @node_property
    def forward(self):
        z = self.z0  # sxd
        u = self.u   # dx1
        w = self.w   # dx1
        b = self.b   # .
        h = self.h   # f
        # h(sxd \dot dx1 + .)  = sx1
        hwz = h(z.dot(w) + b)  # sx1
        # sx1 + (sx1 * 1xd) = sxd
        z1 = z + hwz * u.T     # sxd
        return z1

    @node_property
    def logdet(self):
        z = self.z0  # sxd
        u = self.u   # dx1
        w = self.w   # dx1
        b = self.b   # .
        deriv = self.h.deriv  # f'
        # h^-1(sxd \dot dx1 + .) * 1xd   = sxd
        phi = deriv(z.dot(w) + b) * w.T  # sxd
        # \abs(. + sxd \dot dx1) = sx1
        det = tt.abs_(1. + phi.dot(u))
        det = det.flatten()  # s
        return tt.log(det)

Tanh = FlowFn(
    tt.tanh,
    tt.arctanh,
    lambda x: 1. - tt.tanh(x) ** 2
)


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
        mwu = -1. + tt.nnet.softplus(wu)  # .
        # dx1 + (1x1 - 1x1) * dx1 / .
        u_h = u+(mwu-wu) * w/(w**2).sum()
        return u_h, w


class ReferencePointFlow(AbstractFlow):
    @change_flags(compute_test_value='raise')
    def __init__(self, h, z0=None, dim=None, a=None, b=None, z_ref=None):
        super(ReferencePointFlow, self).__init__(dim=dim, z0=z0)
        if a is None:
            _a = theano.shared(pm.floatX(np.random.randn()))
        else:
            _a = a
        if b is None:
            _b = theano.shared(pm.floatX(np.random.randn()))
        else:
            _b = b
        if z_ref is None:
            z_ref = theano.shared(pm.floatX(np.random.randn(dim)))
        self.h = h
        self.shared_params = dict(_a=_a, _b=_b, z_ref=z_ref)
        self.a, self.b = self.make_ab(_a, _b)
        self.z_ref = z_ref

    def make_ab(self, a, b):
        raise NotImplementedError('Need to specify how to get a, b')

    @node_property
    def forward(self):
        a = self.a  # .
        b = self.b  # .
        z_ref = self.z_ref  # d
        z = self.z0  # sxd
        h = self.h  # h(a, r)
        r = (z - z_ref).norm(2, axis=-1, keepdims=True)  # sx1
        return z + b * h(a, r) * (z-z_ref)

    @node_property
    def logdet(self):
        d = self.dim
        a = self.a  # .
        b = self.b  # .
        z_ref = self.z_ref  # d
        z0 = self.z0  # sxd
        h = self.h  # h(a, r)
        r = (z0 - z_ref).norm(2, axis=-1)  # s
        deriv = self.h.deriv  # h'(a, r)
        har = h(a, r)
        dar = deriv(a, r)
        det = (1. + b*har)**(d-1) * (1. + b*har + b*dar*r)
        return tt.log(det)


Radial = FlowFn(
    lambda a, r: 1./(a+r),
    lambda a, y: 1./y - a,
    lambda a, r: -1./(a+r)**2
)


class RadialFlow(ReferencePointFlow):
    def __init__(self, **kwargs):
        super(RadialFlow, self).__init__(Radial, **kwargs)

    def make_ab(self, a, b):
        a = tt.exp(a)
        b = -a + tt.nnet.softplus(b)
        return a, b


class LocFlow(AbstractFlow):
    def __init__(self, z0=None, dim=None, loc=None):
        super(LocFlow, self).__init__(dim=dim, z0=z0)
        if loc is None:
            loc = theano.shared(pm.floatX(np.random.randn(dim)))
        self.shared_params = dict(loc=loc)
        self.loc = loc

    @node_property
    def forward(self):
        loc = self.loc
        z = self.z0
        return z + loc

    @node_property
    def logdet(self):
        return tt.zeros((self.z0.shape[0],))


class ScaleFlow(AbstractFlow):
    @change_flags(compute_test_value='raise')
    def __init__(self, z0=None, dim=None, log_scale=None):
        super(ScaleFlow, self).__init__(dim=dim, z0=z0)
        if log_scale is None:
            log_scale = theano.shared(pm.floatX(np.random.randn(dim)))
        scale = tt.exp(log_scale)
        self.shared_params = dict(log_scale=log_scale)
        self.scale = scale

    @node_property
    def forward(self):
        z = self.z0
        scale = self.scale
        return z * scale

    @node_property
    def logdet(self):
        return tt.sum(self.shared_params['log_scale'])


class HouseholderFlow(AbstractFlow):
    @change_flags(compute_test_value='raise')
    def __init__(self, z0=None, dim=None, v=None):
        super(HouseholderFlow, self).__init__(dim=dim, z0=z0)
        if v is None:
            v = theano.shared(pm.floatX(np.random.randn(dim)))
        self.shared_params = dict(v=v)
        v = v.dimshuffle(0, 'x')
        self.H = tt.eye(dim) - 2. * v.dot(v.T) / (v**2).sum()

    @node_property
    def forward(self):
        z = self.z0  # sxd
        H = self.H   # dxd
        return z.dot(H)

    @node_property
    def logdet(self):
        return tt.zeros((self.z0.shape[0],))
