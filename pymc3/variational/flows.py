import numpy as np
import theano
from theano import tensor as tt

from pymc3.theanof import change_flags
from .opvi import node_property, collect_shared_to_list

__all__ = [
    'Formula',
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
    __call__(z0, dim, jitter) - initializes and links all flows returning the last one
    """

    def __init__(self, formula):
        _select = dict(
            planar=PlanarFlow,
            radial=RadialFlow,
            hh=HouseholderFlow,
            loc=LocFlow,
            scale=ScaleFlow,
        )
        identifiers = formula.lower().replace(' ', '').split('-')
        self.formula = '-'.join(identifiers)
        identifiers = [idf.split('*') for idf in identifiers]
        self.flows = []

        for tup in identifiers:
            if len(tup) == 1:
                self.flows.append(_select[tup[0]])
            elif len(tup) == 2:
                self.flows.extend([_select[tup[0]]]*int(tup[1]))
            else:
                raise ValueError('Wrong format: %s' % formula)
        if len(self.flows) == 0:
            raise ValueError('No flows in formula')

    def __call__(self, z0=None, dim=None, jitter=.001):
        if len(self.flows) == 0:
            raise ValueError('No flows in formula')
        flow = z0
        for flow_cls in self.flows:
            flow = flow_cls(dim=dim, jitter=jitter, z0=flow)
        return flow

    def __reduce__(self):
        return self.__class__, self.formula

    def __repr__(self):
        return self.formula


class AbstractFlow(object):
    shared_params = None

    def __init__(self, z0=None, dim=None, jitter=.001):
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
            raise ValueError('Cannot infer dimension of flow, '
                             'please provide dim or Flow instance as z0')
        if z0 is None:
            self.z0 = tt.matrix()  # type: tt.TensorVariable
        else:
            self.z0 = z0
        self.parent = parent

    def add_param(self, shape, name=None, ref=0., dtype='floatX'):
        if dtype == 'floatX':
            dtype = theano.config.floatX
        return theano.shared(
            np.asarray(np.random.normal(size=shape) * self.__jitter + ref).astype(dtype),
            name=name
        )

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


class FlowFn(object):
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
    @change_flags(compute_test_value='off')
    def __init__(self, h, z0=None, dim=None, u=None, w=None, b=None, jitter=.001):
        self.h = h
        super(LinearFlow, self).__init__(dim=dim, z0=z0, jitter=jitter)
        if u is None:
            _u = self.add_param(dim, '_u')
        else:
            _u = u
        if w is None:
            _w = self.add_param(dim, '_w')
        else:
            _w = w
        if b is None:
            b = self.add_param(None, 'b')
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


class Tanh(FlowFn):
    fn = tt.tanh
    inv = tt.arctanh

    @staticmethod
    def deriv(*args):
        x, = args
        return 1. - tt.tanh(x) ** 2


class PlanarFlow(LinearFlow):
    def __init__(self, **kwargs):
        super(PlanarFlow, self).__init__(h=Tanh(), **kwargs)

    def make_uw(self, u, w):
        # u : dx1
        # w : dx1
        # --> reparametrize
        # u' : dx1
        # w : dx1
        wu = w.T.dot(u).reshape(())  # .
        mwu = -1. + tt.nnet.softplus(wu)  # .
        # dx1 + (1x1 - 1x1) * dx1 / .
        u_h = u+(mwu-wu) * w/((w**2).sum()+1e-10)
        return u_h, w


class ReferencePointFlow(AbstractFlow):
    @change_flags(compute_test_value='off')
    def __init__(self, h, z0=None, dim=None, a=None, b=None, z_ref=None, jitter=.1):
        super(ReferencePointFlow, self).__init__(dim=dim, z0=z0, jitter=jitter)
        if a is None:
            _a = self.add_param(None, '_a')
        else:
            _a = a
        if b is None:
            _b = self.add_param(None, '_b')
        else:
            _b = b
        if z_ref is None:
            if hasattr(self.z0, 'tag') and hasattr(self.z0.tag, 'test_value'):
                z_ref = self.add_param(
                    self.z0.tag.test_value[0].shape, 'z_ref',
                    ref=self.z0.tag.test_value[0],
                    dtype=self.z0.dtype
                )
            else:
                z_ref = self.add_param(
                    dim, 'z_ref', dtype=self.z0.dtype
                )
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


class Radial(FlowFn):
    @staticmethod
    def fn(*args):
        a, r = args
        return 1./(a+r)

    @staticmethod
    def inv(*args):
        a, y = args
        return 1./y - a

    @staticmethod
    def deriv(*args):
        a, r = args
        return -1. / (a + r) ** 2


class RadialFlow(ReferencePointFlow):
    def __init__(self, **kwargs):
        super(RadialFlow, self).__init__(Radial(), **kwargs)

    def make_ab(self, a, b):
        a = tt.exp(a)
        b = -a + tt.nnet.softplus(b)
        return a, b


class LocFlow(AbstractFlow):
    def __init__(self, z0=None, dim=None, loc=None, jitter=0):
        super(LocFlow, self).__init__(dim=dim, z0=z0, jitter=jitter)
        if loc is None:
            loc = self.add_param(dim, 'loc')
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
    @change_flags(compute_test_value='off')
    def __init__(self, z0=None, dim=None, log_scale=None, jitter=.1):
        super(ScaleFlow, self).__init__(dim=dim, z0=z0, jitter=jitter)
        if log_scale is None:
            log_scale = self.add_param(dim, 'log_scale')
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
        return tt.repeat(tt.sum(self.shared_params['log_scale']), self.z0.shape[0])


class HouseholderFlow(AbstractFlow):
    @change_flags(compute_test_value='raise')
    def __init__(self, z0=None, dim=None, v=None, jitter=.1):
        super(HouseholderFlow, self).__init__(dim=dim, z0=z0, jitter=jitter)
        if v is None:
            v = self.add_param(dim, 'v')
        self.shared_params = dict(v=v)
        v = v.dimshuffle(0, 'x')
        self.H = tt.eye(dim) - 2. * v.dot(v.T) / ((v**2).sum()+1e-10)

    @node_property
    def forward(self):
        z = self.z0  # sxd
        H = self.H   # dxd
        return z.dot(H)

    @node_property
    def logdet(self):
        return tt.zeros((self.z0.shape[0],))
