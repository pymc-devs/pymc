###########################################################
#  Refactored by Dilin Wang
#  Original implementation by Alec Radford, Luke Metz, Soumith Chintala: https://github.com/Newmu/dcgan_code   
#  (c) Alec Radford, Luke Metz, Soumith Chintala
###########################################################

import theano
import theano.tensor as tt
import numpy as np

__all__ = [
    "Regularizer",
    "SGD",
    "Momentum",
    "Adagrad",
    "RMSprop",
    "Adadelta",
    "Adam"
]

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0., l2norm=False, frobnorm=False):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = tt.sqrt(tt.sum(tt.sqr(p), axis=0))
            desired = tt.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def l2_norm(self, p):
        return p/l2norm(p, axis=0)

    def frob_norm(self, p, nrows):
        return (p/tt.sqrt(tt.sum(tt.sqr(p))))*tt.sqrt(nrows)

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += tt.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p)
        if self.frobnorm > 0:
            p = self.frob_norm(p, self.frobnorm)
        return p

class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError


class SGD(Update):

    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, p, g):
        updates = []

        g = self.regularizer.gradient_regularize(p, g)
        updated_p = p - self.lr * g
        updated_p = self.regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
        return updates

class Momentum(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, p, g):
        updates = []

        g = self.regularizer.gradient_regularize(p, g)
        m = theano.shared(p.get_value() * 0.)
        v = (self.momentum * m) - (self.lr * g)
        updates.append((m, v))

        updated_p = p + v
        updated_p = self.regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
        return updates


class RMSprop(Update):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, p, g):
        updates = []

        g = self.regularizer.gradient_regularize(p, g)
        acc = theano.shared(p.get_value() * 0.)
        acc_new = self.rho * acc + (1 - self.rho) * g ** 2
        updates.append((acc, acc_new))

        updated_p = p - self.lr * (g / tt.sqrt(acc_new + self.epsilon))
        updated_p = self.regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))
        return updates


class Adam(Update):

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())  

    def __call__(self, p, g):
        updates = []
        t = theano.shared(floatX(1.))
        b1_t = self.b1*self.l**(t-1)
     
        g = self.regularizer.gradient_regularize(p, g)
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
     
        m_t = b1_t*m + (1 - b1_t)*g
        v_t = self.b2*v + (1 - self.b2)*g**2
        m_c = m_t / (1-self.b1**t)
        v_c = v_t / (1-self.b2**t)
        p_t = p - (self.lr * m_c) / (tt.sqrt(v_c) + self.e)
        p_t = self.regularizer.weight_regularize(p_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t) )

        updates.append((t, t + 1.))
        return updates


class Adagrad(Update):

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, p, g):
        updates = []
        g = self.regularizer.gradient_regularize(p, g)
        acc = theano.shared(p.get_value() * 0.)
        acc_t = acc + g ** 2
        updates.append((acc, acc_t))

        p_t = p - (self.lr / tt.sqrt(acc_t + self.epsilon)) * g
        p_t = self.regularizer.weight_regularize(p_t)
        updates.append((p, p_t))
        return updates  


class Adadelta(Update):

    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, p, g):
        updates = []
        g = self.regularizer.gradient_regularize(p, g)

        acc = theano.shared(p.get_value() * 0.)
        acc_delta = theano.shared(p.get_value() * 0.)
        acc_new = self.rho * acc + (1 - self.rho) * g ** 2
        updates.append((acc,acc_new))

        update = g * tt.sqrt(acc_delta + self.epsilon) / tt.sqrt(acc_new + self.epsilon)
        updated_p = p - self.lr * update
        updated_p = self.regularizer.weight_regularize(updated_p)
        updates.append((p, updated_p))

        acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
        updates.append((acc_delta,acc_delta_new))
        return updates

