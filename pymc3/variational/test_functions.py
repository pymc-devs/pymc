from theano import tensor as tt
from .opvi import TestFunction


__all__ = [
    'rbf'
]


class Kernel(TestFunction):
    """
    Dummy base class for kernel SVGD in case we implement more

    .. math::

        f(x) -> (k(x,.), \nabla_x k(x,.))

    """


class RBF(Kernel):
    def __call__(self, X):
        XY = X.dot(X.T)
        x2 = tt.reshape(tt.sum(tt.square(X), axis=1), (X.shape[0], 1))
        X2e = tt.repeat(x2, X.shape[0], axis=1)
        H = tt.sub(tt.add(X2e, X2e.T), 2 * XY)

        V = tt.sort(H.flatten())
        length = V.shape[0]
        # median distance
        m = tt.switch(tt.eq((length % 2), 0),
                      # if even vector
                      tt.mean(V[((length // 2) - 1):((length // 2) + 1)]),
                      # if odd vector
                      V[length // 2])

        h = 0.5 * m / tt.log(X.shape[0].astype('float32') + 1.0)

        Kxy = tt.exp(-H / h / 2.0)
        dxkxy = -tt.dot(Kxy, X)
        sumkxy = tt.sum(Kxy, axis=1).dimshuffle(0, 'x')
        dxkxy = tt.add(dxkxy, tt.mul(X, sumkxy)) / h

        return Kxy, dxkxy


rbf = RBF()
