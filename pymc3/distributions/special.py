import numpy as np
import theano.tensor as tt
from scipy import special
from theano import scalar, tensor

__all__ = ['gammaln', 'multigammaln', 'psi', 'trigamma']


class GammaLn(scalar.UnaryScalarOp):
    """
    Compute gammaln(x)
    """
    @staticmethod
    def st_impl(x):
        return special.gammaln(x)

    def impl(self, x):
        return GammaLn.st_impl(x)

    def grad(self, inp, grads):
        [x] = inp
        [gz] = grads
        return [gz * scalar_psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        return """%(z)s = lgamma(%(x)s);""" % locals()

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

scalar_gammaln = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tt.Elemwise(scalar_gammaln, name='gammaln')


def multigammaln(a, p):
    """Multivariate Log Gamma

    :Parameters:
        a : tensor like
        p : int degrees of freedom
            p > 0
    """
    i = tt.arange(1, p + 1)
    return (p * (p - 1) * tt.log(np.pi) / 4.
            + tt.sum(gammaln(a + (1. - i) / 2.), axis=0))


cpsifunc = """
#ifndef _PSIFUNCDEFINED
#define _PSIFUNCDEFINED
double _psi(double x){

    /**
     * Taken from
     * Bernardo, J. M. (1976). Algorithm AS 103: Psi (Digamma) Function.
     *     Applied Statistics. 25 (3), 315-317.
     * http://www.uv.es/~bernardo/1976AppStatist.pdf
     */

    double y, R, psi_ = 0;
    double S  = 1.0e-5;
    double C = 8.5;
    double S3 = 8.333333333e-2;
    double S4 = 8.333333333e-3;
    double S5 = 3.968253968e-3;
    double D1 = -0.5772156649  ;

    y = x;

    if (y <= 0.0)
        return psi_;

    if (y <= S )
        return D1 - 1.0/y;

    while (y < C){
        psi_ = psi_ - 1.0 / y;
        y = y + 1;}

    R = 1.0 / y;
    psi_ = psi_ + log(y) - .5 * R ;
    R= R*R;
    psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

    return psi_;}
    #endif
        """


class Psi(scalar.UnaryScalarOp):
    """
    Compute derivative of gammaln(x)
    """
    @staticmethod
    def st_impl(x):
        return special.psi(x)

    def impl(self, x):
        return Psi.st_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * scalar_trigamma(x)]

    def c_support_code(self):
        return cpsifunc

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in scalar.complex_types:
            raise NotImplementedError(
                'type not supported', node.inputs[0].type)

        return """%(z)s = _psi(%(x)s);""" % locals()

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

scalar_psi = Psi(scalar.upgrade_to_float, name='scalar_psi')
psi = tt.Elemwise(scalar_psi, name='psi')


class Trigamma(scalar.UnaryScalarOp):
    """
    Compute 2nd derivative of gammaln(x)
    """
    @staticmethod
    def st_impl(x):
        return special.polygamma(1, x)

    def impl(self, x):
        return Psi.st_impl(x)

    # def grad()  no gradient now

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

scalar_trigamma = Trigamma(scalar.upgrade_to_float, name='scalar_trigamma')
trigamma = tensor.Elemwise(scalar_trigamma, name='trigamma')
