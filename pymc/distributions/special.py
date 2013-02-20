'''
Created on Mar 17, 2011

@author: jsalvatier
'''
from theano import scalar,tensor
import numpy 
from scipy import special, misc

__all__ = ['gammaln', 'multigammaln', 'psi', 'multipsi', 'factln']

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
        x, = inp
        gz, = grads
        return [gz * scalar_psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                lgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    
scalar_gammaln  = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tensor.Elemwise(scalar_gammaln, name='gammaln')

class MultiGammaLn(scalar.UnaryScalarOp):
    """
    Compute multigammaln(p, x)
    """
    @staticmethod
    def st_impl(p, x):
        return special.multigammaln(p, x)

    def impl(self, x):
        return MultiGammaLn.st_impl(x)

    def grad(self, inp, grads):
        p, x = inp
        gz, = grads
        return [p.zeros_like().astype(theano.config.floatX), gz * scalar_multipsi(p, x)]

    def c_support_code(self):
        return (
        """
#ifndef _MULTIGAMMAFUNCDEFINED
#define _MULTIGAMMAFuNCDEFINED

double _multigammaln(int p, double x){

    double lnpi = 1.14472988585 ;
    double res = %(p)s * ( %(p)s -1)*lnpi/4.0;

    for (int i = 0: i < %(p)s; i++){
        res += lgamma(%(x)s - i/2.0);
        }
    return res;
    }

#endif
        """)
    def c_code(self, node, name, inp, out, sub):
        p, x = inp
        z, = out
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                _multigammaln(%(p)s, %(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))


    
scalar_multigammaln  = GammaLn(scalar.upgrade_to_float, name='scalar_multigammaln')
multigammaln = tensor.Elemwise(scalar_multigammaln, name='multigammaln')

cpsifunc = """
#ifndef _PSIFUNCDEFINED
#define _PSIFUNCDEFINED
double _psi(double x){

    /*taken from 
    Bernardo, J. M. (1976). Algorithm AS 103: Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317. 
    http://www.uv.es/~bernardo/1976AppStatist.pdf */
    
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
    def st_impl(p, x):
        return special.psi(p, x)
    def impl(self, x):
        return Psi.st_impl(x)
    
    #def grad()  no gradient now 
    
    def c_support_code(self):
        return cpsifunc
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                _psi(%(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
    
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    
scalar_psi = Psi(scalar.upgrade_to_float, name='scalar_psi')
psi = tensor.Elemwise(scalar_psi, name='psi')


class MultiPsi(scalar.UnaryScalarOp):
    """
    Compute derivative of gammaln(x)
    """
    @staticmethod
    def st_impl(p, x):
        r = p*(p-1)/4. 
        for i in range(p): 
            r += special.psi(x - i/2.)
        return r
    def impl(self, x):
        return Psi.st_impl(x)
    
    #def grad()  no gradient now 
    
    def c_support_code(self):
        return (cpsifunc + 
        """
#ifndef _MULTIPSIFUNCDEFINED
#define _MULTIPSIFuNCDEFINED

double _multipsi(int p, double x){
    double res = 0.0;

    for (int i = 0: i < %(p)s ; i++){
        res += _psi(%(x)s - i/2.0);
        }
    return res;
    }
#endif
        """)
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                _multipsi(%(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
    
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    
scalar_multipsi = Psi(scalar.upgrade_to_float, name='scalar_multipsi')
multipsi = tensor.Elemwise(scalar_multipsi, name='multipsi')


class FactLn(scalar.UnaryScalarOp):
    """
    Compute factln(x)
    """
    @staticmethod
    def st_impl(x):
        return numpy.log(misc.factorial(x))
    def impl(self, x):
        return FactLn.st_impl(x)
    
    #def grad()  no gradient now 
    
    def c_support_code(self):
        return ( 
        """
double factln(int n){
    static double cachedfl[100];
    
    if (n < 0) return -1.0; // need to return -inf here at some point
    if (n <= 1) return 0.0;
    if (n < 100) return cachedfl[n] ? cachedfl[n] : (cachedfl[n]=lgammln(n + 1.0));
    else return lgammln(n+1.0);}
    """ )
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                factln(%(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
    
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    
scalar_factln = Psi(scalar.upgrade_to_float, name='scalar_factln')
factln = tensor.Elemwise(scalar_factln, name='factln')
