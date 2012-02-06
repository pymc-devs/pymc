from numpy.testing import *
import numpy as np
import pymc as pm

try:
    from types import UnboundMethodType
except ImportError:
    # On Python 3, unbound methods are just functions.
    def UnboundMethodType(func, inst, cls):
        return func

submod = pm.gp.GPSubmodel('x5',pm.gp.Mean(lambda x:0*x),pm.gp.FullRankCovariance(pm.gp.cov_funs.exponential.euclidean, amp=1, scale=1),np.linspace(-1,1,21))
x = [pm.MvNormalCov('x0',np.zeros(5),np.eye(5)),
    pm.Gamma('x1',4,4,size=3),
    pm.Gamma('x2',2,2),
    pm.Binomial('x3',100,.4),
    pm.Bernoulli('x4',.5),
    submod.f]

do_not_implement_methods = ['iadd','isub','imul','itruediv','ifloordiv','imod','ipow','ilshift','irshift','iand','ixor','ior']
uni_methods = ['neg','pos','abs','invert','index']
rl_bin_methods = ['div', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']+['add', 'mul', 'sub']
bin_methods = ['call','getitem']

class test_special_methods(TestCase):
    def test_call(self):
        for y in x[:2]:
            assert_equal(y[0].value, y.value[0])
    def test_getitem(self):
        assert_equal(x[5](3).value, x[5].value(3))

for dnim in do_not_implement_methods:
    def meth(self, dnim=dnim):
        for y in x:
            if hasattr(y, '__%s__'%dnim):
                raise AssertionError('Method %s implemented in class %s'%(dnim, y.__class__))
    setattr(test_special_methods, 'test_'+dnim, UnboundMethodType(meth, None, test_special_methods))

for dnim in uni_methods:
    def meth(self, dnim=dnim):
        # These methods only work on integer-valued variables.
        if dnim in ['index','invert']:
            testvars = [x[3]]
        # All the others work on all numeric ar ndarray variables.
        else:
            testvars = x[:5]
        for y in testvars:
            assert_equal(getattr(y,'__%s__'%dnim)().value, getattr(y.value,'__%s__'%dnim)())
    setattr(test_special_methods, 'test_'+dnim, UnboundMethodType(meth, None, test_special_methods))

for dnim in rl_bin_methods:
    def meth(self, dnim=dnim):
        # These only work for boolean-valued variables.
        if dnim in ['or','xor','and']:
            testvars = [x[4]]
            for y in testvars:
                assert_equal(getattr(y,'__%s__'%dnim)(True).value, getattr(y.value,'__%s__'%dnim)(True))
        # These only work for integers and booleans.
        elif dnim in ['lshift','rshift']:
            testvars = x[3:5]
            for y in testvars:
                assert_equal(getattr(y,'__%s__'%dnim)(2).value, getattr(y.value,'__%s__'%dnim)(2))
        # These should work for all numeric or ndarray variables
        else:
            testvars = x[:4]
            for y in testvars:
                assert_equal(getattr(y,'__%s__'%dnim)(3.0).value, getattr(y.value,'__%s__'%dnim)(3.0))
    setattr(test_special_methods, 'test_'+dnim, UnboundMethodType(meth, None, test_special_methods))


if __name__ == '__main__':
    import unittest
    unittest.main()
