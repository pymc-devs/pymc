from numpy.testing import *
from numpy import *
from pymc import *
from numpy.random import randint

class test_Gibbs(NumpyTestCase):
    
    def check_BetaBinomial(self):
        p = Beta('p',.2,alpha=1.,beta=1.)
        d1 = Binomial('d1', randint(0,16,10), n=15, p=p)
        n2 = Uninformative('n2', 4)
        d2 = Binomial('d2', randint(0,5,3), n=n2, p=p)

        p_stepper = BetaBinomial(p,d=[d1,d2],n=[15,n2],alpha=2,beta=1)

        p_values = zeros(10000)
        for i in xrange(10000):
            p_stepper.step()
            p_values[i] = p.value

        a_real = float(sum(d1.value) + sum(d2.value) + 2)
        b_real = float(15 * 10 + n2.value * 3 + 1 -sum(d1.value) - sum(d2.value))
        assert(abs(mean(p_values) - a_real / (a_real + b_real))< .01)
        assert(abs(var(p_values) - (a_real*b_real) / (a_real + b_real)**2 / (a_real + b_real + 1)) < .0001)
    
    def check_VecBetaBinomial(self):
        p = Beta('p',.2*ones(5),alpha=1.,beta=1.)
        d = Binomial('d1', randint(0,16,5), n=15, p=p)
        p_stepper = VecBetaBinomial(p,d=d,n=15*ones(5),alpha=2,beta=1)
    
        p_values = zeros((5,10000))
        for i in xrange(10000):
            p_stepper.step()
            p_values[:,i] = p.value
    
        a_real = d.value + 2.
        b_real = 15-d.value + 1.
        assert((abs(mean(p_values, axis=1) - a_real / (a_real + b_real))<.01).all())
        assert((abs(var(p_values, axis=1) - (a_real*b_real) / (a_real + b_real)**2 / (a_real + b_real + 1))<.01).all())

    def check_GammaNormal(self):
        tau = Gamma('tau', 3., alpha=1., beta=1.)
        mu1 = Uninformative('mu1', ones(5)*.2)
        @stoch
        def d1(value=ones(5), mu=mu1, tau=tau):
            """d1 ~ distribution(mu, tau)"""
            return mvnormal_like(value, mu, tau*(eye(5)+.2))

        d2 = Normal('d2', 2., mu=3., tau=tau)
        tau_stepper = GammaNormal(tau, [d1,d2], [mu1,3.], theta=[eye(5)+.2, 1.], alpha=1., beta=1.)

        tau_values = zeros(10000)
        for i in xrange(10000):
            tau_stepper.step()
            tau_values[i] = tau.value

        beta_real = (.5 * (.8**2 * sum(eye(5)+.2) + 1.)+1.)
        alpha_real = 3.+1.
        assert(abs(mean(tau_values)- alpha_real / beta_real)<.01)
        assert(abs(var(tau_values)- alpha_real / beta_real ** 2)<.01)
    
    
if __name__ == '__main__':
    NumpyTest().run()

