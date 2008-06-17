from pymc.gp import *
from pymc.gp.cov_funs import matern
from pymc import *
from pymc.gp import GP
from numpy import *
from numpy.testing import *
from numpy.random import normal


class test_MCMC(TestCase):
    def test(self):
        
        x = arange(-1.,1.,.1)

        def make_model():
            # Prior parameters of C
            # Matern seems to be segfaulting...
            # diff_degree = Lognormal('diff_degree', mu=1.4, tau=100, verbose=0)
            diff_degree = Uniform('diff_degree',.2,3)
            amp = Lognormal('amp', mu=.4, tau=1., verbose=0)
            scale = Lognormal('scale', mu=.5, tau=1., verbose=0)

            # The deterministic C is valued as a Covariance object.                    
            @deterministic(verbose=0)
            def C(eval_fun = matern.euclidean, diff_degree=diff_degree, amp=amp, scale=scale):
                return Covariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)


            # Prior parameters of M
            a = Normal('a', mu=1., tau=1., verbose=0)
            b = Normal('b', mu=.5, tau=1., verbose=0)
            c = Normal('c', mu=2., tau=1., verbose=0)

            # The mean M is valued as a Mean object.
            def linfun(x, a, b, c):
                return a * x ** 2 + b * x + c    
            @deterministic(verbose=0)
            def M(eval_fun = linfun, a=a, b=b, c=c):
                return Mean(eval_fun, a=a, b=b, c=c)

    
            # The GP itself
            fmesh = array([-.5, .5])
            f = GP(name="f", M=M, C=C, mesh=fmesh, init_mesh_vals = zeros(2), verbose=0)
    
    
            # Observation precision
            V = Gamma('V', alpha=3., beta=.002/3., verbose=0)
    
            # The data d is just array-valued. It's normally distributed about GP.f(obs_x).
            @data
            @stochastic(verbose=0)
            def d(value=array([3.1, 2.9]), mu=f, V=V, verbose=0):
                """
                Data
                """
                mu_eval = mu(fmesh)
                return flib.normal(value, mu_eval, 1./V)
    
            return locals()

        GPSampler = MCMC(make_model())
        GPSampler.use_step_method(GPNormal, f=GPSampler.f, obs_mesh=[-.5,.5], obs_V=GPSampler.V, obs_vals=GPSampler.d)
        GPSampler.assign_step_methods()
        GPSampler.sample(iter=500,burn=0,thin=10)
        
if __name__ == '__main__':
    unittest.main()
