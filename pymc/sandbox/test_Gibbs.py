from numpy.testing import *
from numpy import *
from pymc import *
from numpy.random import randint
from pylab import *
from GibbsStepMethods import *
try:
    from NormalSubmodel import *
    from NormalModel import *
except:
    pass

class test_Gibbs(NumpyTestCase):
    
    def check_BetaBinomial(self):
        p = Beta('p',value=.2,alpha=5.,beta=1.)
        d1 = Binomial('d1', value=randint(0,16,10), n=15, p=p)
        n2 = Uninformative('n2', value=4)
        d2 = Binomial('d2', value=randint(0,5,3), n=n2, p=p)
    
        p_stepper = BetaBinomial(p)
    
        p_values = empty(10000,dtype=float)
        for i in xrange(10000):
            p_stepper.step()
            p_values[i] = p.value
    
        a_real = float(sum(d1.value) + sum(d2.value) + 5)
        b_real = float(15 * 10 + n2.value * 3 + 1 -sum(d1.value) - sum(d2.value))
        assert(abs(mean(p_values) - a_real / (a_real + b_real))< .01)
        assert(abs(var(p_values) - (a_real*b_real) / (a_real + b_real)**2 / (a_real + b_real + 1)) < .001)
              
    def check_GammaPoisson(self):
        mu = Gamma('mu',value=3.,alpha=1.,beta=1.)
        d_list = []
        for i in xrange(10):
            d_list.append(Poisson('d_%i'%i,mu))
        beta_real = 1. + len(d_list)
        alpha_real = 1. + sum([d.value for d in d_list])
        
        mu_stepper = GammaPoisson(mu)
        
        mu_values = empty(10000,dtype=float)
        for i in xrange(10000):
            mu_stepper.step()
            mu_values[i] = mu.value
        
        assert(abs(mean(mu_values)-alpha_real/beta_real)<.05)
        assert(abs(var(mu_values) - alpha_real/beta_real**2)<.05)
        
    def check_GammaExponential(self):
        beta = Gamma('beta',value=3.,alpha=1.,beta=1.)
        d_list = []
        ld_list = []
        for i in xrange(10):
            d_list.append(Exponential('d_%i'%i,beta))
        for i in xrange(10):
            L = LinearCombination('L_%i'%i,[2],[beta])
            ld_list.append(Exponential('ld_%i'%i,L))
    
        alpha_real = 1. + len(d_list) + len(ld_list)
        beta_real = 1. + sum([d.value for d in d_list]) + sum([2.*d.value for d in ld_list])
        
        beta_stepper = GammaExponential(beta)
    
        beta_values = empty(10000,dtype=float)
        for i in xrange(10000):
            beta_stepper.step()
            beta_values[i] = beta.value
        
        assert(abs(mean(beta_values)-alpha_real/beta_real)<.1)
        assert(abs(var(beta_values) - alpha_real/beta_real**2)<.05)
        
    def check_GammaGamma(self):
        beta = Gamma('beta',value=3.,alpha=1.,beta=1.)
        d_list = []
        ld_list = []
        for i in xrange(10):
            d_list.append(Gamma('d_%i'%i,i+1,beta))
        for i in xrange(10):
            L = LinearCombination('L_%i'%i,[2],[beta])
            ld_list.append(Gamma('ld_%i'%i,i+1,L))
    
        alpha_real = 1. + 2.*sum(arange(1,11))
        beta_real = 1. + sum([d.value for d in d_list]) + sum([2.*d.value for d in ld_list])
    
        beta_stepper = GammaGamma(beta)
    
        beta_values = empty(10000,dtype=float)
        for i in xrange(10000):
            beta_stepper.step()
            beta_values[i] = beta.value
    
        assert(abs(mean(beta_values)-alpha_real/beta_real)<.1)
        assert(abs(var(beta_values) - alpha_real/beta_real**2)<.05)
        
    def check_BetaGeometric(self):
        p = Beta('beta',value=.5,alpha=1.,beta=1.)
        d_list = []
        for i in xrange(10):
            d_list.append(Geometric('d_%i'%i,p))
        alpha_real = 1. + len(d_list)
        beta_real = 1. + sum([d.value for d in d_list])
    
        p_stepper = BetaGeometric(p)
    
        p_values = empty(10000,dtype=float)
        for i in xrange(10000):
            p_stepper.step()
            p_values[i] = p.value
    
        assert(abs(mean(p_values)-alpha_real/(alpha_real + beta_real))<.01)
        assert(abs(var(p_values) - (alpha_real*beta_real) / (alpha_real + beta_real)**2 / (alpha_real + beta_real + 1)) < .001)
    
    def check_GammaNormal(self):
        tau = Gamma('tau', value=3., alpha=1., beta=1.)
    
        d1 = Normal('d1',mu=ones(3)*.1,tau=tau,value=ones(3))
        L = LinearCombination('L',[tau],[2.])
        d2 = Normal('d2', mu=3.*ones(3), tau=L, value=ones(3))
        tau_stepper = GammaNormal(tau)
    
        tau_values = empty(10000,dtype=float)
        for i in xrange(10000):
            tau_stepper.step()
            tau_values[i] = tau.value
    
        beta_real = 1.+(np.sum((d1.value-.1)**2) + np.sum((d2.value-3.)**2)*L.y[0])/2.
        alpha_real = 4.
    
        assert(abs(mean(tau_values)- alpha_real / beta_real)<.05)
        assert(abs(var(tau_values)- alpha_real / beta_real ** 2)<.05)
    
    def check_DirichletMultinomial(self):
        p = Dirichlet('p',ones(10)*3.)
        d_list = []
        for i in xrange(10):
            d_list.append(Multinomial('d_%i'%i, sum(arange(10)), p, value=arange(10)))
    
        p_stepper = DirichletMultinomial(p)
        p_values = empty((10,10000),dtype=float)
        for i in xrange(10000):
            p_stepper.step()
            p_values[:,i] = p.value
    
        mean_p_values = mean(p_values,axis=-1)
        var_p_values = var(p_values,axis=-1)
    
        theta_real = 10.*arange(10) + 3.*ones(10)
        real_mean = theta_real / sum(theta_real)
        real_var = theta_real * (sum(theta_real) - theta_real) / sum(theta_real)**2/(sum(theta_real)+1)
    
        assert((abs(real_mean-mean_p_values)/real_mean).max()<.01)
        assert((abs(real_var-var_p_values)/real_mean).max()<.01)
        
    def check_WishartMvNormal(self):
        tau = Wishart('tau',10,(asmatrix(eye(3)+.1*ones((3,3))))*10.)
        orig_value = tau.value
    
        d_list = []
        for i in xrange(5000):
            d_list.append(MvNormal('d_%i'%i,zeros(3),tau))
    
        tau_stepper = WishartMvNormal(tau)
    
        val_mat = asmatrix(empty((len(d_list),3)))
        for i in xrange(len(d_list)):
            val_mat[i,:] = d_list[i].value
    
        tau_values = empty((3,3,1000),dtype=float)
        for i in xrange(1000):
            tau_stepper.step()
            # tau.random()
            tau_values[:,:,i] = tau.value
    
        avg_tau_value = mean(tau_values, axis=-1)
        delta = avg_tau_value - orig_value
    
        assert(np.abs(np.asarray(delta)/np.asarray(orig_value)).max()<.1)

    def check_ClusterModel(self):
        mu = Gamma('mu',value=3.,alpha=1.,beta=1.)
        other_mu = 30.
        d_list = []
        indices = []
        mu_list = []
        for i in xrange(10):
            indices.append(randint(2))
            mu_list.append(Index('mu_list[%i]'%i, x=[mu, other_mu], index=indices[i]))
            d_list.append(Poisson('d_%i'%i,mu_list[i]))

        beta_real = 1.
        alpha_real = 1.
        for i in xrange(len(mu_list)):
            if mu_list[i].index.value == 0:
                beta_real += 1.
                alpha_real += sum(d_list[i].value)

        mu_stepper = GammaPoisson(mu)

        mu_values = empty(10000,dtype=float)
        for i in xrange(10000):
            mu_stepper.step()
            mu_values[i] = mu.value

        assert(abs(mean(mu_values)-alpha_real/beta_real)<.05)
        assert(abs(var(mu_values) - alpha_real/beta_real**2)<.05)    

    def check_NormalNormal(self):
        A = Normal('A',1,1)
        B = Normal('B',A,2*np.ones(2))
        C_tau = np.diag([.5,.5])
        C_tau[0,1] = C_tau[1,0] = .25
        C = MvNormal('C',B, C_tau, observed=True)
        D_mean = LinearCombination('D_mean', x=[np.ones((3,2))], y=[C])

        D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
        # D = Normal('D',D_mean,.5*np.ones(3))
        G = NormalSubmodel([B,C,A,D,D_mean])
        N = NormalModel(G)
        all_stepper = NormalNormal([A,B,C,D])

        all_values = np.empty((10000,6),dtype=float)
        for i in xrange(10000):
            all_stepper.step()
            for s in G.changeable_stochastic_list:
                all_values[i,G.changeable_slices[s]]=s.value

        assert(abs(mean(all_values, axis=0) - N.mu[G.changeable_stochastic_list]).max() < .1)
        assert(abs(var(all_values, axis=0) - np.diag(N.C[G.changeable_stochastic_list])).max() < .1)        
        
        
if __name__ == '__main__':
    NumpyTest().run()

