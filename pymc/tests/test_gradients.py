from pymc import *
from numpy import *
from numpy.testing import *
import nose
import sys
from pymc import utils
import pymc 

float_dtypes = [float, single, float_, longfloat]

def find_variable_set(stochastic):
    set = [stochastic]
    for parameter, variable in stochastic.parents.iteritems():
        if isinstance(variable, Variable):
            set.append(variable)
    return set

class test_gradients(TestCase):
    
    def check_jacobians(self, deterministic):
        
        for parameter, pvalue in deterministic.parents.iteritems():
            
            if isinstance(pvalue, Variable): 
                
                grad = random.normal(.5, .1, size = shape(pvalue.value))
                
                a_partial_grad = self.get_analytic_partial_gradient(deterministic, parameter, pvalue, grad)
                
                n_partial_grad = self.get_numeric_partial_gradient(deterministic, pvalue, grad)

                assert_array_almost_equal(a_partial_grad, n_partial_grad,4,
                         "analytic partial gradient for " + str(deterministic) +
                         " with respect to parameter " + str(parameter) +
                         " is not correct.")
                
                
        
    def get_analytic_partial_gradient(self, deterministic, parameter, variable, grad):
        p = pymc.PyMCObjects.params(deterministic.parents) 
        
        jacobian = deterministic._jacobians[parameter]( **p)

        mapping = deterministic._jacobian_formats.get(parameter, 'full')

            
        return deterministic._format_mapping[mapping](deterministic, variable, jacobian, grad)
        
                
    def get_numeric_partial_gradient(self, deterministic, pvalue, grad ):
        pg = deterministic._format_mapping['full'](deterministic, pvalue, self.get_numeric_jacobian(deterministic, pvalue), grad)
        return reshape(pg, shape(pvalue.value))
    
    def get_numeric_jacobian(self, deterministic, pvalue ): 
        e = 1e-10
        initial_pvalue = pvalue.value
        shape = initial_pvalue.shape
        size = initial_pvalue.size
        
        initial_value = ravel(deterministic.value)

        numeric_jacobian= zeros((deterministic.value.size,size))
        for i in range(size):
            
            delta = zeros(size)
            delta[i] += e
            
            pvalue.value = reshape(initial_pvalue.ravel() + delta, shape)
            value = ravel(deterministic.value)

            numeric_jacobian[:, i] = (value - initial_value)/e
        
        pvalue.value = initial_pvalue
        return numeric_jacobian
        
        
    def test_jacobians(self):
        shape = (3, 10)
        a = Normal('a', mu = zeros(shape), tau = ones(shape))
        b = Normal('b', mu = zeros(shape), tau = ones(shape))
        
        addition = a + b 
        self.check_jacobians(addition)
        
        subtraction = a - b
        self.check_jacobians(subtraction)
        
        multiplication = a * b
        self.check_jacobians(subtraction)
        
        division = a / b
        self.check_jacobians(division)
        
        a2 = Uniform('a2', lower = .1 * ones(shape), upper = 2.0 * ones(shape))
        powering = a2 ** b
        self.check_jacobians(powering)
        
        negation = -a
        self.check_jacobians(negation)
        
        posing = +a 
        self.check_jacobians(posing)
        
        absing = abs(a)
        self.check_jacobians(absing)
        

    def check_gradients(self, stochastic):
        
        stochastics = find_variable_set(stochastic)
        gradients = utils.grad_logp_of_set(stochastics, stochastics)

        for s, analytic_gradient in gradients.iteritems():

                numeric_gradient = self.get_numeric_gradient(stochastics, s)
                
                assert_array_almost_equal(numeric_gradient, analytic_gradient,4,
                                         "analytic gradient for " + str(stochastic) +
                                         " with respect to parameter " + str(s) +
                                         " is not correct.")
        

    def get_numeric_gradient(self, stochastic, pvalue ): 
        e = 1e-10
        initial_value = pvalue.value
        shape = initial_value.shape
        size = initial_value.size
 
        initial_logp = utils.logp_of_set(stochastic)
        numeric_gradient = zeros(size)
        if not (pvalue.dtype in float_dtypes):
            return numeric_gradient 
         
        for i in range(size):
            
            delta = zeros(size)
            delta[i] += e
            
            pvalue.value = reshape(initial_value.ravel() + delta, shape)
            logp = utils.logp_of_set(stochastic)
            
            numeric_gradient[i] = (logp - initial_logp)/e
        
        pvalue.value = initial_value
        return numeric_gradient
        
    def test_gradients(self):
        
        shape = (5,)
        a = Normal('a', mu = zeros(shape), tau = ones(shape))
        b = Normal('b', mu = zeros(shape), tau = ones(shape))
        b2 = Normal('b2', mu = 2, tau = 1.0)
        c = Uniform('c', lower = ones(shape) * .7, upper = ones(shape) * 2.5 )
        d = Uniform('d', lower = ones(shape) * .7, upper = ones(shape) * 2.5 )
        e = Uniform('e', lower = ones(shape) * .2, upper = ones(shape) * 10)
        f = Uniform('f' , lower = ones(shape) * 2, upper = ones(shape) * 30)
        p = Uniform('p', lower = zeros(shape) +.05 , upper = ones(shape)  -.05 )
        n = 5
        
        
        a.value = 2 * ones(shape)
        b.value = 2.5 * ones(shape)
        b2.value = 2.5
        
        norm = Normal('norm', mu = a, tau = b)
        self.check_gradients(norm)
        
        norm2 = Normal('norm2', mu = 0, tau = b2)
        self.check_gradients(norm2)
        
        gamma = Gamma('gamma', alpha = a, beta = b)
        self.check_gradients(gamma)
        
        bern = Bernoulli('bern',p = p)
        self.check_gradients(bern )

        beta = Beta('beta', alpha = c, beta = d)

        self.check_gradients(beta)
        
        cauchy = Cauchy('cauchy', alpha = a, beta = d)
        self.check_gradients(cauchy)
        
        chi2 = Chi2('chi2', nu = e)

        self.check_gradients(chi2)
        
        exponential = Exponential('expon', beta = d)
        self.check_gradients(exponential)
        
        t = T('t', nu = f)
        self.check_gradients(t)
        
        half_normal = HalfNormal('half_normal', tau = e)
        self.check_gradients(half_normal)
        
        inverse_gamma = InverseGamma ('inverse_gamma', alpha = c, beta = d)
        self.check_gradients(inverse_gamma)
        
        laplace = Laplace('laplace', mu = a , tau = c)
        self.check_gradients(laplace)
        
        lognormal = Lognormal('lognormal', mu = a, tau = c)
        self.check_gradients(lognormal)
        
        weibull = Weibull('weibull', alpha = c, beta = d)
        self.check_gradients(weibull)
        
        binomial = Binomial('binomial', p = p, n = n)
        self.check_gradients(binomial)
        
        geometric = Geometric('geometric', p = p)
        self.check_gradients(geometric)
        
        poisson = Poisson('poisson', mu = c)
        self.check_gradients(poisson)
        
        u = Uniform('u', lower = a, upper = b)
        self.check_gradients(u)
        
        negative_binomial = NegativeBinomial('negative_binomial', mu = c, alpha = d )
        self.check_gradients(negative_binomial)
        
        #exponweib = Exponweib('exponweib', alpha = c, k =d , loc = a, scale = e )
        #self.check_gradients(exponweib)
        
        
    def check_model_gradients(self, model):

        # find the markov blanket 
        children = set([])
        for s in model:
            for s2 in s.extended_children:
                if isinstance(s2, Stochastic) and s2.observed == True:
                    children.add( s2)
        

        # self.markov_blanket is a list, because we want self.stochastics to have the chance to
        # raise ZeroProbability exceptions before self.children.
        markov_blanket = list(model)+list(children)

        
        gradients = utils.grad_logp_of_set(model)
        for variable in model:
            
            analytic_gradient = gradients[variable]
            
            numeric_gradient = self.get_numeric_model_gradient(markov_blanket, variable)
            
            assert_array_almost_equal(numeric_gradient, analytic_gradient,3,
                                     "analytic gradient for model " + str(model) +
                                     " with respect to variable " + str(variable) +
                                     " is not correct.")
        

    def get_numeric_model_gradient(self, model, variable): 
        e = 1e-9
        initial_value = variable.value
        shape = initial_value.shape
        size = initial_value.size
        
        initial_logp = utils.logp_of_set(model)
        
        numeric_gradient = zeros(size)
        for i in range(size):
            
            delta = zeros(size)
            delta[i] += e
            
            variable.value = reshape(initial_value.ravel() + delta, shape)
            logp = utils.logp_of_set(model)
            
            numeric_gradient[i] = (logp - initial_logp)/e
        
        variable.value = initial_value
        return numeric_gradient
        
    def test_model(self):
        import model1

        model = model1.model()
        model[0].value = 55.0
        model[1].value = 10.2
        model[2].value = .5 
        
        self.check_gradients(model[0])
        
        self.check_model_gradients(model)
        
        

if __name__ == '__main__':
    C =nose.config.Config(verbosity=1)
    nose.runmodule(config=C)
