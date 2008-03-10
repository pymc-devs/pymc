"""
Gibbs step methods apply to conjugate submodels. In other words, if in the
following model:

B|A ~ d2(A, p2)
A ~ d1(p1)

d1 is a standard distribution and A's distribution conditional on B is d1 
with parameters p1_*, A can be Gibbs sampled using standard random variables.

If, on the other hand, the likelihood p(B|A) regarded as a function of A is 
proportional to a standard distribution d3, but A's distribution conditional
on its parents is /not/ d3, A can be Metropolis sampled using p(B|A) as a 
proposal distribution. In this case the Metropolis-Hastings acceptance 
threshold is

    min(1, p(A_p|parents) / p(A|parents)).
    
Each Gibbs step method has a fully conjugate version and a nonconjugate version.
"""

from pymc import *
import numpy as np

__all__ = ['GammaNormal', 'GammaPoisson', 'GammaExponential', 'GammaGamma', 'WishartMvNormal', 'DirichletMultinomial', 'BetaBinomial', 'BetaGeometric', 'BernoulliAnything', 'check_children', 'check_linear_extended_children', 'check_conjugacy','StandardGibbs']


# If we implement Pareto:
# ParetoUniform
# GammaPareto

# TODO: Finish GammaExponential,
# Bundle quad_term from GammaNormal and WishartMvNormal into StandardGibbs,
# Factor out GammaX base class,
# Depth-1 GammaGamma (screw the high-depth extension),
# Competence functions,
# Done!

# TODO, long-term: Allow long sequences of LinearCombinations.        
# TODO, long-term: Allow children to be of different classes, as long as they're all conjugate.

def check_children(stochastic, stepper_class, child_class, parent_key):
    """
    Make sure all children are of the required class,
    and that self.stochastic is the correct parent of each.
    """
    d = []
    parent_dict = {}
    for name in child_class.parent_names:
        if not name == parent_key:
            parent_dict[name] = []

    for child in stochastic.children:
        if isinstance(child, Stochastic):
            if not isinstance(child, child_class):
                raise ValueError, 'Stochastic children of %s must all be %s for %s\n \
                                    to be able to handle it.' \
                                    %(stochastic.__name__, child_class.__name__, stepper_class)
            d.append(child)
            for name in child_class.parent_names:
                if not name == parent_key:
                    parent_dict[name].append(child.parents[name])
    
    return d, parent_dict



def check_conjugacy(stochastic, target_class):
    """
    See whether the conjugate or non-conjugate sampling
    strategy should be used.
    """
    parent_dict = {}
    if not isinstance(stochastic, target_class):
        for name in stochastic.parents:
            parent_dict[name] = None
        conjugate = False

    else:
        for name in stochastic.parents:
            parent_dict[name] = lambda_deterministic(name, lambda parent = stochastic.parents[name]: parent)
        conjugate = True
        
    return conjugate, parent_dict



def check_linear_extended_children(stochastic, stepper_class, child_class, parent_key):
    """
    Make sure all extended children are of the required class,
    that correct parent is a LinearCombination, and that stochastic 
    appears only once in LinearCombination's coefficient list.
    """
    d = []
    coef = []
    parent_dict = {}
    side = []
    for name in child_class.parent_names:
        if not name == parent_key:
            parent_dict[name] = []

    for child in stochastic.extended_children:
        
        # If extended child is a direct child, make sure everything is in order.
        if child in stochastic.children:
            continue
        
        # If extended child is not direct, make sure direct child is LinearCombination
        # with parent stochastic.                
        elif isinstance(child.parents[parent_key], LinearCombination):

            p = child.parents[parent_key]
            if len(p.x) > 1 or len(p.y) > 1 or ((p.x[0] is stochastic) + (p.y[0] is stochastic) != 1):
                raise ValueError, 'Stochastic %s must appear only once as a parent of %s,\n \
and %s must have only two parents, for %s to apply.' \
                    % (stochastic.__name__, p.__name__, p.__name__, stepper_class)


            # Record the coefficient of stochastic in LinearCombination.    
            if p.x[0] is stochastic:
                coef.append(p.y[0])
                side.append('L')

            else:
                coef.append(p.x[0])
                side.append('R')
            
            d.append(child)    
            for name in child_class.parent_names:
                if not name == parent_key:
                    parent_dict[name].append(child.parents[name])
                
        else:
            raise ValueError, "Child %s's %s parent must be LinearCombination\n or %s itself for %s to apply."\
                %(child.__name__, parent_key, stochastic.__name__, stepper_class)
            
    return d, coef, side, parent_dict
    
def zap_extended_children(stochastic, cls_name):
    # Raise error if any extended children aren't direct children.
    if len(stochastic.extended_children - stochastic.children) > 0:
        raise ValueError, 'Stochastic %s must have only direct children for %s to apply.'\
            %(stochastic.__name__, cls_name)
    


class StandardGibbs(Gibbs):
    """
    All Gibbs steppers in GibbsStepMethods subclass StandardGibbs.
    It keeps the following attributes:
    - N_d: Number of direct children, total.
    - N: Total length of direct children.
    - sum_d: Sum of values of direct children.
    """
    linear_OK = False
    def __init__(self, stochastic, verbose=0):

        self.stochastic = stochastic

        # Get distributional parameters from children and make sure children are Exponential.
        self.d, self.parent_dict = check_children(self.stochastic, self.__class__.__name__, self.child_class, self.parent_label)

        # See whether to use conjugate or non-conjugate version.
        self.conjugate, parent_dict = check_conjugacy(self.stochastic, self.target_class)
        self.__dict__.update(parent_dict)

        self.N =  int(np.sum([safe_len(d_now.value) for d_now in self.d]))
        self.N_d = len(self.d)

        if self.linear_OK:
            # Get distributional parameters from extended children via LinearCombinations.
            self.ld, self.lcoef, self.lside, self.lparent_dict = \
                check_linear_extended_children(self.stochastic, self.__class__.__name__, self.child_class, self.parent_label)
            self.lN = int(np.sum([safe_len(d_now.value) for d_now in self.ld]))
            self.N_ld = len(self.ld)
        else:
            zap_extended_children(stochastic, self.__class__.__name__)

        @deterministic
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return np.sum([np.sum(d_now) for d_now in d])
        self.sum_d = sum_d

        Gibbs.__init__(self, stochastic, verbose)
                
    @classmethod
    def competence(cls, stochastic):
        
        for req_attr in ['child_class', 'target_class', 'parent_label', 'linear_OK']:
            if not hasattr(cls, req_attr):
                return 0
        
        import pymc
        
        try:
            junk = check_children(stochastic, cls.__name__, cls.child_class, cls.parent_label)
            if cls.linear_OK:
                junk = check_linear_extended_children(stochastic, cls.__name__, cls.child_class, cls.parent_label)
            else:
                zap_extended_children(stochastic, cls.__name__)
        except ValueError:
            return 0
        conjugate, junk = check_conjugacy(stochastic, cls.target_class)
        if conjugate:
            return pymc.conjugate_Gibbs_competence
        else:
            return pymc.nonconjugate_Gibbs_competence

# TODO: When allowing mixed conjugacy, allow Categorical children also.
class DirichletMultinomial(StandardGibbs):            
    """
    Applies to p in the following submodel:

    d_i ~ind Multinomial(n_i, p)
    p ~ Dirichlet(theta) [optional] 
    """
    linear_OK = False
    child_class = Multinomial
    parent_label = 'p'
    target_class = Dirichlet
    def __init__(self, p, verbose=0):
        
        StandardGibbs.__init__(self, p, verbose=0)
        
        @deterministic
        def like_theta(d=self.d, N_d = self.N_d):
            out = 0.
            for i in xrange(N_d):
                out += d[i]
            return out
        self.like_theta = like_theta
            

    def propose(self):

        theta = self.like_theta.value
        if self.conjugate:
            theta = theta + self.theta.value
        else:
            theta += 1.
        self.stochastic.value = np.random.dirichlet(theta)



class WishartMvNormal(StandardGibbs):
    """
    Applies to tau in the following submodel:

    d_i ~ind Normal(mu_i, tau)
    tau ~ Wishart(n, C) [optional]

    where the stochastics d are parametrized by precision, not covariance.
    """
    linear_OK = True
    child_class = MvNormal
    parent_label = 'tau'
    target_class = Wishart
    def __init__(self, tau, verbose=0):

        StandardGibbs.__init__(self, tau, verbose=verbose)

        @deterministic
        def like_Tau(N_d = self.N_d, d = self.d, parent_dict = self.parent_dict):
            quad_term = 0.
            for i in xrange(N_d):
                delta_now = d[i] - parent_dict['mu'][i]
                quad_term += np.outer(delta_now, delta_now)
            return np.asmatrix(quad_term)
            
        @deterministic
        def like_lin_Tau(N_ld = self.N_ld, ld = self.ld, lparent_dict = self.lparent_dict, lside = self.lside, lcoef = self.lcoef):
            quad_term = 0.
            for i in xrange(N_ld):
                delta_now =ld[i] - lparent_dict['mu'][i]
                if lside[i] == 'L':
                    quad_term += np.dot(np.outer(delta_now, delta_now), lcoef[i])
                else:
                    quad_term += np.dot(lcoef[i], np.outer(delta_now, delta_now))
            return np.asmatrix(quad_term)
            
        self.like_Tau, self.like_lin_Tau = like_Tau, like_lin_Tau

    def propose(self):
        n = self.N_d
        Tau = self.like_Tau.value + self.like_lin_Tau.value
        if self.conjugate:
            n += self.n.value
            Tau += self.Tau.value
        else:
            n += 1.

        self.stochastic.value = np.asmatrix(rwishart(n, Tau))        


class BetaX(StandardGibbs):
    """
    Base class for conjugacy involving Beta.
    """
    linear_OK = False
    target_class = Beta
    def __init__(self, stochastic, verbose=0):
        StandardGibbs.__init__(self, stochastic, verbose=verbose)
        
    def propose(self):
        alpha = self.like_alpha.value
        beta = self.like_beta.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
            beta += 1.
        self.stochastic.value = np.random.beta(alpha, beta)

class BetaGeometric(BetaX):
    """
    Applies to p in the following submodel:

    d_i ~ind Geometric(p)
    p ~ Beta(alpha, beta) [optional]
    """
    child_class = Geometric
    parent_label = 'p'
    def __init__(self, p, verbose=0):
        BetaX.__init__(self, p, verbose)

        @deterministic
        def like_alpha(N = self.N):
            return N
        
        self.like_alpha = like_alpha
        self.like_beta = self.sum_d
        

# TODO: When allowing mixed conjugacy, allow Bernoulli children also.
class BetaBinomial(BetaX):
    """
    Applies to p in the following submodel:

    d_i ~ind Binomial(n_i, p)
    p ~ Beta(alpha, beta) [optional]
    """
    child_class = Binomial
    parent_label = 'p'
    def __init__(self, p, verbose=0):

        BetaX.__init__(self, p, verbose)
        
        self.like_alpha = self.sum_d
        
        @deterministic
        def like_beta(N_d = self.N_d, d = self.d, parent_dict = self.parent_dict):
            out = 0.
            for i in xrange(N_d):
                out += np.sum(parent_dict['n'][i] - d[i])
            return out
        self.like_beta = like_beta

class GammaX(StandardGibbs):
    """
    Base class for conjugacy involving Gamma.
    """
    linear_OK = True
    target_class = Gamma
    def __init__(self, stochastic, verbose=0):

        StandardGibbs.__init__(self, stochastic, verbose=verbose)

        @deterministic
        def sum_ld(N_ld = self.N_ld, ld = self.ld, lcoef = self.lcoef, lside=self.lside):
            out = 0.
            for i in xrange(N_ld):
                if lside[i] == 'L':
                    out += np.sum(np.dot(np.transpose(lcoef[i]), ld[i]))
                else:
                    out += np.sum(np.dot(ld[i], np.transpose(lcoef[i])))
                    
            return out
        self.sum_ld = sum_ld
            
    def propose(self):
        alpha = self.like_alpha.value
        beta = self.like_beta.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
        self.stochastic.value = np.random.gamma(alpha, 1./beta)


class GammaExponential(GammaX):
    """
    Applies to beta in the following submodel:

    d_i ~ind Exponential(beta)
    beta ~ Gamma(...) [optional]

    Optionally, can apply to beta in:

    d_i ~ind Exponential(l_i)
    l_i = LinearCombination([beta],[y_i])
    beta ~ Gamma(...) [optional]

    where beta is one of the elements of either x or y.
    """
    child_class = Exponential
    parent_label = 'beta'
    def __init__(self, beta, verbose=0):

        self.stochastic = beta
        GammaX.__init__(self, beta, verbose)
        
        @deterministic
        def like_alpha(N = self.N, lN = self.lN):
            return N + lN
        self.like_alpha = like_alpha
        
        @deterministic
        def like_beta(sum_ld = self.sum_ld, sum_d = self.sum_d):
            return sum_d + sum_ld
        self.like_beta = like_beta
        

class GammaPoisson(GammaX):
    """
    Applies to mu in the following submodel:

    d_i ~ind Poisson(mu)
    mu ~ Gamma(...) [optional]
    
    Optionally can apply to mu in:
    d_i ~ind Poisson(l_i)
    l_i = LinearCombination([mu],[y_i])
    mu ~ Gamma(...) [optional]
    
    """
    child_class = Poisson
    parent_label = 'mu'
    def __init__(self, mu, verbose=0):

        self.stochastic = mu
        GammaX.__init__(self, mu, verbose)

        self.like_alpha = self.sum_d
        
        @deterministic
        def like_beta(d = self.ld, lN = self.lN, N = self.N, lcoef = self.lcoef, lside = self.lside):
            out = N
            for i in xrange(lN):
                if lside[i]=='L':
                    out += np.sum(np.sum(lcoef[i], axis=1))
                else:
                    out += np.sum(np.sum(lcoef[i], axis=0))
                    
            return out
        self.like_beta = like_beta
        

class GammaGamma(GammaX):
    """
    Applies to beta in the following submodel:

    d_i ~ind Gamma(alpha_i, beta)
    beta ~ Gamma(...) [optional]

    Optionally can apply to beta in:
    
    d_i ~ind Gamma(alpha_i, l_i)
    l_i = LinearCombination([beta],[y_i])
    beta ~ Gamma(...) [optional]

    """
    child_class = Gamma
    parent_label = 'beta'
    def __init__(self, beta, verbose=0):

        self.stochastic = beta
        GammaX.__init__(self, beta, verbose)

        @deterministic
        def like_alpha(lparent_dict = self.lparent_dict, parent_dict = self.parent_dict, N_d = self.N_d, N_ld = self.N_ld):
            out = 0.
            for i in xrange(self.N_d):
                out += np.sum(parent_dict['alpha'][i])
            for i in xrange(self.N_ld):
                out += np.sum(lparent_dict['alpha'][i])                
            return out
        self.like_alpha = like_alpha

        @deterministic
        def like_beta(sum_ld = self.sum_ld, sum_d = self.sum_d):
            return sum_d + sum_ld
        self.like_beta = like_beta


class GammaNormal(GammaX):
    """
    Applies to tau in the following submodel:
    
    d ~ind N(mu, tau * theta)
    tau ~ Gamma(alpha, beta) [optional]
    
    Optionally can apply to tau in:
    
    d ~ind N(mu, l_i * theta)
    l_i = LinearCombination([tau],[y_i])
    tau ~ Gamma(...) [optional]    
    """
    child_class = Normal
    parent_label = 'tau'
    
    def __init__(self, tau, verbose=0):
        
        self.stochastic = tau

        GammaX.__init__(self, tau, verbose)

        @deterministic    
        def like_alpha(N=self.N, lN=self.lN):
            return .5 * (N + lN)

        @deterministic
        def like_beta(d=self.d, ld=self.ld, parent_dict=self.parent_dict, lparent_dict=self.lparent_dict, lcoef=self.lcoef,
            lside=self.lside, N_d=self.N_d, N_ld=self.N_ld):
            quad_term = 0.
            for i in xrange(self.N_d):
                delta_now = d[i] - parent_dict['mu'][i]
                quad_term += np.dot(delta_now, delta_now)
            
            for i in xrange(N_ld):
                delta_now = ld[i] - lparent_dict['mu'][i]
                if lside[i] == 'L':
                    quad_term += np.dot(np.dot(delta_now, lcoef[i]), delta_now)
                else:
                    quad_term += np.dot(delta_now, np.dot(delta_now, lcoef[i]))
            
            return quad_term * .5
        
        self.like_alpha, self.like_beta = like_alpha, like_beta

    

class BernoulliAnything(Gibbs):
    """
    Formerly known as BinaryMetropolis. 
    """
    
    target_class = Bernoulli
    child_class = None
    parent_label = 'p'
    linear_OK = True
    
    def __init__(self, stochastic, dist=None):
        # BinaryMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stochastic, dist=dist)
        
        # Initialize verbose feedback string
        self._id = stochastic.__name__
        
    def set_stochastic_val(self, i, val, to_value):
        """
        Utility method for setting a particular element of a stochastic's value.
        """
        
        if self._len>1:
            # Vector-valued stochastics
            
            val[i] = to_value
            self.stochastic.value = reshape(val, check_type(self.stochastic)[1])
        
        else:
            # Scalar stochastics
            
            self.stochastic.value = to_value
    
    def step(self):
        """
        This method is substituted for the default step() method in
        BinaryMetropolis.
        """
            
        # Make local variable for value
        if self._len > 1:
            val = self.stochastic.value.ravel()
        else:
            val = self.stochastic.value
        
        for i in xrange(self._len):
            
            self.set_stochastic_val(i, val, True)
            
            try:
                logp_true = self.stochastic.logp
                loglike_true = self.loglike
            except ZeroProbability:
                self.set_stochastic_val(i, val, False)
                continue
            
            self.set_stochastic_val(i, val, False)
            
            try:
                logp_false = self.stochastic.logp
                loglike_false = self.loglike
            except ZeroProbability:
                self.set_stochastic_val(i,val,True)
                continue
            
            p_true = exp(logp_true + loglike_true)
            p_false = exp(logp_false + loglike_false)
            
            # Stochastically set value according to relative
            # probabilities of True and False
            if np.random.random() > p_true / (p_true + p_false):
                self.set_stochastic_val(i,val,True)
        
        # Increment accepted count
        self._accepted += 1
