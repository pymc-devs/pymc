import pymc
from NormalSubmodel import NormalSubmodel
import numpy as np

__all__ = ['Fourier_form', 'power_trend', 'DLM']

def Fourier_form(t, omega, lam=1.):
    """
    F, G = Fourier_form(t, omega[, lam])
    
    Returns diagonal block of system matrix G corresponding to
    Fourier component with frequency 'omega' and growth rate 'lam', 
    and corresponding subvectors of design vector F.
    
    F and G will be lists indexed by t.
    """
    
    G = np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]]) * lam
    F = np.array([1.,0.])
        
    return [F]*t, [G]*t
    
def power_trend(t, c):
    """
    F, G = power_trend(t, c)
    
    Returns diagonal element of system matrix G corresponding to power
    trend t**c and corresponding design vetor.
    
    F and G will be lists indexed by t.
    """
    F = [1.]*t
    G = [0.,1.]
    for i in xrange(2,t):
        G.append((i/(i-1.))**c)
    
    return F,G

def isvector(arr):
    diag = False
    if np.isscalar(arr):
        diag = True
    elif len(np.shape(arr)) == 1:
        diag = True
    elif np.shape(arr)[0] != np.shape(arr)[1]:
        diag = True
    return diag
    
class DLM(NormalSubmodel):
    def __init__(self, F, G, V, W, m_0, C_0, Y_vals = None):
        """
        D = DLM(F, G, V, W, m_0, C_0[, Y_vals])
        
        Returns special GaussianSubmodel instance representing the dynamic
        linear model formed by F, G, V and W.
    
        Resulting probability model:

            theta[0] | m_0, C_0 ~ N(m_0, C_0)
        
            theta[t] | theta[t-1], G[t], W[t] ~ N(G[t] theta[t-1], W[t]), t = 1..T    

            Y[t] | theta[t], F[t], V[t] ~ N(F[t] theta[t], V[t]), t = 0..T
    
    
        Arguments F, G, V, W should be dictionaries keyed by name of component.
            F[comp], G[comp], V[comp] and W[comp] should be lists.
                F[comp][t] should be the design vector of component 'comp' at time t.
                G[comp][t] should be the system matrix.
                W[comp][t] should be the system covariance or variance at time t.
            
        Arguments V and Y_vals, if given, should be lists. 
            V[t] should be the observation covariance or variance at time t.
            Y_vals[t] should give the value of output Y at time t.
    
        Arguments m_0 and C_0 should be dictionaries keyed by name of component.
            m_0[comp] should be the mean of theta[comp][0].
            C_0[comp] should be the covariance or variance of theta[comp][0].
            
        Note: if multiple components are correlated in W or V, they should be made into
        a single component.
    
        D.comp is a handle to a list.
            D.comp[t] is a Stochastic representing the value of system state 'theta'
            sliced according to component 'comp' at time t.
        
        D.theta is a dictionary of lists analogous to F, G, V and W.
    
        D.Y is a list. D.Y[t] is a Stochastic representing the value of the output
        'Y' at time t.
        """

        self.F = pymc.DictContainer(F)
        self.G = pymc.DictContainer(G)
        self.V = pymc.ListTupleContainer(V)
        self.W = pymc.DictContainer(W)
        self.m_0 = pymc.DictContainer(m_0)
        self.C_0 = pymc.DictContainer(C_0)
        self.T = len(self.F)
        theta = {}
        theta_next_mean = {}

        Y_mean = []
        Y = []
    
        # ==============
        # = Make theta =
        # ==============
        for comp in self.F.iterkey():
            # Is diagonal the covariance or variance?
            diag = isvector(self.W[comp][0].value)

            if diag:
                # Normal variates if diagonal.
                theta[comp] = [pymc.Normal('%s_0'%comp, m_0[comp], C_0[comp])]
            else:
                # MV normal otherwise.
                theta[comp] = [pymc.MvNormalCov('%s_0'%comp, m_0[comp], C_0[comp])]
            
            theta_next_mean[comp] = [pymc.LinearCombination('m_%s_1'%comp, [G[comp][0]], [theta[comp][0]])]

            for t in xrange(self.T):

                theta_next_mean[comp].append(pymc.LinearCombination('m_%s_%i'%(comp, t), [G[comp][t]], [theta[comp][t-1]]))

                if diag:
                    # Normal variates if diagonal.
                    theta[comp].append(pymc.Normal('%s_%i'%(comp,t), theta_next_mean[comp][t], W[t]))
                else:
                    # MV normal otherwise.
                    theta[comp].append(pymc.MvNormalCov('%s_%i'%(comp,t), theta_next_mean[comp][t], W[t]))
                

        self.theta = pymc.DictContainer(theta)
        self.theta_next_mean = pymc.DictContainer(theta_next_mean)


        # ==========
        # = Make Y =
        # ==========
        Y_diag = isvector(self.V[0].value):

        for t in xrange(self.T):
            x_coef = []
            y_coef = []
        
            for comp in self.F.iterkeys():
                x_coef.append(self.F[comp][t])
                y_coef.append(theta[comp][t])
                    
            Y_mean.append(pymc.LinearCombination('Y_mean_%i'%t, x_coef, y_coef))
            if Y_diag:
                # Normal variates if diagonal.
                Y.append(pymc.Normal('Y_%i'%t, Y_mean[t], V[t]))
            else:
                # MV normal otherwise.
                Y.append(pymc.MvNormalCov('Y_%i'%t, Y_mean[t], V[t]))
            
            # If data provided, use it.
            if Y_vals is not None:
                Y[t].value = Y_vals[t]
                Y[t].isdata = True
            
        self.Y_mean = ListTupleContainer(Y_mean)
        self.Y = ListTupleContainer(Y)

    
        # Initialize Normal submodel.
        
        # Find all normal stochastics. Eventually do this with crawl() inside NormalSubmodel.
        normal_list = [self.Y, self.Y_mean, self.theta, self.theta_next_mean]
        need_reinit = False

        for comp in self.theta.iterkeys():

            for l in [self.F, self.G]:
                for t in xrange(self.T)
                    if l[comp][t].__class__ in normal_classes: 
                        normal_list.append(l[comp][t])
                    elif isinstance(l[comp][t], pymc.Stochastic):
                        need_reinit = True

            for l in [self.m_0, self.C_0]:
                if l[comp].__class__ in normal_classes:
                    normal_list.append(l[comp])            
                elif isinstance(l[t], pymc.Stochastic):
                    need_reinit = True
        
        # Only this line will remain once crawl is working!
        NormalSubmodel.__init__(self, normal_list)
        
        # Call ListTupleContainer.__init__ to claim all of self's stochastics, even those that
        # aren't Gaussian. Eventually take care of this with crawl() somehow.
        if need_reinit:
            ListTupleContainer.__init__(self, [self.Y, self.Y_mean, self.theta, self.theta_next_mean, 
                                                self.F, self.G, self.V, self.W, self.m_0, self.C_0])
    
        for comp in self.theta.iteritems():
            if self.hasattr(comp):
                print 'Warning: Name %s conflicts with a preexisting attribute and cannot be used as a component handle.'
            else:
                setattr(self, comp[0], comp[1])

