import pymc
from NormalSubmodel import NormalSubmodel
import numpy as np

__all__ = ['fourier_form', 'poly_trend', 'DLM']

def fourier_form(t, omega, lam=1.):
    """
    F, G = fourier_form(t, omega[, lam])
    
    Returns diagonal block of system matrix G corresponding to
    fourier component with frequency 'omega' and growth rate 'lam', 
    and corresponding subvectors of design vector F.
    
    F and G will be lists indexed by t.
    """
    
    G = np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]]) * lam
    F = np.array([1.,0.])
        
    return [F]*t, [G]*t
    
def poly_trend(t, c):
    """
    F, G = poly_trend(t, c)
    
    Returns diagonal element of system matrix G corresponding to polynomial
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
    

# TODO: RecarrayContainer class.
# Hard part will be pulling apart existing recarray to get attributes.
# Need this before anything will work, it's the only way to expose 'stochastics', etc properly.
def dict_to_recarray(dict):
    return pymc.Container(dict)
    # if isinstance(dict, np.recarray):
    #     return dict
    # keys = dict.keys()
    # return np.rec.fromarrays([np.asarray(dict[key]) for key in keys], names=keys)

def fourier_components(omegas, T):
    F = {}
    G = {}
    for o in omegas:
        G['fourier_%f'%o] = [np.asmatrix([[np.cos(np.pi*o), -np.sin(np.pi*o)],[np.sin(np.pi*o), np.cos(np.pi*o)]])]*T
        F['fourier_%f'%o] = [np.array([1,0])]*(T+1)
    return F, G

def polynomial_components(orders, T):
    F = {}
    G = {}
    for o in orders:
        this_G = np.asmatrix(np.eye(o+1))
        for i in xrange(o):
            this_G[i,i+1] = 1
        this_F = np.zeros(o+1)
        this_F[0]=1
        G['polynomial_%i'%o] = [this_G]*T
        F['polynomial_%i'%o] = [this_F]*(T+1)
    return F,G

def combine_components(*comps):
    out = ()
    for comp in comps:
        this_dict = {}
        for dict_now in comp:
            this_dict.update(dict_now)
        out = out + (this_dict,)
    return out
    
class DLM(NormalSubmodel):
    def __init__(self, F, G, V, W, m_0, C_0, Y_vals = None):
        """
        D = DLM(F, G, V, W, m_0, C_0[, Y_vals])
        
        Returns special NormalSubmodel instance representing the dynamic
        linear model formed by F, G, V and W.
    
        Resulting probability model:

            theta[0] | m_0, C_0 ~ N(m_0, C_0)
        
            theta[t] | theta[t-1], G[t], W[t] ~ N(G[t] theta[t-1], W[t]), t = 1..T    

            Y[t] | theta[t], F[t], V[t] ~ N(F[t] theta[t], V[t]), t = 0..T
    
    
        Arguments F, G, V should be dictionaries keyed by name of component.
            F[comp], G[comp], V[comp] should be lists.
                F[comp][t] should be the design vector of component 'comp' at time t.
                G[comp][t] should be the system matrix.

        Argument W should be either a number between 0 and 1 or a dictionary of lists
        like V.
            If a dictionary of lists, W[comp][t] should be the system covariance or 
            variance at time t.
            If a scalar, W should be the discount factor for the DLM.
            
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

        self.comps = F.keys()
        
        self.F = dict_to_recarray(F)
        self.G = dict_to_recarray(G)
        self.V = pymc.ListContainer(V)
        if np.isscalar(W):
            self.discount = True
            self.delta = W
        else:
            self.W = dict_to_recarray(W)
            self.discount = False
            self.delta = None
        if self.discount:
            raise NotImplemented, "Have yet to code up the discount factor."
        self.m_0 = dict_to_recarray(m_0)
        self.C_0 = dict_to_recarray(C_0)
        self.T = len(self.V)
            
        theta = {}
        theta_mean = {}

        Y_mean = []
        Y = []
    
        # ==============
        # = Make theta =
        # ==============
        for comp in self.comps:
            # Is diagonal the covariance or variance?
            if isinstance(self.W[comp][0], pymc.Variable):
                diag = isvector(self.W[comp][0].value)
            else:
                diag = isvector(self.W[comp][0])

            if diag:
                # Normal variates if diagonal.
                theta[comp] = [pymc.Normal('%s[0]'%comp, m_0[comp], C_0[comp])]
            else:
                # MV normal otherwise.
                theta[comp] = [pymc.MvNormal('%s[0]'%comp, m_0[comp], C_0[comp])]
            
            theta_mean[comp] = []

            for t in xrange(1,self.T):

                theta_mean[comp].append(pymc.LinearCombination('%s_mean[%i]'%(comp, t), [G[comp][t-1]], [theta[comp][t-1]]))

                if diag:
                    # Normal variates if diagonal.
                    theta[comp].append(pymc.Normal('%s[%i]'%(comp,t), theta_mean[comp][t-1], W[comp][t-1]))
                else:
                    # MV normal otherwise.
                    theta[comp].append(pymc.MvNormal('%s[%i]'%(comp,t), theta_mean[comp][t-1], W[comp][t-1]))
                

        self.theta = dict_to_recarray(theta)
        self.theta_mean = dict_to_recarray(theta_mean)


        # ==========
        # = Make Y =
        # ==========
        Y_diag = isvector(self.V.value[0])

        for t in xrange(self.T):
            x_coef = []
            y_coef = []
        
            for comp in self.comps:
                x_coef.append(self.F[comp][t])
                y_coef.append(theta[comp][t])
                    
            Y_mean.append(pymc.LinearCombination('Y_mean[%i]'%t, x_coef, y_coef))
            if Y_diag:
                # Normal variates if diagonal.
                Y.append(pymc.Normal('Y[%i]'%t, Y_mean[t], V[t]))
            else:
                # MV normal otherwise.
                Y.append(pymc.MvNormal('Y[%i]'%t, Y_mean[t], V[t]))
            
            # If data provided, use it.
            if Y_vals is not None:
                Y[t].value = Y_vals[t]
                Y[t].isdata = True
            
        self.Y_mean = pymc.Container(np.array(Y_mean))
        self.Y = pymc.Container(np.array(Y))

        # No sense creating a NormalSubmodel here... just stay a ListContainer.
        NormalSubmodel.__init__(self, [F,G,W,V,m_0,C_0,Y,theta,theta_mean,Y_mean])
        
if __name__ == '__main__':
    from NormalModel import NormalModel
    # F, G, V, W, m_0, C_0
    T=1
    o=.1
    F_f, G_f = fourier_components(np.arange(1,3)*o, T)
    # F_p, G_p = polynomial_components(np.arange(1,2),T)
    # F, G = combine_components((F_f, F_p), (G_f, G_p))
    F, G = F_f, G_f
    
    comps = G.keys()
    V ={}
    W = {}
    m_0 = {}
    C_0 = {}
    
    from numpy import cos, sin, pi
    
    for i in xrange(len(comps)):
        comp = comps[i]
        this_sh = F[comp][0].shape
        W[comp] = [np.ones(this_sh)*1e3]*T
        m_0[comp] = np.zeros(this_sh)
        C_0[comp] = np.ones(this_sh)
    V = [200.]*(T+1)    
    
    D = DLM(F,G,V,W,m_0,C_0)
    old_value = D.theta.value
    
    # N = pymc.sandbox.GibbsStepMethods.NormalNormal(list(D.variables))
    # N = pymc.sandbox.NormalSubmodel.NormalSubmodel(D)
    new_value = D.theta.value
    N = NormalModel(D)