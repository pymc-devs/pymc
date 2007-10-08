from PyMC2 import *

def disaster_no_r():
    @discrete_stoch
    def s(value=50, length=110):
        """Change time for rate stoch."""
        return 0.

    @stoch
    def e(value=1., rate=1.):
        """Rate stoch of poisson distribution."""
        return 0.

    @stoch
    def l(value=.1, rate = 1.):
        """Rate stoch of poisson distribution."""
        return 0.
        
    @data(discrete=True)
    def D(  value = 0.,
            switchpoint = s,
            early_rate = e,
            late_rate = l):
        """Annual occurences of coal mining disasters."""
        return 0.
        
    return locals()
        
M = Model(disaster_no_r())
M.DAG(consts=False, path='DisasterModel.dot', format='raw', legend=False)

def disaster_yes_r():
    @discrete_stoch
    def s(value=50, length=110):
        """Change time for rate stoch."""
        return 0.

    @stoch
    def e(value=1., rate=1.):
        """Rate stoch of poisson distribution."""
        return 0.

    @stoch
    def l(value=.1, rate = 1.):
        """Rate stoch of poisson distribution."""
        return 0.
    
    @dtrm
    def r(switchpoint = s,
        early_rate = e,
        late_rate = l):
        return 0.
    
    
    @data(discrete=True)
    def D(  value = 0.,
            rate = r):
        """Annual occurences of coal mining disasters."""
        return 0.
        
    return locals()
        
M = Model(disaster_yes_r())
M.DAG(consts=False, path='DisasterModel2.dot', format='raw', legend=False)

def dtrm_pre():
    @stoch
    def A(value=0):
        return 0.
        
    @stoch
    def B(value=0):
        return 0.
        
    @dtrm
    def C(p1=A, p2=B):
        return 0.
        
    @stoch
    def D(value=0, C = C):
        return 0.
        
    @stoch
    def E(value=0, C=C):
        return 0.
        
    return locals()
    
M = Model(dtrm_pre())
M.DAG(consts=False, path='DeterministicPreInheritance.dot', format='raw', legend=False)    
    
def dtrm_post():
    @stoch
    def A(value=0):
        return 0.
        
    @stoch
    def B(value=0):
        return 0.
        
    @stoch
    def D(value=0, C_p1 = A, C_p2=B):
        return 0.
        
    @stoch
    def E(value=0, C_p1=A, C_p2 = B):
        return 0.
        
    return locals()
    
M = Model(dtrm_post())
M.DAG(consts=False, path='DeterministicPostInheritance.dot', format='raw', legend=False)    
    
    
def survival():
    @stoch
    def beta(value=0):
        return 0.
        
    @data
    @stoch
    def x(value=0):
        return 0.
        
    @dtrm
    def S(covariates = x, coefs = beta):
        return 0.
        
    @data
    @stoch
    def t(value=0, survival = S):
        return 0.
        
    @stoch
    def a(value=0):
        return 0.
        
    @stoch
    def b(value=0):
        return 0.
    
    @potential
    def gamma(survival = S, stoch1=a, stoch2=b):
        return 0.
    
    return locals()
    
M = Model(survival())
M.DAG(consts=False, path='SurvivalModel.dot', format='raw', legend=False)    
    
M = Model(survival())
M.DAG(consts=False, path='SurvivalModelCollapsed.dot', format='raw', legend=False, collapse_potentials=True)