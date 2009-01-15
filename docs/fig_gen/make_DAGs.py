from pymc import *

def disaster_no_r():
    @stoch(discrete=True)
    def s(value=50, length=110):
        """Change time for rate stochastic."""
        return 0.

    @stochastic
    def e(value=1., rate=1.):
        """Rate stochastic of poisson distribution."""
        return 0.

    @stochastic
    def l(value=.1, rate = 1.):
        """Rate stochastic of poisson distribution."""
        return 0.

    @observed(discrete=True)
    def D(  value = 0.,
            switchpoint = s,
            early_rate = e,
            late_rate = l):
        """Annual occurences of coal mining disasters."""
        return 0.

    return locals()

M = Model(disaster_no_r())
#M.DAG(consts=False, path='DisasterModel.dot', format='raw', legend=False)
graph.graph(M, consts=False, path='DisasterModel.pdf', format='pdf', legend=False)

def disaster_yes_r():
    @stochastic(discrete=True)
    def s(value=50, length=110):
        """Change time for rate stochastic."""
        return 0.

    @stochastic
    def e(value=1., rate=1.):
        """Rate stochastic of poisson distribution."""
        return 0.

    @stochastic
    def l(value=.1, rate = 1.):
        """Rate stochastic of poisson distribution."""
        return 0.

    @deterministic
    def r(switchpoint = s,
        early_rate = e,
        late_rate = l):
        return 0.


    @observed(discrete=True)
    def D(  value = 0.,
            rate = r):
        """Annual occurences of coal mining disasters."""
        return 0.

    return locals()

M = Model(disaster_yes_r())
#M.DAG(consts=False, path='DisasterModel2.dot', format='raw', legend=False)
graph.graph(M, consts=False, path='DisasterModel2.pdf', format='pdf', legend=False)

def deterministic_pre():
    @stochastic
    def A(value=0):
        return 0.

    @stochastic
    def B(value=0):
        return 0.

    @deterministic
    def C(p1=A, p2=B):
        return 0.

    @stochastic
    def D(value=0, C = C):
        return 0.

    @stochastic
    def E(value=0, C=C):
        return 0.

    return locals()

M = Model(deterministic_pre())
#   M.DAG(consts=False, path='DeterministicPreInheritance.dot', format='raw', legend=False)
graph.graph(M,consts=False, path='DeterministicPreInheritance.pdf', format='pdf', legend=False)


def deterministic_post():
    @stochastic
    def A(value=0):
        return 0.

    @stochastic
    def B(value=0):
        return 0.

    @stochastic
    def D(value=0, C_p1 = A, C_p2=B):
        return 0.

    @stochastic
    def E(value=0, C_p1=A, C_p2 = B):
        return 0.

    return locals()

M = Model(deterministic_post())
#M.DAG(consts=False, path='DeterministicPostInheritance.dot', format='raw', legend=False)
graph.graph(M, consts=False, path='DeterministicPostInheritance.pdf', format='pdf', legend=False)

def survival():
    @stochastic
    def beta(value=0):
        return 0.

    @data
    @stochastic
    def x(value=0):
        return 0.

    @deterministic
    def S(covariates = x, coefs = beta):
        return 0.

    @data
    @stochastic
    def t(value=0, survival = S):
        return 0.

    @stochastic
    def a(value=0):
        return 0.

    @stochastic
    def b(value=0):
        return 0.

    @potential
    def gamma(survival = S, stochastic1=a, stochastic2=b):
        return 0.

    return locals()

M = Model(survival())
#M.DAG(consts=False, path='SurvivalModel.dot', format='raw', legend=False)
graph.graph(M, consts=False, path='SurvivalModel.pdf', format='pdf', legend=False)

M = Model(survival())
#M.DAG(consts=False, path='SurvivalModelCollapsed.dot', format='raw', legend=False, collapse_potentials=True)
graph.graph(M, consts=False, path='SurvivalModelCollapsed.pdf', format='pdf', legend=False, collapse_potentials=True)
