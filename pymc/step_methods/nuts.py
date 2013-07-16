from quadpotential import *
from arraystep import *
from ..core import *
from numpy import exp, log
from numpy.random import uniform
from hmc import leapfrog, Hamiltonian, bern, energy
from ..tuning import guess_scaling

__all__ = ['NUTS']

class NUTS(ArrayStep):
    """
    Automatically tunes step size and adjust number of steps for good performance.

    Implements "Algorithm 6: Efficient No-U-Turn Sampler with Dual Averaging" in:

    Hoffman, Matthew D., & Gelman, Andrew. (2011).
    The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
    """
    def __init__(self, vars=None, scaling=None, step_scale = .25, is_cov = False, state = None,
                 Emax = 1000,
                 target_accept = .65,
                 gamma = .05,
                 k =.75,
                 t0 = 10,
                 model = None):
        """
        Parameters
        ----------
            vars : list of Theano variables, default continuous vars
            scaling : array_like, ndim = {1,2} or point
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4)
            is_cov : bool, default=False
                Treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            state
                state to start from
            Emax : float, default 1000
                maximum energy
            target_accept : float (0,1) default .65
                target for avg accept probability between final branch and initial position
            gamma : float, default .05
            k : float (.5,1) default .75
                scaling of speed of adaptation
            t0 : int, default 10
                slows inital adapatation
            model : Model
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars

        if scaling is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(Point(scaling, model=model), model=model)



        n = scaling.shape[0]

        self.step_size = step_scale / n**(1/4.)


        self.potential = quad_potential(scaling, is_cov, as_cov=False)

        if state is None:
            state = SamplerHist()
        self.state = state
        self.Emax = Emax

        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.k = k

        self.Hbar = 0
        self.u = log(self.step_size*10)
        self.m = 0



        ArrayStep.__init__(self,
                vars, [model.logpc, model.dlogpc(vars)]
                )

    def astep(self, q0, logp, dlogp):
        H = Hamiltonian(logp, dlogp, self.potential)
        Emax = self.Emax
        e = self.step_size

        p0 = H.pot.random()
        u = uniform()
        q = qn = qp = q0
        p = pn = pp = p0

        n=1
        s=1
        j=0

        while s == 1:
            v = bern(.5) * 2 -1

            if v == -1:
                qn,pn,_,_, q1,n1,s1,a,na = buildtree(H, qn,pn,u, v,j,e, Emax, q0, p0)
            else:
                _,_,qp,pp, q1,n1,s1,a,na = buildtree(H, qp,pp,u, v,j,e, Emax, q0, p0)

            if s1 == 1 and bern(min(1, n1*1./n)):
                q = q1

            n = n + n1

            span = qp - qn
            s = s1 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            j = j + 1

        p = -p

        w = 1./(self.m+self.t0)
        self.Hbar = (1 - w)* self.Hbar + w*(self.target_accept - a*1./na)

        self.step_size = exp(self.u - (self.m**.5/self.gamma)*self.Hbar)
        self.m += 1



        return q

def buildtree(H, q, p, u, v, j,e, Emax, q0, p0):
    if j == 0:
        q1, p1 = leapfrog(H, q, p, 1, v*e)
        E = energy(H, q1,p1)
        E0 = energy(H, q0,p0)

        dE = E - E0

        n1 = int(log(u) + dE  <= 0)
        s1 = int(log(u) + dE < Emax)
        return q1, p1, q1, p1, q1, n1, s1, min(1, exp(-dE)), 1
    else:
        qn,pn,qp,pp, q1,n1,s1,a1, na1 = buildtree(H, q,p,u, v,j - 1,e, Emax, q0, p0)
        if s1 == 1:
            if v == -1:
                qn,pn,_,_, q11,n11,s11,a11,na11 = buildtree(H, qn,pn,u, v,j - 1,e, Emax, q0, p0)
            else:
                _,_,qp,pp, q11,n11,s11,a11,na11 = buildtree(H, qp,pp,u, v,j - 1,e, Emax, q0, p0)

            if bern(n11*1./(max(n1 + n11, 1))):
                q1 = q11

            a1 = a1 + a11
            na1 = na1 + na11

            span = qp - qn
            s1 = s11 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            n1 = n1 + n11
        return qn, pn, qp, pp, q1, n1, s1, a1, na1
    return
