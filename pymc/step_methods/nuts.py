from quadpotential import *
from arraystep import *
from ..core import *
from numpy import exp
from numpy.random import uniform
from hmc import *

__all__ = ['NoUTurn']

#TODO:
#add constraint handling via page 37 of Radford's http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html


class NoUTurn(ArrayStep):
    def __init__(self, vars, C, step_scale = .25, is_cov = False, state = None, model = None):
        """
        Parameters
        ----------
            vars : list of Theano variables
            C : array_like, ndim = {1,2}
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4) (defaults to .25)
            path_length : float, default=2

                total length to travel
            is_cov : bool, default=False
                Treat C as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            step_rand : function float -> float, default=unif
                A function which takes the step size and returns an new one used to randomize the step size at each iteration.
            model : Model
        """
        model = modelcontext(model)
        n = C.shape[0]

        self.step_size = step_scale / n**(1/4.)

        self.potential = quad_potential(C, is_cov, as_cov = False)

        if state is None:
            state = SamplerHist()
        self.state = state
        self.Emax = 1000


        ArrayStep.__init__(self,
                vars, [model.logpc, model.dlogpc(vars)]
                )

    def astep(self, q0, logp, dlogp):
        H = Hamiltonian(logp, dlogp, self.potential)
        Emax = self.Emax
        e = self.step_size

        p0 = H.pot.random()
        u = uniform(0, exp(-energy(H,q0,p0)))
        q = qn = qp = q0
        p = pn = pp = p0



        n=1
        s=1
        j=0

        while s == 1:
            v = bern(.5) * 2 -1

            if v == -1:
                qn,pn,_,_, q1,n1,s1 = buildtree(H, qn,pn,u, v,j,e, Emax)
            else:
                _,_,qp,pp, q1,n1,s1 = buildtree(H, qp,pp,u, v,j,e, Emax)

            if s1 == 1 and bern(min(1, n1*1./n)):
                q = q1

            n = n + n1

            span = qp - qn
            s = s1 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            j = j + 1

        p = -p
        return q

def buildtree(H, q, p, u, v, j,e, Emax):
    if j == 0:
        q1, p1 = leapfrog(H, q, p, 1, v*e)
        E = energy(H, q1,p1)
        n1 = u <= exp(-E)
        s1 = u <  exp(Emax -E)
        return q1, p1, q1, p1, q1, n1, s1
    else:
        qn,pn,qp,pp, q1,n1,s1 = buildtree(H, q,p,u, v,j - 1,e, Emax)
        if s1 == 1:
            if v == -1:
                qn,pn,_,_, q11,n11,s11 = buildtree(H, qn,pn,u, v,j - 1,e, Emax)
            else:
                _,_,qp,pp, q11,n11,s11 = buildtree(H, qp,pp,u, v,j - 1,e, Emax)

            if bern(n11*1./(n1 + n11)):
                q1 = q11


            span = qp - qn
            s1 = s11 * (span.dot(pn) >= 0) * (span.dot(pp) >= 0)
            n1 = n1 + n11
        return qn, pn, qp, pp, q1, n1, s1
    return
