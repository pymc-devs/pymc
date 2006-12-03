      SUBROUTINE normal(x,mu,tau,n,n_mu,n_tau,like)

c Normal log-likelihood function      

cf2py real dimension(n),intent(in) :: x
cf2py real dimension(n_mu),intent(in) :: mu
cf2py real dimension(n_tau),intent(in) :: tau
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: n_mu=len(mu)
cf2py integer intent(hide),depend(tau) :: n_tau=len(tau)
      
      INTEGER n,n_mu,n_tau,i
      REAL like
      REAL x(n),mu(n_mu),tau(n_tau)
      LOGICAL mu_is_scalar, tau_is_scalar

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 

c
c Uncomment the following to print lengths of x,mu,tau
c      print *, 'length of x: ',n
c      print *, 'length of mu: ',n_mu,'length of tau: ',n_tau
c
      mu_is_scalar = (n_mu .LT. n)
	  tau_is_scalar = (n_tau .LT. n)
      like = 0.0
      do i=1,n
        if (tau_is_scalar) then
          tau_now = tau(1)
        else
          tau_now = tau(i)
        endif
        if (mu_is_scalar) then
          mu_now = mu(1)
        else
          mu_now = mu(i)
        endif
        like = like - 0.5 * tau_now * (x(i)-mu_now)**2
        like = like + 0.5*log(0.5*tau_now/PI)
      enddo
      return
      END

