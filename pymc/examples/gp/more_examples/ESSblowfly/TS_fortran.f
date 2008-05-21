      subroutine TS_fortran(m,r,s,IC,tau,psi,phi,TS,T)

cf2py double precision dimension(T),intent(out):: TS
cf2py integer intent(hide),depend(psi):: T = shape(psi,0)
cf2py integer intent(hide),depend(IC)::tau = shape(IC,0)

      integer T, i, tau
      double precision TS(T), IC(tau), psi(T), phi(T)
      double precision m, r, s
      
      do i=1,tau
          TS(i) = IC(i)
      enddo

      do i=tau+1,T
          TS(i) = TS(i-1) + s*exp(-r*TS(i-tau))*TS(i-tau)*psi(i-tau)
          TS(i) = TS(i) - m * TS(i-1) * phi(i-1)
          if (TS(i) .LE. 0.0D0) then
              TS(i) = 0.0D0
          endif
      enddo

      return
      end