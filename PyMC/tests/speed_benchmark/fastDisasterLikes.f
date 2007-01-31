      SUBROUTINE fpoisson(x,early_mean,late_mean,switchpoint,n,like)
      
c Poisson log-likelihood function      
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py real dimension(1),intent(in) :: early_mean
cf2py real dimension(1),intent(in) :: late_mean_mean
cf2py integer dimension(1),intent(in) :: switchpoint
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
     
      INTEGER x(n)
      REAL early_mean, late_mean
      REAL like
      INTEGER n,i,switchpoint

      sumx = 0.0
      sumfact = 0.0
      do i=1,switchpoint
        like = like + x(i)*log(early_mean) - early_mean
      enddo
      do i=switchpoint+1,n
        like = like + x(i)*log(late_mean) - late_mean
      enddo

      return
      END
