

      DOUBLE PRECISION FUNCTION combinationln(n,k)

c Ln of the number of different combinations of n different things, taken k at a time. 
c DH, 5.02.2007

      IMPLICIT NONE
      INTEGER n, k
      DOUBLE PRECISION factln

      combinationln= factln(n) - factln(k) - factln(n-k)

      END FUNCTION combinationln

      SUBROUTINE expand_triangular(d, f, nf, t, n)
!       d is diagonal,
!       f is flattened lower triangle in column-major format,
!       t is unflattened triangular output matrix.
!       nf must be n * (n-1) / 2
cf2py intent(hide) nf
cf2py intent(hide) n
cf2py intent(out) t
       DOUBLE PRECISION f(nf), t(n,n), d(n)
       INTEGER i_f,i_t,j_t
       i_f = 0
       do j_t = 1,n
           t(j_t,j_t) = d(j_t)
           do i_t = j_t+1, n
               i_f = i_f + 1
               t(i_t,j_t) = f(i_f)
            end do
       end do
       
       RETURN
       END
       

      SUBROUTINE standardize(x, loc, scale, n, nloc, nscale, z)
      
c Compute z = (x-mu)/scale

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(n), intent(out) :: z
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)


      DOUBLE PRECISION x(n), loc(nloc), scale(nscale), z(n)
      DOUBLE PRECISION mu, sigma
      INTEGER n, nloc, nscale, i
      LOGICAL not_scalar_loc, not_scalar_scale
      
      mu = loc(1)
      sigma = scale(1)
      not_scalar_loc = (nloc .NE. 1)
      not_scalar_scale = (nscale .NE. 1)
      
      do i=1,n
        if (not_scalar_loc) mu = loc(i)     
        if (not_scalar_scale) sigma = scale(i)
        z(i) = (x(i) - mu)/sigma
      enddo
      END


      DOUBLE PRECISION FUNCTION gammln(xx) 
C Returns the value ln[gamma(xx)] for xx > 0. 

      DOUBLE PRECISION xx
      INTEGER j 
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6) 

C Internal arithmetic will be done in double precision, 
C a nicety that you can omit if five-figure accuracy is good enough. 

      SAVE cof,stp 
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0, 
     +24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2, 
     +-.5395239384953d-5,2.5066282746310005d0/ 
      x=xx
      y=x 
      tmp=x+5.5d0 
      tmp=(x+0.5d0)*dlog(tmp)-tmp 
      ser=1.000000000190015d0 
      do j=1,6
         y=y+1.d0 
         ser=ser+cof(j)/y 
      enddo 
      gammln=tmp+dlog(stp*ser/x) 
      return 
      END
      
      
      DOUBLE PRECISION FUNCTION mvgammln(x, k)
C Returns the logarithm of the multivariate gamma function for x > 0
      IMPLICIT NONE
      DOUBLE PRECISION PI, x
      DOUBLE PRECISION gammln
      PARAMETER (PI=3.141592653589793238462643d0)
      INTEGER j,k
      
      mvgammln = k * (k-1) / 4 * log(PI)
      
      do j=1,k
        mvgammln = mvgammln + gammln(x + (1-j)/2)
      enddo
      
      return
      END
      

      DOUBLE PRECISION FUNCTION factrl(n) 
C Returns the value n! as a floating-point number. 

      INTEGER n 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      INTEGER j,ntop 
C Table to be filled in only as required. 
      DOUBLE PRECISION a(33),gammln 
      SAVE ntop,a 
C Table initialized with 0! only. 
      DATA ntop,a(1)/0,1./

      if (n.lt.0) then 
c        write (*,*) 'negative factorial in factrl' 
        factrl=-infinity
        return
      else if (n.le.ntop) then 
C Already in table. 
        factrl=a(n+1) 
      else if (n.le.32) then 
C Fill in table up to desired value. 
        do j=ntop+1,n
          a(j+1)=j*a(j) 
        enddo
        ntop=n 
        factrl=a(n+1) 
      else 
C Larger value than size of table is required. Actually, 
C this big a value is going to overflow on many computers, 
C but no harm in trying. 
        factrl=dexp(gammln(n+1.d0)) 
      endif 
      return 
      END 

      DOUBLE PRECISION FUNCTION factln(n) 
C USES gammln Returns ln(n!). 

      INTEGER n 
      DOUBLE PRECISION a(100),gammln, pass_val 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
      SAVE a 
C Initialize the table to negative values. 
      DATA a/100*-1./ 
      pass_val = n + 1
      if (n.lt.0) then
c        write (*,*) 'negative factorial in factln' 
        factln=-infinity
        return
      endif
C In range of the table. 
      if (n.le.99) then
C If not already in the table, put it in.
        if (a(n+1).lt.0.) a(n+1)=gammln(pass_val) 
        factln=a(n+1) 
      else 
C Out of range of the table. 
        factln=gammln(pass_val) 
      endif 
      return 
      END


      SUBROUTINE categor(x,hist,mn,step,n,k,like)

c Categorical log-likelihood function
c Need to return -Infs when appropriate

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(k),intent(in) :: hist
cf2py double precision intent(in) :: mn,step
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(hist) :: k=len(hist)
cf2py double precision intent(out) :: like
            
      DOUBLE PRECISION hist(k),x(n),mn,step,val,like
      INTEGER n,k,i,j
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      like = 0.0
c loop over number of elements in x      
      do i=1,n
c initialize current value      
        val = mn
        j = 1
c check for appropriate bin        
    1   if (x(i).gt.val) then
c increment value    
          val = val + step
          j = j + 1
        goto 1
        endif
c increment log-likelihood        
        like = like + dlog(hist(j))
      enddo
      return
      END


      SUBROUTINE RSKEWNORM(x,nx,mu,tau,alph,nmu,ntau,nalph,rn,tnx)
cf2py intent(hide) nmu, ntau, nalph, tnx
cf2py intent(out) x


      INTEGER i, nx, nalph, nmu, ntau, tnx
      DOUBLE PRECISION x(nx), mu(nmu), tau(ntau), alph(nalph)
      DOUBLE PRECISION U1,U2, mu_now, tau_now, alph_now, d_now
      DOUBLE PRECISION rn(tnx)
      LOGICAL vec_mu, vec_tau, vec_alph
      
      vec_mu = (nmu.GT.1)
      vec_tau = (ntau.GT.1)
      vec_alph = (nalph.GT.1)
      
      alph_now = alph(1)
      tau_now = tau(1)
      mu_now = mu(1)
      
      do i=1,nx
         if (vec_mu) then
             mu_now = mu(i)
         end if
         if (vec_alph) then
             alph_now = alph(i)
         end if
         if (vec_tau) then
             tau_now = tau(i)
         end if
         
         U1 = rn(2*i-1)
         U2 = rn(2*i)
         d_now = alph_now / dsqrt(1.0D0 + alph_now * alph_now)
         
         x(i)=(d_now*dabs(U1)+dsqrt(1.0D0-d_now**2)*U2)
     *   /dsqrt(tau_now)+mu_now
                           
      end do

      RETURN
      END
      subroutine uniform_like(x,lower,upper,n,nlower,nupper,like)
        
c Return the uniform likelihood of x.
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(nlower), intent(in) :: lower
cf2py double precision dimension(nupper), intent(in) :: upper 
cf2py integer intent(hide), depend(x) :: n=len(x)
cf2py integer intent(hide), depend(lower) :: nlower=len(lower)
cf2py integer intent(hide), depend(upper) :: nupper=len(upper)
cf2py double precision intent(out) :: like

        IMPLICIT NONE
        
        INTEGER n, nlower, nupper, i
        DOUBLE PRECISION x(n), lower(nlower), upper(nupper)
        DOUBLE PRECISION like, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)
                
        low = lower(1)
        high = upper(1)       
        like = 0.0
        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            like = -infinity
            RETURN
          else
            like = like - dlog(high-low)
          endif
        enddo
      END subroutine uniform_like

      SUBROUTINE exponweib(x,a,c,loc,scale,n,na,nc,nloc,nscale,like)
      
c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nc), intent(in) :: c
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(a) :: na=len(a)
cf2py integer intent(hide),depend(c) :: nc=len(c)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision intent(out) :: like

      DOUBLE PRECISION x(n), z(n), a(na)
      DOUBLE PRECISION c(nc), loc(nloc), scale(nscale)
      INTEGER i, n, na, nc, nloc, nscale
      DOUBLE PRECISION like
      LOGICAL not_scalar_a, not_scalar_c, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
      aa = a(1)
      cc = c(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_c = (nc .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)
     
      like = 0.0
      do i=1,n
        if (not_scalar_a) aa = a(i)
        if (not_scalar_c) cc = c(i)
        if (not_scalar_scale) sigma = scale(i)

! Check c(i) > 0
        if (cc .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        
! Check a(i) > 0
        if (aa .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        
! Check z(i) > 0
        if (z(i) .LE. 0.0) then
          like = -infinity
          RETURN
        endif

        t1 = dexp(-z(i)**cc)
        pdf = aa*cc*(1.0-t1)**(aa-1.0)*t1*z(i)**(cc-1.0)
        like = like + dlog(pdf/sigma)
      enddo
      END SUBROUTINE EXPONWEIB

      SUBROUTINE exponweib_ppf(q,a,c,n,na,nc,ppf)
      
c     Compute the percentile point function for the 
c     Exponentiated Weibull distribution.
c     Accept parameters a,c of length 1 or n.

c CREATED 12/06 DH.

cf2py double precision dimension(n), intent(in) :: q
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nc), intent(in) :: c      
cf2py integer intent(hide),depend(q) :: n=len(q)
cf2py integer intent(hide),depend(a) :: na=len(a)
cf2py integer intent(hide),depend(c) :: nc=len(c)
cf2py double precision dimension(n), intent(out) :: ppf


      IMPLICIT NONE
      INTEGER n,na,nc,i
      DOUBLE PRECISION q(n), a(na), c(nc), ppf(n),ta,tc
      LOGICAL not_scalar_a, not_scalar_c
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
c     Check length of input arrays.
      not_scalar_a = (na .NE. 1)
      not_scalar_c = (nc .NE. 1)
      if ((not_scalar_a) .AND. (na .NE. n)) return
      if ((not_scalar_c) .AND. (nc .NE. n)) return
             
      ta = a(1)
      tc = c(1)

      DO i=1,n
        if (not_scalar_a) ta = a(i)
        if (not_scalar_c) tc = c(i)
        ppf(i) = (-dlog(1.0 - q(i)**(1.0/ta)))**(1.0/tc)
      ENDDO

      END SUBROUTINE exponweib_ppf


      SUBROUTINE constrain(pass, x, a, b, allow_equal, n, na, nb)

c Check that x is in [a, b] if allow_equal, or
c that x is in ]a, b[ if not. 

cf2py integer, intent(out) :: pass
cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nb), intent(in) :: b
cf2py integer intent(hide), depend(x) :: n = len(x)
cf2py integer intent(hide), depend(a) :: na = len(a)
cf2py integer intent(hide), depend(b) :: nb = len(b)
cf2py logical intent(in) :: allow_equal


      IMPLICIT NONE
      INTEGER n, na, nb, i, pass
      DOUBLE PRECISION x(n), a(na), b(nb), ta, tb
      LOGICAL allow_equal, not_scalar_a, not_scalar_b
      pass = 1 

      ta = a(1)
      tb = b(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      if (allow_equal) then
        do i=1,n
          if (not_scalar_a) ta = a(i)
          if (not_scalar_b) tb = b(i) 
          if ((x(i) .LT. ta) .OR. (x(i) .GT. tb)) then
            pass = 0 
            RETURN
          endif
        enddo
      else 
        do i=1,n
          if (not_scalar_a) ta = a(i)
          if (not_scalar_b) tb = b(i) 
          if ((x(i) <= ta) .OR. (x(i) >= tb)) then 
            pass = 0 
            RETURN
          endif
        enddo
      endif
      END


      SUBROUTINE poisson(x,mu,n,nmu,like)

c Poisson log-likelihood function      
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)

      IMPLICIT NONE
      INTEGER n, i, nmu
      INTEGER x(n)
      DOUBLE PRECISION mu(nmu), like
      DOUBLE PRECISION sumx, mut, infinity, sumfact
      DOUBLE PRECISION factln
      PARAMETER (infinity = 1.7976931348623157d308)


      mut = mu(1)
      
c      CALL constrain(x,0,INFINITY,allow_equal=1)
c      CALL constrain(mu,0,INFINITY,allow_equal=0)

      sumx = 0.0
      sumfact = 0.0
      do i=1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif
        
        if (mut .LT. 0.0) then
          like = -infinity
          RETURN
        endif
    
        if (x(i) .LT. 0.0) then
          like = -infinity
          RETURN
        endif
    
        if (.NOT.((x(i) .EQ. 0.0) .AND. (mut .EQ. 0.0))) then
          sumx = sumx + x(i)*dlog(mut) - mut      
          sumfact = sumfact + factln(x(i))
        endif
      enddo
      like = sumx - sumfact
      return
      END

      SUBROUTINE multinomial(x,n,p,nx,nn,np,k,like)

c Multinomial log-likelihood function     
c Updated 12/02/2007 DH. N-D still buggy.
c Fixed 22/11/2007 CF

cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
cf2py integer intent(hide),depend(n) :: nn=shape(n,0)
cf2py integer intent(hide),depend(p) :: np=shape(p,0)
cf2py integer intent(hide),depend(x,p),check(k==shape(p,1)) :: k=shape(x,1)
cf2py intent(out) like      

      DOUBLE PRECISION like, factln, infinity, sump
      DOUBLE PRECISION p(np,k), p_tmp(k)
      INTEGER i,j,n(nn),n_tmp,sumx
      INTEGER x(nx,k)
      PARAMETER (infinity = 1.7976931348623157d308)


      like = 0.0
      n_tmp = n(1)
      do i=1,k
            p_tmp(i) = p(1,i)
      enddo
      do j=1,nx
        if (np .NE. 1) then
              do i=1,k
                    p_tmp(i) = p(j,i)
              enddo
        endif
        if (nn .NE. 1) n_tmp = n(j)
            
!       protect against negative n
        if (n_tmp .LT. 0) then
          like=-infinity
          RETURN
        endif
        
        sump = 0.0
        sumx = 0
        do i=1,k
            
!         protect against negative x or negative p
          if ((x(j,i) .LT. 0) .OR. (p_tmp(i).LT.0.0D0)) then
            like = -infinity
            RETURN
          endif

!         protect against zero p AND nonzero x
          if (p_tmp(i) .EQ. 0.0D0) then
              if (x(j,i) .GT. 0.0D0) then
                  like=-infinity
                  return
              end if
          else
             like = like + x(j,i)*dlog(p_tmp(i))
          end if
        
          like = like - factln(x(j,i))
          
          sump = sump + p_tmp(i)
          sumx = sumx + x(j,i)

        enddo
c This is to account for the kth term that is not passed!
c The kth term does get passed... we can check for consistency.
c But roundoff error ofter triggers a false alarm.

          if (sumx .NE. n_tmp) then
              like=-infinity
              return
          endif
         if ((sump .GT. 1.000001) .OR. (sump .LT. 0.999999)) then
             like=-infinity
             return
         endif
!         xk = n_tmp - sumx
!         pk = 1.0 - sump
!         print *,sump, pk, sumx, n_tmp, xk
!         if ((xk .LT. 0) .OR. (pk .LT. 0)) then
!           like = -infinity
!           RETURN
!         endif
!         like = like + (xk) * dlog(pk) 
!         like = like - factln(xk)
        like = like + factln(n_tmp)
      enddo
      RETURN
      END
      
      SUBROUTINE weibull(x,alpha,beta,n,nalpha,nbeta,like)

c Weibull log-likelihood function      
c UPDATED 1/16/07 AP

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nbeta),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(beta) :: nbeta=len(beta)

      DOUBLE PRECISION x(n),alpha(nalpha),beta(nbeta)
      DOUBLE PRECISION like, alphat, betat
      INTEGER n,nalpha,nbeta,i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alphat = alpha(1)
      betat = beta(1)

      like = 0.0      
      do i=1,n
        if (nalpha .NE. 1) alphat = alpha(i)
        if (nbeta .NE. 1) betat = beta(i)
        if ((alphat .LE. 0.0) .OR. (betat .LE. 0.0)) then
          like=-infinity
          RETURN
        endif
        if (x(i) .LE. 0.0) then
          like=-infinity
          RETURN
        endif
            
c normalizing constant
        like = like + (dlog(alphat) - alphat*dlog(betat))
c kernel of distribution
        like = like + (alphat-1) * dlog(x(i))
        like = like - (x(i)/betat)**alphat
      enddo
      return
      END
      
      
      SUBROUTINE logistic(x, mu, tau, n, nmu, ntau, like)

c Logistic log-likelihood function      

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if (tau_tmp .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        like = like + dlog(tau_tmp) - tau_tmp * (x(i)-mu_tmp)
        like = like - 2.0*dlog(1.0 + dexp(-tau_tmp * (x(i)-mu_tmp)))
      enddo
      return
      END
      
      
      SUBROUTINE normal(x,mu,tau,n,nmu, ntau, like)

c Normal log-likelihood function      

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if (tau_tmp .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        like = like - 0.5 * tau_tmp * (x(i)-mu_tmp)**2
        like = like + 0.5*dlog(0.5*tau_tmp/PI)
      enddo
      return
      END


      SUBROUTINE hnormal(x,tau,n,ntau,like)

c Half-normal log-likelihood function    

c Updated 24/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(tau,n),check(ntau==1 || ntau==n) :: ntau=len(tau)

      IMPLICIT NONE
      INTEGER n,i,ntau
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),tau(ntau),tau_tmp
      LOGICAL not_scalar_tau

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_tau = (ntau .NE. 1)

      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_tau) tau_tmp = tau(i)
        if ((tau_tmp .LE. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        like = like + 0.5 * (dlog(2. * tau_tmp / PI)) 
        like = like - (0.5 * x(i)**2 * tau_tmp)
      enddo
      return
      END


      SUBROUTINE lognormal(x,mu,tau,n,nmu,ntau,like)

c Log-normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if ((tau_tmp .LE. 0.0).OR.(x(i) .LE. 0.0)) then
          like = -infinity
          RETURN
        endif            
        like = like + 0.5 * (dlog(tau_tmp) - dlog(2.0*PI)) 
        like = like - 0.5*tau_tmp*(dlog(x(i))-mu_tmp)**2 - dlog(x(i))
      enddo
      return
      END


      SUBROUTINE arlognormal(x, mu, sigma, rho, beta, n, nmu, like)

C Autocorrelated lognormal loglikelihood.
C David Huard, June 2007

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision intent(in) :: sigma, rho, beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)

      IMPLICIT NONE
      INTEGER n,i,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),sigma, rho, beta
      DOUBLE PRECISION mu_tmp,logx(n),r(n),t1,t2,t3,t4,quad
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      if ((abs(rho) > 1).OR.(sigma <= 0.0)) then 
        like = -infinity
        RETURN
      endif
    
      t1 = n/2. *dlog(2*pi)
      t2 = .5 * (dlog(beta) + 2*n*dlog(sigma) - dlog(1.-rho**2))

      t3 = 0.0
      mu_tmp = mu(1)
      do i=1,n
        if (x(i) <= 0.0) then
          like = -infinity
          RETURN 
        endif
        logx(i) = dlog(x(i))
        t3 = t3+logx(i)
        if (nmu .NE. 1) mu_tmp=mu(i)
        r(i) = logx(i) - mu_tmp
      enddo

      quad = 1.0/beta * (1.-rho**2)*r(1)**2
      do i=1,n-1
        quad = quad + (r(i+1) - rho*r(i))**2
      enddo

      t4 = .5 * quad/sigma**2
      like = -t1-t2-t3-t4
      END SUBROUTINE
    


      SUBROUTINE gev(x,xi,mu,sigma,n,nxi,nmu,nsigma,like)
C
C     COMPUTE THE LIKELIHOOD OF THE GENERALIZED EXTREME VALUE DISTRIBUTION.
C
Cf2py double precision dimension(n), intent(in):: x
Cf2py double precision dimension(nxi), intent(in):: xi
Cf2py double precision dimension(nmu), intent(in):: mu
Cf2py double precision dimension(nsigma), intent(in):: sigma
Cf2py integer intent(hide), depend(x) :: n=len(x)
Cf2py integer intent(hide), depend(xi,n),check(nxi==1||nxi==n) :: nxi=len(xi)
Cf2py integer intent(hide), depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
Cf2py integer intent(hide), depend(sigma,n),check(nsigma==1||nsigma==n) :: nsigma=len(sigma)
Cf2py double precision intent(out):: like

      INTEGER n, nmu, nxi, nsigma, i
      DOUBLE PRECISION x(n), xi(nxi), mu(nmu), sigma(nsigma), like
      DOUBLE PRECISION Z(N), EX(N), PEX(N)
      DOUBLE PRECISION XI_tmp, SIGMA_tmp
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      CALL standardize(x,mu,sigma,n,nmu,nsigma,z)

      xi_tmp = xi(1)
      sigma_tmp = sigma(1)
      LIKE = 0.0
      DO I=1,N
        if (nxi .NE. 1) xi_tmp = xi(i)
        if (nsigma .NE. 1) sigma_tmp = sigma(i)          
        IF (ABS(xi_tmp) .LT. 10.**(-5.)) THEN
          LIKE = LIKE - Z(I) - dexp(-Z(I)) - dlog(sigma_tmp)
        ELSE 
          EX(I) = 1. + xi_tmp*z(i)
          IF (EX(I) .LT. 0.) THEN
            LIKE = -infinity
            RETURN
          ENDIF
          PEX(I) = EX(I)**(-1./xi_tmp)  
          LIKE = LIKE - dlog(sigma_tmp) - PEX(I) 
          LIKE = LIKE - (1./xi_tmp +1.)* dlog(EX(I))
        ENDIF
      ENDDO

      end subroutine gev    


      SUBROUTINE gev_ppf(q,xi,n,nxi,ppf)
C
C     COMPUTE THE Percentile Point function (PPF) OF THE 
C     GENERALIZED EXTREME VALUE DISTRIBUTION.
C
C Created 29/01/2007 DH.
C
Cf2py double precision dimension(n), intent(in):: q
Cf2py double precision dimension(nxi), intent(in):: xi
Cf2py integer intent(hide), depend(q)::n=len(q)
Cf2py integer intent(hide), depend(xi,n),check(nxi==1 || nxi==n) :: nxi=len(xi)
Cf2py double precision dimension(n), intent(out):: ppf

      IMPLICIT NONE
      INTEGER n,nxi,i
      DOUBLE PRECISION q(n), xi(nxi), ppf(n)
      DOUBLE PRECISION xi_tmp

      xi_tmp = xi(1)
      do i=1,n
        if (nxi .NE. 1) xi_tmp= xi(i)
        IF (ABS(xi_tmp) .LT. 10.**(-5.)) THEN
          ppf(i) = -dlog(-dlog(q(i)))
        ELSE
          ppf(i) = 1./xi_tmp * ( (-dlog(q(i)))**(-xi_tmp) -1. )
        ENDIF
      enddo
      return 
      END SUBROUTINE gev_ppf




      SUBROUTINE gamma(x,alpha,beta,n,na,nb,like)

c Gamma log-likelihood function      

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)


      INTEGER i,n,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)
      like = 0.0
      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)
        if ((x(i) .LT. 0.0) .OR. (alpha_tmp .LE. 0.0) .OR. 
     +(beta_tmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if (x(i).EQ.0.0) then

            if (alpha_tmp.EQ.1.0) then
                like = like + beta_tmp
            else if (alpha_tmp.LT.1.0) then
                like = infinity
                RETURN
            else
                like = -infinity
                RETURN
            end if
        else            
            like = like - gammln(alpha_tmp) + alpha_tmp*dlog(beta_tmp)
            like = like + (alpha_tmp - 1.0)*dlog(x(i)) - beta_tmp*x(i)
        end if
      enddo     

      return
      END

      SUBROUTINE igamma(x,alpha,beta,n,na,nb,like)

c Inverse gamma log-likelihood function      

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha,n),check(na==1||na==n) :: na=len(alpha)
cf2py integer intent(hide),depend(beta,n),check(nb==1||nb==n) :: nb=len(beta)

      IMPLICIT NONE
      INTEGER i,n,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      DOUBLE PRECISION alpha_tmp, beta_tmp
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alpha_tmp=alpha(1)
      beta_tmp=beta(1)
      like = 0.0D0
      do i=1,n
        if (na .NE. 1) alpha_tmp=alpha(i)
        if (nb .NE. 1) beta_tmp=beta(i)
        if ((alpha_tmp .LT. 0.0) .OR. (beta_tmp .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        if ((x(i) .LE. 0.0).OR.(alpha_tmp.LE.0.0).OR.
     +       (beta_tmp.LE.0.0)) then
          like = -infinity
          RETURN
        endif
        like = like - gammln(alpha_tmp) - alpha_tmp*dlog(beta_tmp)
        like = like - (alpha_tmp+1.0D0)*dlog(x(i)) - 1.0D0/x(i)/beta_tmp
      enddo

      return
      END


      SUBROUTINE hyperg(x,draws,success,total,n,nd,ns,nt,like)

c Hypergeometric log-likelihood function
c Updated 5/02/07, DH. Changed variable names. 

c Distribution models the probability of drawing x successes in a 
c given number of draws, knowing the population composition (success, failures).
c where failures = total - success

cf2py integer dimension(n),intent(in) :: x
cf2py integer dimension(nd),intent(in) :: draws
cf2py integer dimension(ns),intent(in) :: success
cf2py integer dimension(nt),intent(in) :: total
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(draws,n),check(nd==1||nd==n) :: nd=len(draws)
cf2py integer intent(hide),depend(success,n),check(ns==1||ns==n) :: ns=len(success)
cf2py integer intent(hide),depend(total,n),check(nt==1||nt==n) :: nt=len(total)
cf2py double precision intent(out) :: like

      IMPLICIT NONE
      INTEGER i,n,nd,ns,nt
      INTEGER x(n),draws(nd), success(ns),total(nt)
      INTEGER draws_tmp, s_tmp, t_tmp
      DOUBLE PRECISION like, combinationln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

c      CALL constrain(d,x,total,allow_equal=1)
c      CALL constrain(red,x,total,allow_equal=1)
c      CALL constrain(x, 0, d, allow_equal=1)

      draws_tmp = draws(1)
      s_tmp = success(1)
      t_tmp = total(1)
      
!       print *,draws,success,total

      like = 0.0
      do i=1,n
c Combinations of x red balls
        if (nd .NE. 1) draws_tmp = draws(i)
        if (ns .NE. 1) s_tmp = success(i)
        if (nt .NE. 1) t_tmp = total(i)
        if ((draws_tmp .LE. 0) .OR. (s_tmp .LT. 0)) then
          like = -infinity
          RETURN
        endif
        if (t_tmp .LE. 0) then
          like = -infinity
          RETURN
        endif
        if (x(i) .LT. MAX(0, draws_tmp - t_tmp + s_tmp)) then
          like = -infinity
          RETURN
        else if (x(i) .GT. MIN(draws_tmp, s_tmp)) then
          like = -infinity
          RETURN
        endif
c        like = like + combinationln(t_tmp-s_tmp, x(i))
c        like = like + combinationln(s_tmp,draws_tmp-x(i))
        like = like + combinationln(t_tmp-s_tmp, draws_tmp-x(i))
        like = like + combinationln(s_tmp, x(i))
        like = like - combinationln(t_tmp, draws_tmp)
      enddo
      return
      END


      SUBROUTINE geometric(x,p,n,np,like)

c Geometric log-likelihood

c Created 29/01/2007 DH.

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(p,n),check(np==1 || np==n) :: np=len(p)
cf2py double precision intent(out) :: like      

      IMPLICIT NONE
      INTEGER n,np,i
      INTEGER x(n)
      DOUBLE PRECISION p(np), p_tmp
      DOUBLE PRECISION like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      

      p_tmp = p(1)
      like = 0.0
      do i=1, n
        if (np .NE. 1) p_tmp = p(i)
        if ((p_tmp .LE. 0.0) .OR. (p_tmp .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
        if (x(i) .LT. 1) then
          like = -infinity
          RETURN
        endif            
        like = like + dlog(p_tmp) + (x(i)-1)* dlog(1.0D0-p_tmp)
      enddo
      return
      END SUBROUTINE geometric


      SUBROUTINE dirichlet(x,theta,k,like)

! Updated 3/7/08... AP couldn't figure out how to 
! make x length k-1, so using single x and theta values
! for now.

c Dirichlet multivariate log-likelihood function      

cf2py intent(out) like
cf2py intent(hide) k

      IMPLICIT NONE
      INTEGER j,k
      DOUBLE PRECISION like,sumt,sumx
      DOUBLE PRECISION x(k-1),theta(k)
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
      like = 0.0D0

        sumt = 0.0D0
        sumx = 0.0D0
        do j=1,k-1
c kernel of distribution      

          if ((x(j) .LE. 0.0).OR.(theta(j).LE.0.0)) then
            like = -infinity
            RETURN
          endif
          like = like + (theta(j)-1.0D0)*dlog(x(j))

c normalizing constant        
          like = like - gammln(theta(j))
          sumt = sumt + theta(j)
          sumx = sumx + x(j)
        enddo
        
c implicit k'th term
        if (sumx .GT. 1.0) then
          like = -infinity
          RETURN
        endif          
        
       if ((theta(k).LE.0.0).OR.(sumx.GE.1.0)) then
          like = -infinity
          RETURN
       end if
           
       like = like + (theta(k)-1.0D0)*dlog(1.0D0-sumx)
       like = like - gammln(theta(k))
       sumt = sumt + theta(k)
       
        like = like + gammln(sumt)

      return
      END SUBROUTINE dirichlet


!       SUBROUTINE dirichlet(x,theta,k,nx,nt,like)
! 
! c Dirichlet multivariate log-likelihood function      
! 
! c Updated 22/01/2007 DH. 
! 
! cf2py double precision dimension(nx,k),intent(in) :: x
! cf2py double precision dimension(nt,k),intent(in) :: theta
! cf2py double precision intent(out) :: like
! cf2py integer intent(hide), depend(x,theta),check(k==shape(theta,1)||(k==shape(theta,0) && shape(theta,1)==1)) :: k=shape(x,1)
! cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
! cf2py integer intent(hide),depend(theta,nx),check(nt==1 || nt==nx) :: nt=shape(theta,0)
! 
!       IMPLICIT NONE
!       INTEGER i,j,nx,nt,k
!       DOUBLE PRECISION like,sumt
!       DOUBLE PRECISION x(nx,k),theta(nt,k)
!       DOUBLE PRECISION theta_tmp(k)
!       DOUBLE PRECISION gammln
!       DOUBLE PRECISION infinity
!       PARAMETER (infinity = 1.7976931348623157d308)
!       
! 
!       like = 0.0
!       do j=1,k
!         theta_tmp(j) = theta(1,j)
!       enddo
! 
!       do i=1,nx
!         sumt = 0.0
!         do j=1,k
!           if (nt .NE. 1) theta_tmp(j) = theta(i,j)      
! c kernel of distribution      
!           like = like + (theta_tmp(j)-1.0)*dlog(x(i,j))  
!           if ((x(i,j) .LE. 0.0) .OR. (theta_tmp(j) .LE. 0.0)) then
!             like = -infinity
!             RETURN
!           endif          
! c normalizing constant        
!           like = like - gammln(theta_tmp(j))
!           sumt = sumt + theta_tmp(j)
!         enddo
!         like = like + gammln(sumt)
!       enddo
!       return
!       END SUBROUTINE dirichlet


      SUBROUTINE cauchy(x,alpha,beta,nx, na, nb,like)

c Cauchy log-likelihood function      

c UPDATED 17/01/2007 DH. 

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)

      IMPLICIT NONE
      INTEGER nx,na,nb,i
      DOUBLE PRECISION x(nx),alpha(na),beta(nb)
      DOUBLE PRECISION like, atmp, btmp, PI
      LOGICAL not_scalar_alpha, not_scalar_beta
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)
      like = -nx*dlog(PI)
      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)

        if (btmp .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        
        like = like - dlog(btmp)
        like = like -  dlog( 1. + ((x(i)-atmp) / btmp) ** 2 )
      enddo
      return
      END

      SUBROUTINE negbin(x,r,p,n,nr,np,like)

c Negative binomial log-likelihood function     

c Updated 24/01/2007. 

cf2py integer dimension(n),intent(in) :: x
cf2py integer dimension(nr),intent(in) :: r
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(r,n),check(nr==1 || nr==n) :: nr=len(r)
cf2py integer intent(hide),depend(p,n),check(np==1 || np==n) :: np=len(p)
cf2py double precision intent(out) :: like      

      IMPLICIT NONE
      INTEGER n,nr,np,i
      DOUBLE PRECISION like
      DOUBLE PRECISION p(np),p_tmp
      INTEGER x(n),r(nr),r_tmp
      LOGICAL not_scalar_r, not_scalar_p
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_r = (nr .NE. 1)
      not_scalar_p = (np .NE. 1)

      r_tmp = r(1)
      p_tmp = p(1)
      like = 0.0
      do i=1,n
        if (not_scalar_r) r_tmp = r(i)
        if (not_scalar_p) p_tmp = p(i)
        
        if ((r_tmp .LE. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        
        if ((p_tmp .LE. 0.0) .OR. (p_tmp .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
            
        like = like + r_tmp*dlog(p_tmp) + x(i)*dlog(1.-p_tmp)
        like = like+factln(x(i)+r_tmp-1)-factln(x(i))-factln(r_tmp-1) 
      enddo
      return
      END


      SUBROUTINE negbin2(x,mu,a,n,nmu,na,like)

c Negative binomial log-likelihood function 
c (alternative parameterization)    
c Updated 1/4/08 CF

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: a
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,x),check(nmu==1 || nmu==len(x)) :: nmu=len(mu)
cf2py integer intent(hide),depend(a,x),check(na==1 || na==len(x)) :: na=len(a)
cf2py double precision intent(out) :: like      

      IMPLICIT NONE
      INTEGER n,i,nmu,na
      DOUBLE PRECISION like
      DOUBLE PRECISION a(na),mu(nmu), a_tmp, mu_tmp
      INTEGER x(n)
      LOGICAL not_scalar_a, not_scalar_mu
      DOUBLE PRECISION gammln, factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_a = (na .NE. 1)

      mu_tmp = mu(1)
      a_tmp = a(1)
      like = 0.0
      do i=1,n
        if ((x(i) .LT. 0) .OR. (mu_tmp .LE. 0.0) .OR. 
     +(a_tmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_a) a_tmp=a(i)
        like=like + gammln(x(i)+a_tmp) - factln(x(i)) - gammln(a_tmp)
        like=like + x(i)*(dlog(mu_tmp/a_tmp) - dlog(1.0+mu_tmp/a_tmp))
        like=like - a_tmp*dlog(1.0 + mu_tmp/a_tmp)
      enddo
      return
      END




      SUBROUTINE binomial(x,n,p,nx,nn,np,like)

c Binomial log-likelihood function     

c  Updated 17/01/2007. DH. 

cf2py integer dimension(nx),intent(in) :: x
cf2py integer dimension(nn),intent(in) :: n
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py integer intent(hide),depend(p),check(np==1 || np==len(x)) :: np=len(p)
cf2py double precision intent(out) :: like      
      IMPLICIT NONE
      INTEGER nx,nn,np,i
      DOUBLE PRECISION like, p(np)
      INTEGER x(nx),n(nn)
      LOGICAL not_scalar_n,not_scalar_p
      INTEGER ntmp
      DOUBLE PRECISION ptmp
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_n = (nn .NE. 1)
      not_scalar_p = (np .NE. 1) 

      ntmp = n(1)
      ptmp = p(1)

      like = 0.0
      do i=1,nx
        if (not_scalar_n) ntmp = n(i)
        if (not_scalar_p) ptmp = p(i)
        
        if ((x(i) .LT. 0) .OR. (ntmp .LT. 0) .OR. (x(i) .GT. ntmp)) then
          like = -infinity
          RETURN
        endif
        
        if ((ptmp .LE. 0.0D0) .OR. (ptmp .GE. 1.0D0)) then
!         if p = 0, number of successes must be 0
          if (ptmp .EQ. 0.0D0) then
            if (x(i) .GT. 0.0D0) then
                like = -infinity
                RETURN
!                 else like = like + 0
            end if
          else if (ptmp .EQ. 1.0D0) then
!           if p = 1, number of successes must be n
            if (x(i) .LT. ntmp) then
                like = -infinity
                RETURN
!                 else like = like + 0
            end if
          else
            like = -infinity
            RETURN
          endif
        else
            like = like + x(i)*dlog(ptmp) + (ntmp-x(i))*dlog(1.-ptmp)
            like = like + factln(ntmp)-factln(x(i))-factln(ntmp-x(i)) 
        end if
      enddo
      return
      END


      SUBROUTINE bernoulli(x,p,nx,np,like)
         
c Modified on Jan 16 2007 by D. Huard to allow scalar p.

cf2py logical dimension(nx),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(p),check(len(p)==1 || len(p)==len(x)):: np=len(p) 
cf2py double precision intent(out) :: like      
      IMPLICIT NONE

      INTEGER np,nx,i
      DOUBLE PRECISION p(np), ptmp, like
      LOGICAL x(nx)
      LOGICAL not_scalar_p
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

C     Check parameter size
      not_scalar_p = (np .NE. 1)

      like = 0.0
      ptmp = p(1)
      do i=1,nx
        if (not_scalar_p) ptmp = p(i)
        if (ptmp .LT. 0.0) then
          like = -infinity
          RETURN
        endif
        
        if (x(i)) then 
          like = like + dlog(ptmp)
        else 
          like = like + dlog(1.0D0 - ptmp)
        endif
          
      enddo
      return
      END


      SUBROUTINE beta_like(x,alpha,beta,nx,na,nb,like)

c Beta log-likelihood function      
c Modified by D. Huard on Jan 17 2007 to accept scalar parameters.
c Renamed to use alpha and beta arguments for compatibility with 
c random.beta.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision intent(out) :: like
      IMPLICIT NONE
      INTEGER i,nx,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(nx),alpha(na),beta(nb), atmp, btmp
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity, e, zero, one
      PARAMETER (infinity = 1.7976931348623157d308)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/
      

      atmp = alpha(1)
      btmp = beta(1)
      like = 0.0
      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        if ((atmp .LE. 0.0) .OR. (btmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if ((x(i) .LE. 0.0) .OR. (x(i) .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
        like =like + (gammln(atmp+btmp) - gammln(atmp) - gammln(btmp))
        like =like + (atmp-one)*dlog(x(i)) + (btmp-one)*dlog(one-x(i))
      enddo   

      return
      END
      

      SUBROUTINE mvhyperg(x,color,k,like)

c Multivariate hypergeometric log-likelihood function
c Using the analogy of an urn filled with balls of different colors, 
c the mv hypergeometric distribution describes the probability of 
c drawing x(i) balls of a given color. 
c
c x : (array) Number of draws for each color.
c color : (array) Number of balls of each color.

c Total number of draws = sum(x)
c Total number of balls in the urn = sum(color)

cf2py integer dimension(k),intent(in) :: x,color
cf2py integer intent(hide),depend(x) :: k=len(x)
cf2py double precision intent(out) :: like

      INTEGER x(k),color(k)
      INTEGER d,total,i,k
      DOUBLE PRECISION like
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      total = 0
      d = 0
      like = 0.0
      do i=1,k
c Combinations of x balls of color i     
        like = like + factln(color(i))-factln(x(i))
     +-factln(color(i)-x(i))
        if ((color(i) .LT. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        d = d + x(i)
        total = total + color(i)
      enddo
      if (total .LE. 0.0) then
        like = -infinity
        RETURN
      endif
c Combinations of d draws from total    
      like = like - (factln(total)-factln(d)-factln(total-d))
      return
      END


      SUBROUTINE dirmultinom(x,theta,k,like)

c Dirichlet-multinomial log-likelihood function      

cf2py integer dimension(k),intent(in) :: x
cf2py double precision dimension(k),intent(in) :: theta      
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)

      INTEGER i,k,sumx
      INTEGER x(k)
      DOUBLE PRECISION like,sumt
      DOUBLE PRECISION theta(k)
      DOUBLE PRECISION factln, gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      like = 0.0
      sumt = 0.0
      sumx = 0
      do 222 i=1,k
c kernel of distribution
        like = like + dlog(x(i) + theta(i)) - dlog(theta(i)) 
        sumt = sumt + theta(i)
        sumx = sumx + x(i)
        if ((theta(i) .LT. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
  222 continue
c normalizing constant 

      if ((sumx .LE. 0.0) .OR. (sumt .LE. 0.0)) then
        like = -infinity
        RETURN
      endif
      
      like = like + factln(sumx)
      like = like + gammln(sumt)
      like = like - gammln(sumx + sumt)

      return
      END


      SUBROUTINE wishart(X,k,n,sigma,like)

c Wishart log-likelihood function      

cf2py double precision dimension(k,k),intent(in) :: X,sigma
cf2py double precision intent(in) :: n
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)

      INTEGER i,k
      DOUBLE PRECISION X(k,k),sigma(k,k),bx(k,k)
      DOUBLE PRECISION dx,n,db,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      

c determinants
      call dtrm(X,k,dx)
      call dtrm(sigma,k,db)
c trace of sigma*X     
      call matmult(sigma,X,bx,k,k,k,k)
      call trace(bx,k,tbx)
      
      if ((dx .LE. 0.0) .OR. (db .LE. 0.0)) then
        like = -infinity
        RETURN
      endif
      
      if (k .GT. n) then
        like = -infinity
        RETURN
      endif
      
      like = (n - k - 1)/2.0 * dlog(dx)
      like = like + (n/2.0)*dlog(db)
      like = like - 0.5*tbx
      like = like - (n*k/2.0)*dlog(2.0d0)

      do i=1,k
        a = (n - i + 1)/2.0
        call gamfun(a, g)
        like = like - dlog(g)
      enddo

      return
      END



      SUBROUTINE trace(mat,k,tr)

c matrix trace (sum of diagonal elements)

      INTEGER k,i
      DOUBLE PRECISION mat(k,k),tr

      tr = 0.0
      do i=1,k
        tr = tr + mat(k,k)
      enddo
      return
      END


      SUBROUTINE gamfun(xx,gx)

c Return the logarithm of the gamma function
c Corresponds to scipy.special.gammaln

cf2py double precision intent(in) :: xx
cf2py double precision intent(out) :: gx

      INTEGER i
      DOUBLE PRECISION x,xx,ser,tmp,gx
      DIMENSION coeff(6)
      DATA coeff/76.18009173,-86.50532033,24.01409822,
     +-1.231739516,0.00120858003,-0.00000536382/

      x = xx
      tmp = x + 5.5
      tmp = tmp - (x+0.5) * dlog(tmp)
      ser = 1.000000000190015
      do i=1,6
        x = x+1
        ser = ser + coeff(i)/x
      enddo
      gx = -tmp + dlog(2.50662827465*ser/xx)
      return
      END
      
      
      double precision function gammds (y,p,ifault)
      
cf2py double precision intent(in) :: y,p,ifault
cf2py double precision intent(out) :: gammds
      
c
c        Algorithm AS 147  Appl. Statist. (1980) Vol. 29, No. 1
c
c        Computes the incomplete gamma integral for positive
c        parameters y,p using an infinite series
c
c        Auxiliary function required: ALNGAM = CACM algorithm 291
c
c	 AS239 should be considered as an alternative to AS147
c
      implicit double precision (a-h,o-z)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/, uflo/1.0d-37/
c
c        Checks admissibility of arguments and value of f
c
      ifault = 1
      gammds = zero
      if(y.le.zero .or. p.le.zero) return
      ifault = 2
c
c        alngam is natural log of gamma function
c
      arg = p*log(y)-gammln(p+one)-y
      if(arg.lt.log(uflo)) return
      f = exp(arg)
      if(f.eq.zero) return
      ifault = 0
c
c          Series begins
c
      c = one
      gammds = one
      a = p
    1 a = a+one
      c = c*y/a
      gammds = gammds+c
      if (c/gammds.gt.e) goto 1
      gammds = gammds*f
      return
      end


      SUBROUTINE trans(mat,tmat,m,n)

c matrix transposition      

cf2py double precision dimension(m,n),intent(in) :: mat
cf2py double precision dimension(n,m),intent(out) :: tmat
cf2py integer intent(hide),depend(mat) :: m=len(mat)
cf2py integer intent(hide),depend(mat) :: n=shape(mat,1)

      INTEGER i,j,m,n
      DOUBLE PRECISION mat(m,n),tmat(n,m)

      do 88 i=1,m
        do 99 j=1,n
          tmat(j,i) = mat(i,j)
 99     continue
 88   continue

      return
      END


      SUBROUTINE matmult(mat1, mat2, prod, m, n, p, q)

c matrix multiplication

cf2py double precision dimension(m,q),intent(out) :: prod
cf2py double precision dimension(m,n),intent(in) :: mat1
cf2py double precision dimension(p,q),intent(in) :: mat2
cf2py integer intent(hide),depend(mat1) :: m=len(mat1),n=shape(mat1,1)
cf2py integer intent(hide),depend(mat2) :: p=len(mat2),q=shape(mat2,1)


      INTEGER i,j,k,m,n,p,q
      DOUBLE PRECISION mat1(m,n), mat2(p,q), prod(m,q)
      DOUBLE PRECISION sum

      if (n.eq.p) then
        do 30 i = 1,m
          do 20 j = 1,q
            sum = 0.0
            do 10 k = 1,n
              sum = sum + mat1(i,k) * mat2(k,j)
10          continue
            prod(i,j) = sum
20        continue
30      continue
      else
        write (*,*) 'Matrix dimensions do not match'
      end if
      return
      END


c Updated 10/24/2001.
c
ccccccccccccccccccccccccc     Program 4.2     cccccccccccccccccccccccccc
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c Please Note:                                                         c
c                                                                      c
c (1) This computer program is part of the book, "An Introduction to   c
c     Computational Physics," written by Tao Pang and published and    c
c     copyrighted by Cambridge University Press in 1997.               c
c                                                                      c
c (2) No warranties, express or implied, are made for this program.    c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      SUBROUTINE DTRM(A,N,D)
C
C Subroutine for evaluating the determinant of a matrix using 
C the partial-pivoting Gaussian elimination scheme.
C

cf2py double precision dimension(N,N),intent(in) :: A
cf2py double precision intent(out) :: D
cf2py integer intent(hide),depend(A) :: N=len(A)      

      DOUBLE PRECISION A(N,N), D
      INTEGER INDX(N)

      CALL ELGS(A,N,INDX)
C
      D    = 1.0
      DO     100 I = 1, N
         D = D*A(INDX(I),I)
  100 CONTINUE
C
      MSGN = 1
      DO     200 I = 1, N
        DO   150 WHILE (I.NE.INDX(I))
          MSGN = -MSGN
          J = INDX(I)
          INDX(I) = INDX(J)
          INDX(J) = J
  150   END DO
  200 CONTINUE
      D = MSGN*D

      RETURN
      END


      SUBROUTINE ELGS(A,N,INDX)

C Subroutine to perform the partial-pivoting Gaussian elimination.
C A(N,N) is the original matrix in the input and transformed
C matrix plus the pivoting element ratios below the diagonal in
C the output.  INDX(N) records the pivoting order.


      DOUBLE PRECISION A(N,N),C(N),C1
      INTEGER INDX(N)

C Initialize the index

      DO     50    I = 1, N
        INDX(I) = I
   50 CONTINUE

C Find the rescaling factors, one from each row

        DO     100   I = 1, N
          C1= 0.0
          DO    90   J = 1, N
            C1 = MAX(C1,ABS(A(I,J)))
   90     CONTINUE
          C(I) = C1
  100   CONTINUE

C Search the pivoting (largest) element from each column

      DO     200   J = 1, N-1
        PI1 = 0.0
        DO   150   I = J, N
          PI = ABS(A(INDX(I),J))/C(INDX(I))
          IF (PI.GT.PI1) THEN
            PI1 = PI
            K   = I
          ELSE
          ENDIF
  150   CONTINUE

C Interchange the rows via INDX(N) to record pivoting order

        ITMP    = INDX(J)
        INDX(J) = INDX(K)
        INDX(K) = ITMP
        DO   170   I = J+1, N
          PJ  = A(INDX(I),J)/A(INDX(J),J)

C Record pivoting ratios below the diagonal

          A(INDX(I),J) = PJ

C Modify other elements accordingly

          DO 160   K = J+1, N
            A(INDX(I),K) = A(INDX(I),K)-PJ*A(INDX(J),K)
  160     CONTINUE
  170   CONTINUE
  200 CONTINUE

      RETURN
      END


      FUNCTION bico(n,k) 
C USES factln Returns the binomial coefficient as a 
C floating point number.
      INTEGER k,n
      DOUBLE PRECISION bico
      DOUBLE PRECISION factln
C The nearest-integer function cleans up roundoff error
C for smaller values of n and k. 
      bico=nint(dexp(factln(n)-factln(k)-factln(n-k))) 
      return 
      END

      subroutine chol(n,a,c)
c...perform a Cholesky decomposition of matrix a, returned as c
      implicit double precision (a-h,o-z)
      double precision c(n,n),a(n,n)

cf2py double precision dimension(n,n),intent(in) :: a
cf2py double precision dimension(n,n),intent(out) :: c
cf2py integer intent(in),depend(a) :: n=len(a)

      c(1,1) = sqrt(a(1,1))
      do i=2,n
         c(i,1) = a(i,1) / c(1,1)
      enddo
      do j=2,n
         do i=j,n
            s = a(i,j)
            do k=1,j-1
               s = s - c(i,k) * c(j,k)
            enddo
            if(i .eq. j) then
               c(j,j) = sqrt(s)
            else
               c(i,j) = s / c(j,j)
               c(j,i) = 0.d0
            endif
         enddo
      enddo
      return
      end

      SUBROUTINE hermpoly( n, x, cx )

C*******************************************************************************
C
CC HERMPOLY evaluates the Hermite polynomials at X.
C
C  Differential equation:
C
C    Y'' - 2 X Y' + 2 N Y = 0
C
C  First terms:
C
C      1
C      2 X
C      4 X**2     -  2
C      8 X**3     - 12 X
C     16 X**4     - 48 X**2     + 12
C     32 X**5    - 160 X**3    + 120 X
C     64 X**6    - 480 X**4    + 720 X**2    - 120
C    128 X**7   - 1344 X**5   + 3360 X**3   - 1680 X
C    256 X**8   - 3584 X**6  + 13440 X**4  - 13440 X**2   + 1680
C    512 X**9   - 9216 X**7  + 48384 X**5  - 80640 X**3  + 30240 X
C   1024 X**10 - 23040 X**8 + 161280 X**6 - 403200 X**4 + 302400 X**2 - 30240
C
C  Recursion:
C
C    H(0,X) = 1,
C    H(1,X) = 2*X,
C    H(N,X) = 2*X * H(N-1,X) - 2*(N-1) * H(N-2,X)
C
C  Norm:
C
C    Integral ( -Infinity < X < Infinity ) exp ( - X**2 ) * H(N,X)**2 dX
C    = sqrt ( PI ) * 2**N * N!
C
C    H(N,X) = (-1)**N * exp ( X**2 ) * dn/dXn ( exp(-X**2 ) )
C
C  Modified:
C
C    01 October 2002
C
C  Author:
C
C    John Burkardt
C
C  Reference:
C
C    Milton Abramowitz and Irene Stegun,
C    Handbook of Mathematical Functions,
C    US Department of Commerce, 1964.
C
C    Larry Andrews,
C    Special Functions of Mathematics for Engineers,
C    Second Edition, 
C    Oxford University Press, 1998.
C
C  Parameters:
C
C    Input, integer N, the highest order polynomial to compute.
C    Note that polynomials 0 through N will be computed.
C
C    Input, double precision ( kind = 8 ) X, the point at which the polynomials are 
C    to be evaluated.
C
C    Output, double precision ( kind = 8 ) CX(0:N), the values of the first N+1 Hermite
C    polynomials at the point X.
C

cf2py double precision intent(in) :: x
cf2py integer intent(in) :: n
cf2py double precision dimension(n+1),intent(out) :: cx

      integer n,i
      double precision cx(n+1)
      double precision x

      if ( n < 0 ) then
        return
      end if

      cx(1) = 1.0

      if ( n == 0 ) then
        return
      end if

      cx(2) = 2.0 * x

      do i = 3, n+1
        cx(i) = 2.0 * x * cx(i-1) - 2.0 * real(i - 1) * cx(i-2)
      end do

      return
      end

        double precision function uniform()
c
c    Generate uniformly distributed random numbers using the 32-bit
c    generator from figure 3 of:
c    L`Ecuyer, P. Efficient and portable combined random number
c    generators, C.A.C.M., vol. 31, 742-749 & 774-?, June 1988.
c
c    The cycle length is claimed to be 2.30584E+18
c
c    Seeds can be set by calling the routine set_uniform
c
c    It is assumed that the Fortran compiler supports long variable
c    names, and integer*4.
c
        integer*4 z, k, s1, s2
        common /unif_seeds/ s1, s2
        save /unif_seeds/
c
        k = s1 / 53668
        s1 = 40014 * (s1 - k * 53668) - k * 12211
        if (s1 .lt. 0) s1 = s1 + 2147483563
c
        k = s2 / 52774
        s2 = 40692 * (s2 - k * 52774) - k * 3791
        if (s2 .lt. 0) s2 = s2 + 2147483399
c
        if (z .lt. 1) z = z + 2147483562
c
        uniform = z / 2147483563.
        return
        end


        subroutine set_uniform(seed1, seed2)
c
c    Set seeds for the uniform random number generator.
c
        integer*4 s1, s2, seed1, seed2
        common /unif_seeds/ s1, s2
        save /unif_seeds/

        s1 = seed1
        s2 = seed2
        return
      END


      SUBROUTINE categorical(x,n,hist,k,mn,step,logp)

cf2py intent(out) logp
cf2py intent(hide) n,k

      DOUBLE PRECISION hist(k),logp,x(n),mn,step,nrm
      INTEGER n,k,i,j
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      LOGICAL match
      
      logp = 0.0D0
      nrm = 0.0D0
      do i=1,k
          nrm = nrm + hist(k)
      end do
      if (dabs(nrm-0.0D0).GT.1.0D-7) then
          logp = -infinity
          return
      end if
      
      do i=1,n
          match = .FALSE.
          
          j = int(x(i)-mn/step)+1
          logp = logp + dlog(hist(j))

      end do
      
      return
      END
              

      SUBROUTINE rcat(hist,mn,step,n,s,k,rands)

c Returns n samples from categorical random variable (histogram)

cf2py double precision dimension(k),intent(in) :: hist
cf2py double precision intent(in) :: mn,step
cf2py integer intent(in) :: n
cf2py double precision dimension(n),intent(out) :: s
cf2py integer intent(hide),depend(hist) :: k=len(hist)

      DOUBLE PRECISION hist(k),s(n),mn,step,sump,u
      DOUBLE PRECISION rands(n)
      INTEGER n,k,i,j
      
!       print *,mn,step,k

c repeat for n samples
      do i=1,n
c initialize sum      
        sump = 0.0
c random draw
!         call random_number(u)
        u = rands(i)
        j = 0
        
c find index to value        
    1   if (u.gt.sump) then
          sump = sump + hist(j+1)
          j = j + 1
        goto 1
        endif
c assign value to array        
        s(i) = mn + step*(j-1)
      enddo
      return
      END



      subroutine logit(theta,n,ltheta)
c Maps (0,1) -> R.
cf2py intent(hide) n
cf2py intent(out) ltheta
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      INTEGER n, i
      do i=1,n
          if (theta(i).LE.0) then
              ltheta(i) = -inf
          else if (theta(i).GE.1) then
              ltheta(i) = inf
          else
              ltheta(i) = dlog(theta(i) / (1.0D0-theta(i)))
          endif
      end do
      RETURN
      END
c 
      

      subroutine invlogit(ltheta,n,theta)
c Maps R -> (0,1).
cf2py intent(hide) n
cf2py intent(out) theta
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      INTEGER n, i
      do i=1,n
          theta(i) = 1.0D0 / (1.0D0 + dexp(-ltheta(i)))
      end do
      RETURN
      END      

c

      subroutine stukel_logit(theta,n,ltheta,a1,a2,na1,na2)
!
! Reference: Therese A. Stukel, 'Generalized Logistic Models',
! JASA vol 83 no 402, pp.426-431 (June 1988)
!
cf2py intent(hide) n, na1, na2
cf2py intent(out) ltheta
cf2py intent(copy) theta
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION a1(na1), a2(na2), a1t, a2t
      LOGICAL a1_isscalar, a2_isscalar      
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      INTEGER n, i, na1, na2

      a1t = a1(1)
      a2t = a2(1)

      CALL logit(theta,n,ltheta)

      a1_isscalar = (na1.LT.n)
      a2_isscalar = (na2.LT.n)      

      do i=1,n
          
          if (ltheta(i).GT.0.0D0) then
              if (.NOT.a1_isscalar) then
                  a1t = a1(i)
              end if
              if (a1t.GT.0.0D0) then
                  ltheta(i)=dlog(ltheta(i)*a1t+1.0D0)/a1t
              else if (a1t.LT.0.0D0) then
                  ltheta(i) = (1.0D0-dexp(-ltheta(i)*a1t))/a1t
              end if
              
          else if (ltheta(i).LT.0.0D0) then
              if (.NOT.a2_isscalar) then
                  a2t = a2(i)
              end if
              if (a2t.GT.0.0D0) then
                  ltheta(i)=-dlog(-ltheta(i)*a2t+1.0D0)/a2t
              else if (a2t.LT.0.0D0) then
                  ltheta(i)=-(1.0D0-dexp(ltheta(i)*a2t))/a2t
              end if
              
          else
              ltheta(i) = 0.0D0
          end if

      end do
      
      RETURN
      END

c

      subroutine stukel_invlogit(ltheta,n,theta,a1,a2,na1,na2)
!
! Reference: Therese A. Stukel, 'Generalized Logistic Models',
! JASA vol 83 no 402, pp.426-431 (June 1988)
!
cf2py intent(hide) n, na1, na2
cf2py intent(out) theta
cf2py intent(copy) ltheta
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION a1(na1), a2(na2), a1t, a2t
      LOGICAL a1_isscalar, a2_isscalar
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      INTEGER n, i, na1, na2

      a1t = a1(1)
      a2t = a2(1)
      
      a1_isscalar = (na1.LT.n)
      a2_isscalar = (na2.LT.n)      
      
      do i=1,n
          if (ltheta(i).GT.0.0D0) then
              if (.NOT.a1_isscalar) then
                  a1t = a1(i)
              end if
              if (a1t.GT.0.0D0) then
                  ltheta(i) = (dexp(a1t*ltheta(i))-1.0D0)/a1t
              else if (a1t.LT.0.0D0) then
                  ltheta(i) = -dlog(1.0D0-a1t*ltheta(i))/a1t
              end if
              
          else if (ltheta(i).LT.0.0D0) then
              if (.NOT.a2_isscalar) then
                  a2t = a2(i)
              end if
              if (a2t.GT.0.0D0) then
                  ltheta(i) = -(dexp(-a2t*ltheta(i))-1.0D0)/a2t
              else if (a2t.LT.0.0D0) then
                  ltheta(i) = dlog(1.0D0+a2t*ltheta(i))/a2t
              end if
              
          else
              ltheta(i) = 0.5D0
          end if

      end do

      
      CALL invlogit(ltheta,n,theta)
      
      RETURN
      END      


c

!       SUBROUTINE RNORM(U1, U2, rands)
! C
! C     ALGORITHM AS 53.1  APPL. STATIST. (1972) VOL.21, NO.3
! C
! C     Sets U1 and U2 to two independent standardized random normal
! C     deviates.   This is a Fortran version of the method given in
! C     Knuth(1969).
! C
! C     Function RAND must give a result randomly and rectangularly
! C     distributed between the limits 0 and 1 exclusive.
! C     Note- this seems to be faster than Leva's algorithm from
! C     ACM Trans Math Soft, Dec. 1992 - AP
! C
!       DOUBLE PRECISION U1, U2
! C
! C     Local variables
! C
!       DOUBLE PRECISION X, Y, S, ONE, TWO
!       DOUBLE PRECISION rands(2)
!       DATA ONE /1.0/, TWO /2.0/
! C
!     1 X = rands(1)
!       Y = rands(2)
!       X = TWO * X - ONE
!       Y = TWO * Y - ONE
!       S = X * X + Y * Y
!       IF (S .GT. ONE) GO TO 1
!       S = SQRT(- TWO * dlog(S) / S)
!       U1 = X * S
!       U2 = Y * S
!       RETURN
!       END

! wshrt is not used by anything
!       SUBROUTINE WSHRT(D, N, NP, NNP, SB, SA, rnorms)
! C
! C     ALGORITHM AS 53  APPL. STATIST. (1972) VOL.21, NO.3
! C
! C     Wishart variate generator.  On output, SA is an upper-triangular
! C     matrix of size NP * NP (written in linear form, column ordered)
! C     whose elements have a Wishart(N, SIGMA) distribution.
! C
! C     D is an upper-triangular array such that SIGMA = D'D (see AS 6)
! C
! C     Auxiliary function required: a random no. generator called RAND.
! C     The Wichmann & Hill generator is included here.   It should be
! C     initialized in the calling program.
! 
! 
! cf2py double precision dimension(NNP),intent(in) :: D
! cf2py double precision dimension(NNP),intent(out) :: SA
! cf2py double precision dimension(NNP),intent(hide) :: SB
! cf2py integer intent(hide),depend(D) :: NNP=len(D)
! cf2py integer intent(in) :: NP
! cf2py integer intent(in) :: N
! 
!       INTEGER N, NP, NNP
!       DOUBLE PRECISION D(NNP), SB(NNP), SA(NNP)
! C 
! C     Local variables
! C
!       INTEGER K, NS, I, J, NR, IP, NQ, II
!       DOUBLE PRECISION DF, U1, U2, RN, C
!       DOUBLE PRECISION ZERO, ONE, TWO, NINE
!       DOUBLE PRECISION rnorms(NNP)
!       DATA ZERO /0.0/, ONE /1.0/, TWO /2.0/, NINE /9.0/
! C
!       K = 1
! C
! C     Load SB with independent normal (0, 1) variates
! C
! 
!     1 continue
!       SB(K) = rnorms(k)
!       K = K + 1
!       IF (K .GT. NNP) GO TO 2
!       SB(K) = rnorsm(k)
!       K = K + 1
!       IF (K .LE. NNP) GO TO 1
!     2 NS = 0
! C
! C     Load diagonal elements with square root of chi-square variates
! C
!       DO 3 I = 1, NP
!         DF = N - I + 1
!         NS = NS + I
!         U1 = TWO / (NINE * DF)
!         U2 = ONE - U1
!         U1 = SQRT(U1)
! C
! C     Wilson-Hilferty formula for approximating chi-square variates
! C
!         SB(NS) = SQRT(DF * (U2 + SB(NS) * U1)**3)
!     3 CONTINUE
! C
!       RN = N
!       NR = 1
!       DO 5 I = 1, NP
!         NR = NR + I - 1
!         DO 5 J = I, NP
!           IP = NR
!           NQ = (J*J - J) / 2 + I - 1
!           C = ZERO
!           DO 4 K = I, J
!             IP = IP + K - 1
!             NQ = NQ + 1
!             C = C + SB(IP) * D(NQ)
!     4     CONTINUE
!           SA(IP) = C
!     5 CONTINUE
! C
!       DO 7 I = 1, NP
!         II = NP - I + 1
!         NQ = NNP - NP
!         DO 7 J = 1, I
!           IP = (II*II - II) / 2
!           C = ZERO
!           DO 6 K = I, NP
!             IP = IP + 1
!             NQ = NQ + 1
!             C = C + SA(IP) * SA(NQ)
!     6     CONTINUE
!           SA(NQ) = C / RN
!           NQ = NQ - 2 * NP + I + J - 1
!     7 CONTINUE
! C
!       RETURN
!       END
! C

! !      SUBROUTINE WSHRT_WRAP(DIN,N,NP,SAOUT,NNP,D,SA,SB)
!       SUBROUTINE WSHRT_WRAP(DIN,N,NP,SAOUT,NNP)
! cf2py intent(out) SAOUT
! cf2py intent(hide) NP
! cf2py integer intent(hide),depend(NP)::NNP=NP*(NP+1)/2
!       INTEGER N, NP, NNP, i, j, ni
!       DOUBLE PRECISION DIN(NP,NP), SAOUT(NP,NP)
!       DOUBLE PRECISION D(NNP), SA(NNP), SB(NNP)
!       
!       ni=0
!       do i=1,np
!           do j=i,np
!               ni = ni + 1
!               D(ni) = DIN(j,i)
!           end do
!       end do
!       
!       CALL WSHRT(D,N,NP,NNP,SB,SA)
!       
!       print *,D
!       print *,SA
!       
!       ni=0
!       do i=1,np
!           do j=i,np
!               ni = ni + 1
!               SAOUT(i,j) = SA(ni)
!               SAOUT(j,i) = SA(ni)
!           end do
!       end do
!       
!       RETURN
!       END

! rbin and fill_stdnormal don't seem to be used by anything
!       SUBROUTINE rbin(n,pp,x) 
! 
! cf2py double precision intent(in) :: pp
! cf2py integer intent(in) :: n
! cf2py integer intent(out) :: x  
! 
!       INTEGER n,x 
!       DOUBLE PRECISION pp,PI 
! C USES gammln,rand 
!       PARAMETER (PI=3.141592654) 
! C Returns as a floating-point number an integer value that is a random deviate drawn from 
! C a binomial distribution of n trials each of probability pp, using rand as a source 
! C of uniform random deviates. 
!       INTEGER j,nold
!       DOUBLE PRECISION am,em,en,g,oldg,p,pc,rn
!       DOUBLE PRECISION pclog,plog,pold,sq,t,y,gammln
!       SAVE nold,pold,pc,plog,pclog,en,oldg 
! C     Arguments from previous calls.
!       DATA nold /-1/, pold /-1./  
!       if(pp.le.0.5)then 
! C       The binomial distribution is invariant under changing pp to 
! C       1.-pp, if we also change the answer to n minus itself; 
! C       we’ll remember to do this below. 
!         p=pp 
!       else 
!         p=1.-pp 
!       endif 
! C     This is the mean of the deviate to be produced. 
!       am=n*p
!       if (n.lt.25) then 
! C       Use the direct method while n is not too large. This can 
! C       require up to 25 calls to ran1.
!         x=0. 
!         do 11 j=1,n 
! !           call random_number(rn)
!           rn = 0
!           if(rn.lt.p) x=x+1. 
!    11   enddo 
!       else if (am.lt.1.) then 
! C       If fewer than one event is expected out of 25 or more tri- 
! C       als, then the distribution is quite accurately Poisson. Use 
! C       direct Poisson method. 
!         g=dexp(-am) 
!         t=1. 
!         do 12 j=0,n
! !         call random_number(rn) 
!           rn = rand()
!         t=t*rn
!         if (t.lt.g) goto 1 
!    12   enddo  
!         j=n 
!     1   x=j 
!       else 
! C       Use the rejection method. 
!         if (n.ne.nold) then 
! C         If n has changed, then compute useful quantities. 
!           en=n 
!           oldg=gammln(en+1.) 
!           nold=n 
!         endif 
!         if (p.ne.pold) then 
! C         If p has changed, then compute useful quantities. 
!           pc=1.-p 
!           plog=dlog(p) 
!           pclog=dlog(pc) 
!           pold=p 
!         endif 
!         sq=sqrt(2.*am*pc) 
! C       The following code should by now seem familiar: rejection 
! C       method with a Lorentzian comparison function.
! !         call random_number(rn)
!           rn = rand()
!     2   y=tan(PI*rn) 
!         em=sq*y+am 
! C       Reject.
!         if (em.lt.0..or.em.ge.en+1.) goto 2  
! 
! C       Trick for integer-valued distribution.
!         em=int(em)
!         t=1.2*sq*(1.+y**2)*dexp(oldg-gammln(em+1.) 
!      +-gammln(en-em+1.)+em*plog+(en-em)*pclog) 
! C       Reject. This happens about 1.5 times per deviate, on average.
! !         call random_number(rn)
!           rn = rand()
!         if (rn.gt.t) goto 2 
!         x=em 
!         endif 
! C     Remember to undo the symmetry transformation. 
!       if (p.ne.pp) x=n-x 
!       return 
!       END 
! 
! 
! 
!       SUBROUTINE fill_stdnormal(array_in,n)
! 
! c Fills an input array with standard normals in-place.
! c Created 2/4/07, AP
! 
! cf2py double precision dimension(n),intent(inplace) :: array_in
! cf2py integer intent(hide),depend(array_in),check(n>0) :: n=len(array_in)
! 
!       INTEGER i, n, n_blocks, index
!       DOUBLE PRECISION U1, U2, array_in(n)
!       LOGICAL iseven
! 
!       iseven = (MOD(n,2) .EQ. 0)
! 
!       if(iseven) then
!         n_blocks = n/2
!       else 
!         n_blocks = (n-1)/2
!       endif
! 
!       do i=1,n_blocks
!         call RNORM(U1,U2)
!         index = 2*(i-1) + 1
!         array_in(index) = U1
!         array_in(index+1) = U2
!       enddo
! 
!       if(.NOT.iseven) then
!         call RNORM(U1,U2)
!         array_in(n) = U1
!       endif
! 
! 
!       return
!       END


c
! Commented this out- numpy's rng is much better.
!       double precision function whrand()
! c
! c     Algorithm AS 183 Appl. Statist. (1982) vol.31, no.2
! c
! c     Returns a pseudo-random number rectangularly distributed
! c     between 0 and 1.   The cycle length is 6.95E+12 (See page 123
! c     of Applied Statistics (1984) vol.33), not as claimed in the
! c     original article.
! c
! c     IX, IY and IZ should be set to integer values between 1 and
! c     30000 before the first entry.
! c
! c     Integer arithmetic up to 30323 is required.
! c
!       integer ix, iy, iz
!       common /randc/ ix, iy, iz
! c
!       ix = 171 * mod(ix, 177) - 2 * (ix / 177)
!       iy = 172 * mod(iy, 176) - 35 * (iy / 176)
!       iz = 170 * mod(iz, 178) - 63 * (iz / 178)
! c
!       if (ix .lt. 0) ix = ix + 30269
!       if (iy .lt. 0) iy = iy + 30307
!       if (iz .lt. 0) iz = iz + 30323
! c
! c     If integer arithmetic up to 5212632 is available, the preceding
! c     6 statements may be replaced by:
! c
! c     ix = mod(171 * ix, 30269)
! c     iy = mod(172 * iy, 30307)
! c     iz = mod(170 * iz, 30323)
! c
!       whrand = mod(float(ix) / 30269. + float(iy) / 30307. +
!      +                        float(iz) / 30323., 1.0)
!       return
!       end
! 
