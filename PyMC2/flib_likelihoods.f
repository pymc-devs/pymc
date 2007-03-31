
      DOUBLE PRECISION FUNCTION combinationln(n,k)

c Ln of the number of different combinations of n different things, taken k at a time. 
c DH, 5.02.2007

      IMPLICIT NONE
      INTEGER n, k
      DOUBLE PRECISION factln

      combinationln= factln(n) - factln(k) - factln(n-k)

      END FUNCTION combinationln


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


      FUNCTION gammln(xx) 
C Returns the value ln[gamma(xx)] for xx > 0. 

      DOUBLE PRECISION gammln, xx
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
      tmp=(x+0.5d0)*log(tmp)-tmp 
      ser=1.000000000190015d0 
      do j=1,6
         y=y+1.d0 
         ser=ser+cof(j)/y 
      enddo 
      gammln=tmp+log(stp*ser/x) 
      return 
      END

      DOUBLE PRECISION FUNCTION factrl(n) 
C Returns the value n! as a floating-point number. 

      INTEGER n 

      INTEGER j,ntop 
C Table to be filled in only as required. 
      DOUBLE PRECISION a(33),gammln 
      SAVE ntop,a 
C Table initialized with 0! only. 
      DATA ntop,a(1)/0,1./

      if (n.lt.0) then 
        write (*,*) 'negative factorial in factrl' 
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
        factrl=exp(gammln(n+1.)) 
      endif 
      return 
      END 

      DOUBLE PRECISION FUNCTION factln(n) 
C USES gammln Returns ln(n!). 

      INTEGER n 
      DOUBLE PRECISION a(100),gammln, pass_val 
      SAVE a 
C Initialize the table to negative values. 
      DATA a/100*-1./ 
      pass_val = n + 1
      if (n.lt.0) write (*,*) 'negative factorial in factln' 
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

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(k),intent(in) :: hist
cf2py double precision intent(in) :: mn,step
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(hist) :: k=len(hist)
cf2py double precision intent(out) :: like
            
      DOUBLE PRECISION hist(k),x(n),mn,step,val,like
      INTEGER n,k,i,j

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
        like = like + log(hist(j))
      enddo
      return
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
                
        low = lower(1)
        high = upper(1)       
        like = 0.0
        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            like = -3.4028235E+38
            RETURN
          else
            like = like - log(high-low)
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
      
      aa = a(1)
      cc = c(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_c = (nc .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Check parameter c > 0
c      CALL constrain(c, 0.0, Infinity, nc, .FALSE.)
c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)
c Check z > 0
c      CALL constrain(z, 0.0, Infinity, n, .FALSE.)
     
      like = 0.0
      do i=1,n
        if (not_scalar_a) aa = a(i)
        if (not_scalar_c) cc = c(i)
        if (not_scalar_scale) sigma = scale(i)
        t1 = exp(-z(i)**cc)
        pdf = aa*cc*(1.0-t1)**(aa-1.0)*t1*z(i)**(cc-1.0)
        like = like + log(pdf/sigma)
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
        ppf(i) = (-log(1.0 - q(i)**(1.0/ta)))**(1.0/tc)
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

      INTEGER x(n)
      DOUBLE PRECISION mu(nmu)
      DOUBLE PRECISION like,sumx, mut
      INTEGER n,i
      LOGICAL not_scalar_mu
      DOUBLE PRECISION factln

      not_scalar_mu =  (nmu .NE. 1)
      mut = mu(1)

c      CALL constrain(x,0,INFINITY,allow_equal=1)
c      CALL constrain(mu,0,INFINITY,allow_equal=0)

      sumx = 0.0
      sumfact = 0.0
      do i=1,n
        if (not_scalar_mu) mut = mu(i)
        sumx = sumx + x(i)*log(mut) - mut
        sumfact = sumfact + factln(x(i))
      enddo
      like = sumx - sumfact
      return
      END

      SUBROUTINE multinomial(x,n,p,nx,nn,np,k,like)

c Multinomial log-likelihood function     
c Updated 12/02/2007 DH. N-D still buggy.

cf2py integer dimension(nx,k),intent(in) :: x
cf2py integer dimension(nn), intent(in) :: n
cf2py double precision dimension(np,k),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
cf2py integer intent(hide),depend(n) :: nn=shape(n,0)
cf2py integer intent(hide),depend(p) :: np=shape(p,0)
cf2py integer intent(hide),depend(x,p),check(k==shape(p,1)) :: k=shape(x,1)
cf2py double precision intent(out) :: like      

      DOUBLE PRECISION like,sump
      DOUBLE PRECISION p(np,k), p_tmp
      INTEGER i,j,ll,n(nn),sumx, n_tmp
      INTEGER x(nx,k)
      DOUBLE PRECISION factln, log

      like = 0.0
      n_tmp = n(1)
      ll=1
      do j=1,nx
        sumx = 0
        sump = 0.0
        if (np .NE. 1) ll=j
        if (nn .NE. 1) n_tmp = n(j)
        do i=1,k
          p_tmp = p(ll,i)+1E-10
          like = like + x(j,i)*log(p_tmp) - factln(x(j,i))
          sumx = sumx + x(j,i)
          sump = sump + p_tmp
        enddo
        like=like+factln(n_tmp)
c I don't understand this
c like = like + factln(n_tmp-sumx)+(n_tmp-sumx)*log(max(1.0-sump,1E-10))
      enddo
      return
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
      DOUBLE PRECISION like
      INTEGER n,nalpha,nbeta,i
      LOGICAL not_scalar_alpha
      LOGICAL not_scalar_beta

      not_scalar_alpha = (nalpha .NE. 1)
      not_scalar_beta = (nbeta .NE. 1)
      alphat = alpha(1)
      betat = beta(1)

      like = 0.0      
      do i=1,n
        if (not_scalar_alpha) alphat = alpha(i)
        if (not_scalar_beta) betat = beta(i)
c normalizing constant
        like = like + (log(alphat) - alphat*log(betat))
c kernel of distribution
        like = like + (alphat-1) * log(x(i))
        like = like - (x(i)/betat)**alphat
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

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        like = like - 0.5 * tau_tmp * (x(i)-mu_tmp)**2
        like = like + 0.5*log(0.5*tau_tmp/PI)
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

      not_scalar_tau = (ntau .NE. 1)

      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_tau) tau_tmp = tau(i)
        like = like + 0.5 * (log(2. * tau_tmp / PI)) 
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

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        like = like + 0.5 * (log(tau_tmp) - log(2.0*PI)) 
        like = like - 0.5*tau_tmp*(log(x(i))-mu_tmp)**2 - log(x(i))
      enddo
      return
      END


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

      CALL standardize(x,mu,sigma,n,nmu,nsigma,z)

      xi_tmp = xi(1)
      sigma_tmp = sigma(1)
      LIKE = 0.0
      DO I=1,N
        if (nxi .NE. 1) xi_tmp = xi(i)
        if (nsigma .NE. 1) sigma_tmp = sigma(i)          
        IF (ABS(xi_tmp) .LT. 10.**(-5.)) THEN
          LIKE = LIKE - Z(I) - EXP(-Z(I))/SIGMA_TMP
        ELSE 
          EX(I) = 1. + xi_tmp*z(i)
          IF (EX(I) .LT. 0.) THEN
            LIKE = -3.4028235E+38
            RETURN
          ENDIF
          PEX(I) = EX(I)**(-1./xi_tmp)  
          LIKE = LIKE - LOG(sigma_tmp) - PEX(I) 
          LIKE = LIKE - (1./xi_tmp +1.)* LOG(EX(I))
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
          ppf(i) = -LOG(-LOG(q(i)))
        ELSE
          ppf(i) = 1./xi_tmp * ( (-log(q(i)))**(-xi_tmp) -1. )
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

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)
      like = 0.0
      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)
        like = like - gammln(alpha_tmp) - alpha_tmp*log(beta_tmp)
        like = like + (alpha_tmp - 1.0)*log(x(i)) - x(i)/beta_tmp
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

      alpha_tmp=alpha(1)
      beta_tmp=beta(1)
      like = 0.0
      do i=1,n
        if (na .NE. 1) alpha_tmp=alpha(i)
        if (nb .NE. 1) beta_tmp=beta(i)
        like = like - (gammln(alpha(i)) + alpha(i)*log(beta(i)))
        like = like - (alpha(i)+1.0)*log(x(i)) - 1./(x(i)*beta(i)) 
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

c      CALL constrain(d,x,total,allow_equal=1)
c      CALL constrain(red,x,total,allow_equal=1)
c      CALL constrain(x, 0, d, allow_equal=1)

      draws_tmp = draws(1)
      s_tmp = success(1)
      t_tmp = total(1)

      like = 0.0
      do i=1,n
c Combinations of x red balls
        if (nd .NE. 1) draws_tmp = draws(i)
        if (ns .NE. 1) s_tmp = success(i)
        if (nt .NE. 1) t_tmp = total(i) 
        like = like + combinationln(t_tmp-s_tmp, x(i))
        like = like + combinationln(s_tmp,draws_tmp-x(i))
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

      p_tmp = p(1)
      like = 0.0
      do i=1, n
        if (np .NE. 1) p_tmp = p(i)
        like = like + log(p_tmp) + (x(i)-1)* log(1-p_tmp)
      enddo
      return
      END SUBROUTINE geometric


      SUBROUTINE dirichlet(x,theta,k,nx,nt,like)

c Dirichlet multivariate log-likelihood function      

c Updated 22/01/2007 DH. 

cf2py double precision dimension(nx,k),intent(in) :: x
cf2py double precision dimension(nt,k),intent(in) :: theta
cf2py double precision intent(out) :: like
cf2py integer intent(hide), depend(x,theta),check(k==shape(theta,1)||(k==shape(theta,0) && shape(theta,1)==1)) :: k=shape(x,1)
cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
cf2py integer intent(hide),depend(theta,nx),check(nt==1 || nt==nx) :: nt=shape(theta,0)

      IMPLICIT NONE
      INTEGER i,j,nx,nt,k
      DOUBLE PRECISION like,sumt
      DOUBLE PRECISION x(nx,k),theta(nt,k)
      DOUBLE PRECISION theta_tmp(k)
      DOUBLE PRECISION gammln

      like = 0.0
      do j=1,k
        theta_tmp(j) = theta(1,j)
      enddo

      do i=1,nx
        sumt = 0.0
        do j=1,k
          if (nt .NE. 1) theta_tmp(j) = theta(i,j)      
c kernel of distribution      
          like = like + (theta_tmp(j)-1.0)*log(x(i,j))  
c normalizing constant        
          like = like - gammln(theta_tmp(j))
          sumt = sumt + theta_tmp(j)
        enddo
        like = like + gammln(sumt)
      enddo
      return
      END SUBROUTINE dirichlet


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

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)
      like = -nx*log(PI)
      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)
        like = like - log(btmp)
        like = like -  log( 1. + ((x(i)-atmp) / btmp) ** 2 )
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

      not_scalar_r = (nr .NE. 1)
      not_scalar_p = (np .NE. 1)

      r_tmp = r(1)
      p_tmp = p(1)
      like = 0.0
      do i=1,n
        if (not_scalar_r) r_tmp = r(i)
        if (not_scalar_p) p_tmp = p(i)
        like = like + r_tmp*log(p_tmp) + x(i)*log(1.-p_tmp)
        like = like + factln(x(i)+r_tmp-1)-factln(x(i))-factln(r_tmp-1) 
      enddo
      return
      END


      SUBROUTINE negbin2(x,mu,a,n,nmu,na,like)

c Negative binomial log-likelihood function 
c (alternative parameterization)    
c Updated 24/01/2007 DH.

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

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_a = (na .NE. 1)

      mu_tmp = mu(1)
      a_tmp = a(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_a) a_tmp=a(i)
        like=like+gammln(x(i)+a_tmp)-factln(x(i))-gammln(a_tmp)
        like=like+x(i)*(log(mu_tmp/a_tmp)-log(1.0+mu_tmp/a_tmp))
        like=like-a_tmp * log(1.0 + mu_tmp/a_tmp)
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

      not_scalar_n = (nn .NE. 1)
      not_scalar_p = (np .NE. 1) 

      ntmp = n(1)
      ptmp = p(1)

      like = 0.0
      do i=1,nx
        if (not_scalar_n) ntmp = n(i)
        if (not_scalar_p) ptmp = p(i)
        like = like + x(i)*log(ptmp) + (ntmp-x(i))*log(1.-ptmp)
        like = like + factln(ntmp)-factln(x(i))-factln(ntmp-x(i)) 
      enddo
      return
      END


      SUBROUTINE bernoulli(x,p,nx,np,like)
      
c TODO: Why is bernoulli actually binomial?

c Binomial log-likelihood function     
c Modified on Jan 16 2007 by D. Huard to allow scalar p.

cf2py integer dimension(nx),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(p),check(len(p)==1 || len(p)==len(x)):: np=len(p) 
cf2py double precision intent(out) :: like      
      IMPLICIT NONE

      INTEGER np,nx,i
      DOUBLE PRECISION p(np), ptmp, like
      INTEGER x(nx)
      LOGICAL not_scalar_p

C     Check parameter size
      not_scalar_p = (np .NE. 1)

      like = 0.0
      ptmp = p(1)
      do i=1,nx
        if (not_scalar_p) ptmp = p(i)
        like = like + log(ptmp**x(i)) + log((1.-ptmp)**(1-x(i)))
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

      atmp = alpha(1)
      btmp = beta(1)
      like = 0.0
      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        like = like + (gammln(atmp+btmp) - gammln(atmp) - gammln(btmp))
        like = like + (atmp-1.0)*log(x(i)) + (btmp-1.0)*log(1.0-x(i))
      enddo   

      return
      END
      

      SUBROUTINE mvhyperg(x,color,k,like)

c Multivariate hypergeometric log-likelihood function

cf2py integer dimension(k),intent(in) :: x,color
cf2py integer intent(hide),depend(x) :: k=len(x)
cf2py double precision intent(out) :: like

      INTEGER x(k),color(k)
      INTEGER d,total,i,k
      DOUBLE PRECISION like
      DOUBLE PRECISION factln

      total = 0
      d = 0
      like = 0.0
      do i=1,k
c Combinations of x balls of color i     
        like = like + factln(color(i))-factln(x(i))
     +-factln(color(i)-x(i))
        d = d + x(i)
        total = total + color(i)
      enddo
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

      like = 0.0
      sumt = 0.0
      sumx = 0
      do 222 i=1,k
c kernel of distribution      
        like = like + log(x(i) + theta(i)) - log(theta(i)) 
        sumt = sumt + theta(i)
        sumx = sumx + x(i)
  222 continue
c normalizing constant 
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

c determinants
      call dtrm(X,k,dx)
      call dtrm(sigma,k,db)
c trace of sigma*X     
      call matmult(sigma,X,bx,k,k,k,k)
      call trace(bx,k,tbx)

      like = (n - k - 1)/2.0 * log(dx)
      like = like + (n/2.0)*log(db)
      like = like - 0.5*tbx
      like = like - (n*k/2.0)*log(2.0)

      do i=1,k
        a = (n - i + 1)/2.0
        call gamfun(a, g)
        like = like - log(g)
      enddo

      return
      END


      SUBROUTINE mvnorm(x,mu,tau,k,like)

c Multivariate normal log-likelihood function      

cf2py double precision dimension(k),intent(in) :: x,mu
cf2py double precision dimension(k,k),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)

      INTEGER i,k
      DOUBLE PRECISION x(k),dt(1,k),dtau(k),mu(k),d(k),tau(k,k)
      DOUBLE PRECISION like,det,dtaud

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 

c calculate determinant of precision matrix     
      call dtrm(tau,k,det)

c calculate d=x-mu
      do i=1,k
        d(i) = x(i)-mu(i)
      enddo

c transpose 
      call trans(d,dt,k,1)
c mulitply t(d) by tau 
      call matmult(dt,tau,dtau,1,k,k,k)
c multiply dtau by d      
      call matmult(dtau,d,dtaud,1,k,k,1)

      like = 0.5*log(det) - (k/2.0)*log(2.0*PI) - (0.5*dtaud)

      return
      END


        SUBROUTINE vec_mvnorm(x,mu,tau,k,n,nmu,like)

c Vectorized multivariate normal log-likelihood function 
c CREATED 12/06 DH.     
c TODO: link BLAS/LAPACK, eliminate explicit transposition

cf2py double precision dimension(k,n),intent(in) :: x
cf2py double precision dimension(k,nmu),intent(in) :: mu
cf2py double precision dimension(k,k),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=shape(x,0)
cf2py integer, intent(hide), depend(x):: n=shape(x,1)
cf2py integer, intent(hide), depend(mu):: nmu=shape(mu,1)

      INTEGER i,j,k,n,nmu
      DOUBLE PRECISION x(k,n), mu(k,n), tau(k,k),mut(k)
      DOUBLE PRECISION dt(n,k),dtau(n,k),d(k,n), s(n)
      DOUBLE PRECISION like,det
      LOGICAL mu_not_1d

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      mu_not_1d = (nmu .NE. 1)
c calculate determinant of precision matrix     
      call dtrm(tau,k,det)


c calculate d=(x-mu)
      do i=1,k
        mut(i) = mu(i,1)
        do j=1,n
          if (mu_not_1d) mut(i) = mu(i,j)
          d(i,j) = x(i,j)-mut(i)
        enddo
      enddo

c transpose 
      call trans(d,dt,k,n)
c mulitply t(d) by tau -> dtau (n,k)
      call matmult(dt,tau,dtau,n,k,k,k)

      like = 0.0
      do j=1,n
        s(j) = 0.0
        do i=1,k
          s(j) = s(j) + dtau(j,i)*d(i,j)
        enddo
        like = like + s(j)
      enddo

      like = n*0.5*log(det) - (n*k/2.0)*log(2.0*PI) - (0.5*like)
      END subroutine

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
      tmp = tmp - (x+0.5) * log(tmp)
      ser = 1.000000000190015
      do i=1,6
        x = x+1
        ser = ser + coeff(i)/x
      enddo
      gx = -tmp + log(2.50662827465*ser/xx)
      return
      END


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
99      continue
88    continue

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

      DIMENSION A(N,N),INDX(N)

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


      DIMENSION A(N,N),INDX(N),C(N)

C Initialize the index

      DO     50    I = 1, N
        INDX(I) = I
   50 CONTINUE

C Find the rescaling factors, one from each row

        DO     100   I = 1, N
          C1= 0.0
          DO    90   J = 1, N
            C1 = AMAX1(C1,ABS(A(I,J)))
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
      bico=nint(exp(factln(n)-factln(k)-factln(n-k))) 
      return 
      END

      subroutine chol(n,a,c)
c...perform a Cholesky decomposition of matrix a, returned as c
      implicit double precision*8 (a-h,o-z)
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

