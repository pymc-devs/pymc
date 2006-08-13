      real function whrand()
c
c     Algorithm AS 183 Appl. Statist. (1982) vol.31, no.2
c
c     Returns a pseudo-random number rectangularly distributed
c     between 0 and 1.   The cycle length is 6.95E+12 (See page 123
c     of Applied Statistics (1984) vol.33), not as claimed in the
c     original article.
c
c     IX, IY and IZ should be set to integer values between 1 and
c     30000 before the first entry.
c
c     Integer arithmetic up to 30323 is required.
c
      integer ix, iy, iz
      common /randc/ ix, iy, iz
c
      ix = 171 * mod(ix, 177) - 2 * (ix / 177)
      iy = 172 * mod(iy, 176) - 35 * (iy / 176)
      iz = 170 * mod(iz, 178) - 63 * (iz / 178)
c
      if (ix .lt. 0) ix = ix + 30269
      if (iy .lt. 0) iy = iy + 30307
      if (iz .lt. 0) iz = iz + 30323
c
c     If integer arithmetic up to 5212632 is available, the preceding
c     6 statements may be replaced by:
c
c     ix = mod(171 * ix, 30269)
c     iy = mod(172 * iy, 30307)
c     iz = mod(170 * iz, 30323)
c
      whrand = mod(float(ix) / 30269. + float(iy) / 30307. +
     +                        float(iz) / 30323., 1.0)
      return
      end

	    real function uniform()
c
c	Generate uniformly distributed random numbers using the 32-bit
c	generator from figure 3 of:
c	L`Ecuyer, P. Efficient and portable combined random number
c	generators, C.A.C.M., vol. 31, 742-749 & 774-?, June 1988.
c
c	The cycle length is claimed to be 2.30584E+18
c
c	Seeds can be set by calling the routine set_uniform
c
c	It is assumed that the Fortran compiler supports long variable
c	names, and integer*4.
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
c	Set seeds for the uniform random number generator.
c
	    integer*4 s1, s2, seed1, seed2
	    common /unif_seeds/ s1, s2
	    save /unif_seeds/

	    s1 = seed1
	    s2 = seed2
	    return
      END
	  

      SUBROUTINE rcat(hist,mn,step,n,s,k)

c Returns n samples from categorical random variable (histogram)

cf2py real dimension(k),intent(in) :: hist
cf2py real intent(in) :: mn,step
cf2py integer intent(in) :: n
cf2py real dimension(n),intent(out) :: s
cf2py integer intent(hide),depend(hist) :: k=len(hist)

      REAL hist(k),s(n),mn,step,sump,u,rand
      INTEGER n,k,i,j

c repeat for n samples
      do i=1,n
c initialize sum      
        sump = 0.0
c random draw
        u = rand()
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

      SUBROUTINE categor(x,hist,mn,step,n,k,like)

c Categorical log-likelihood function

cf2py real dimension(n),intent(in) :: x
cf2py real dimension(k),intent(in) :: hist
cf2py real intent(in) :: mn,step
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(hist) :: k=len(hist)
cf2py real intent(out) :: like
            
      REAL hist(k),x(n),mn,step,val,like
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


      SUBROUTINE hyperg(x,d,red,total,n,like)

c Hypergeometric log-likelihood function

c Distribution models the probability of drawing x red balls in d
c draws from an urn of 'red' red balls and 'total' total balls.

cf2py integer dimension(n),intent(in) :: x, d,red,total
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py real intent(out) :: like

      INTEGER x(n),d(n),red(n),total(n)
      INTEGER i,n
      REAL like
      
      like = 0.0
      do i=1,n
c Combinations of x red balls      
        like = like + factln(red(i))-factln(x(i))-factln(red(i)-x(i))
c Combinations of d-x other balls        
        like = like + factln(total(i)-red(i))-factln(d(i)-x(i))
     +-factln(total(i)-red(i)-d(i)+x(i))
c Combinations of d draws from total
        like = like - (factln(total(i))-factln(d(i))- 
     +-factln(total(i)-d(i)))
      enddo
      return
      END

      
      SUBROUTINE mvhyperg(x,color,k,like)

c Multivariate hypergeometric log-likelihood function

cf2py integer dimension(k),intent(in) :: x,color
cf2py integer intent(hide),depend(x) :: k=len(x)
cf2py real intent(out) :: like

      INTEGER x(k),color(k)
      INTEGER d,total,i,k
      REAL like
      
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


      SUBROUTINE poisson(x,mu,n,like)
      
c Poisson log-likelihood function      

cf2py integer dimension(n),intent(in) :: x
cf2py real dimension(n),intent(in) :: mu
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
     
      INTEGER x(n)
      REAL mu(n)
      REAL like,sumx
      INTEGER n,i

      sumx = 0.0
      sumfact = 0.0
      do i=1,n
        sumx = sumx + x(i)*log(mu(i)) - mu(i)
        sumfact = sumfact + factln(x(i))
      enddo
      like = sumx - sumfact
      return
      END


      SUBROUTINE weibull(x,alpha,beta,n,like)

c Weibull log-likelihood function      

cf2py real dimension(n),intent(in) :: x,alpha,beta 
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)

      REAL x(n),alpha(n),beta(n)
      REAL like
      INTEGER n,i


      like = 0.0      
      do i=1,n
c normalizing constant
        like = like + (log(alpha(i)) - alpha(i)*log(beta(i)))
c kernel of distribution
        like = like + (alpha(i)-1) * log(x(i))
        like = like - (x(i)/beta(i))**alpha(i)
      enddo
      return
      END


      SUBROUTINE cauchy(x,alpha,beta,n,like)

c Cauchy log-likelihood function      

cf2py real dimension(n),intent(in) :: x,alpha,beta
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)

      REAL x(n),alpha(n),beta(n)
      REAL like
      INTEGER n,i
      PARAMETER (PI=3.141592653589793238462643d0) 
      
      like = -n*log(PI)
      do i=1,n
        like = like + log(beta(i))
        like = like -  log( 1. + ((x(i)-alpha(i)) / beta(i)) ** 2 )
      enddo
      return
      END

      SUBROUTINE negbin(x,r,p,n,like)

c Negative binomial log-likelihood function     
      
cf2py integer dimension(n),intent(in) :: x,r
cf2py real dimension(n),intent(in) :: p
cf2py integer intent(hide),depend(x) :: m=len(x)
cf2py real intent(out) :: like      
      
      REAL like
      REAL p(n)
      INTEGER n,i
      INTEGER x(n),r(n)

      like = 0.0
      do i=1,n
        like = like + r(i)*log(p(i)) + x(i)*log(1.-p(i))
        like = like + factln(x(i)+r(i)-1)-factln(x(i))-factln(r(i)-1) 
      enddo
      return
      END


      SUBROUTINE binomial(x,n,p,m,like)

c Binomial log-likelihood function     
      
cf2py integer dimension(m),intent(in) :: x,n
cf2py real dimension(m),intent(in) :: p
cf2py integer intent(hide),depend(x) :: m=len(x)
cf2py real intent(out) :: like      
      
      REAL like
      REAL p(m)
      INTEGER m,i
      INTEGER x(m),n(m)

      like = 0.0
      do i=1,m
        like = like + x(i)*log(p(i)) + (n(i)-x(i))*log(1.-p(i))
        like = like + factln(n(i))-factln(x(i))-factln(n(i)-x(i)) 
      enddo
      return
      END


      SUBROUTINE bernoulli(x,p,m,like)

c Binomial log-likelihood function     
      
cf2py integer dimension(m),intent(in) :: x
cf2py real dimension(m),intent(in) :: p
cf2py integer intent(hide),depend(x) :: m=len(x)
cf2py real intent(out) :: like      
      
      REAL like
      REAL p(m)
      INTEGER m,i
      INTEGER x(m)

      like = 0.0
      do i=1,m
        like = like + log(p(i)**x(i) * (1-p(i))**(1-x(i)))
      enddo
      return
      END


      SUBROUTINE multinomial(x,n,p,m,like)

c Multinomial log-likelihood function     

cf2py integer dimension(m),intent(in) :: x
cf2py integer intent(in) :: n
cf2py real dimension(m),intent(in) :: p
cf2py integer intent(hide),depend(x) :: m=len(x)
cf2py real intent(out) :: like      

      REAL like,sump,pp
      REAL p(m)
      INTEGER m,i,n,sumx
      INTEGER x(m)

      like = 0.0
      sumx = 0
      sump = 0.0
      do i=1,m
        pp = p(i)+1E-10
        like = like + x(i)*log(pp) - factln(x(i))
        sumx = sumx + x(i)
        sump = sump + pp
      enddo
      like = like + factln(n) + (n-sumx)*log(max(1.0-sump,1E-10)) 
     +- factln(n-sumx)
      return
      END
      

      SUBROUTINE normal(x,mu,tau,n,like)

c Normal log-likelihood function      

cf2py real dimension(n),intent(in) :: x,mu,tau
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
      
      INTEGER n,i
      REAL like
      REAL x(n),mu(n),tau(n)

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 

      like = 0.0
      do i=1,n
        like = like - 0.5 * tau(i) * (x(i)-mu(i))**2
        like = like + 0.5*log(0.5*tau(i)/PI)
      enddo
      return
      END


      SUBROUTINE hnormal(x,tau,n,like)

c Half-normal log-likelihood function    

cf2py real dimension(n),intent(in) :: x,tau
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
      
      INTEGER n,i
      REAL like
      REAL x(n),tau(n)

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 

      like = 0.0
      do i=1,n
        like = like + 0.5 * (log(2. * tau(i) / PI)) 
        like = like - (0.5 * x(i)**2 * tau(i))
      enddo
      return
      END


      SUBROUTINE lognormal(x,mu,tau,n,like)

c Log-normal log-likelihood function

cf2py real dimension(n),intent(in) :: x,mu,tau
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
      
      INTEGER n,i
      REAL like
      REAL x(n),mu(n),tau(n)

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 

      like = 0.0
      do i=1,n
        like = like + 0.5 * (log(tau(i)) - log(2.0*PI)) 
        like = like - 0.5*tau(i)*(log(x(i))-mu(i))**2 - log(x(i))
      enddo
      return
      END
      
      
      SUBROUTINE gamma(x,alpha,beta,n,like)

c Gamma log-likelihood function      

cf2py real dimension(n),intent(in) :: x,alpha,beta
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)

      INTEGER i,n
      REAL like
      REAL x(n),alpha(n),beta(n)

      like = 0.0
      do i=1,n
        like = like - (gammln(alpha(i)) + alpha(i)*log(beta(i)))
        like = like + (alpha(i) - 1.0)*log(x(i)) - x(i)/beta(i)
      enddo     

      return
      END
      
      SUBROUTINE igamma(x,alpha,beta,n,like)

c Inverse gamma log-likelihood function      

cf2py real dimension(n),intent(in) :: x,alpha,beta
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)

      INTEGER i,n
      REAL like
      REAL x(n),alpha(n),beta(n)
      
      like = 0.0
      do i=1,n
        like = like - (gammln(alpha(i)) + alpha(i)*log(beta(i)))
        like = like - (alpha(i)+1.0)*log(x(i)) - 1./(x(i)*beta(i)) 
      enddo

      return
      END


      SUBROUTINE beta(x,a,b,n,like)

c Beta log-likelihood function      
      
cf2py real dimension(n),intent(in) :: x,a,b
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)

      INTEGER i,n
      REAL like
      REAL x(n),a(n),b(n)

      like = 0.0
      do i=1,n
        like = like + (gammln(a(i)+b(i)) - gammln(a(i)) - gammln(b(i)))
        like = like + (a(i)-1.0)*log(x(i)) + (b(i)-1.0)*log(1.0-x(i))
      enddo   

      return
      END


      SUBROUTINE dirichlet(x,theta,k,like)

c Dirichlet log-likelihood function      
      
cf2py real dimension(k),intent(in) :: x,theta      
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)

      INTEGER i,k
      REAL like,sumt
      REAL x(k),theta(k)

      like = 0.0
      sumt = 0.0
      do 111 i=1,k
c kernel of distribution      
        like = like + (theta(i)-1.0)*log(x(i))  
c normalizing constant        
        like = like - gammln(theta(i))
        sumt = sumt + theta(i)
  111 continue
      like = like + gammln(sumt)

      return
      END
      

      SUBROUTINE dirmultinom(x,theta,k,like)

c Dirichlet-multinomial log-likelihood function      
      
cf2py integer dimension(k),intent(in) :: x
cf2py real dimension(k),intent(in) :: theta      
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)

      INTEGER i,k,sumx
      INTEGER x(k)
      REAL like,sumt
      REAL theta(k)

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

cf2py real dimension(k,k),intent(in) :: X,sigma
cf2py real intent(in) :: n
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)

      INTEGER i,k
      REAL X(k,k),sigma(k,k),bx(k,k)
      REAL dx,n,db,tbx,a,g,like
      
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
      
cf2py real dimension(k),intent(in) :: x,mu
cf2py real dimension(k,k),intent(in) :: tau
cf2py real intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)

      INTEGER i,k
      REAL x(k),dt(1,k),dtau(k),mu(k),d(k),tau(k,k)
      REAL like,det,dtaud
      
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

      
      SUBROUTINE trace(mat,k,tr)

c matrix trace (sum of diagonal elements)
      
      INTEGER k,i
      REAL mat(k,k),tr

      tr = 0.0
      do i=1,k
        tr = tr + mat(k,k)
      enddo
      return
      END
   
      
      SUBROUTINE gamfun(xx,gx)

c the gamma function

cf2py real intent(in) :: xx
cf2py real intent(out) :: gx

      INTEGER i
      REAL x,xx,ser,tmp,gx
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

cf2py real dimension(m,n),intent(in) :: mat
cf2py real dimension(n,m),intent(out) :: tmat
cf2py integer intent(hide),depend(mat) :: m=len(mat)
cf2py integer intent(hide),depend(mat) :: n=shape(mat,1)

      INTEGER i,j,m,n
      REAL mat(m,n),tmat(n,m)
      
      do 88 i=1,m
        do 99 j=1,n
          tmat(j,i) = mat(i,j)
99      continue
88    continue
      
      return
      END

      
      SUBROUTINE matmult(mat1, mat2, prod, m, n, p, q)
      
c matrix multiplication

cf2py real dimension(m,q),intent(out) :: prod
cf2py real dimension(m,n),intent(in) :: mat1
cf2py real dimension(p,q),intent(in) :: mat2
cf2py integer intent(hide),depend(mat1) :: m=len(mat1),n=shape(mat1,1)
cf2py integer intent(hide),depend(mat2) :: p=len(mat2),q=shape(mat2,1)


      INTEGER i,j,k,m,n,p,q
      REAL mat1(m,n), mat2(p,q), prod(m,q)
      REAL sum
      
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
        pause 'Matrix dimensions do not match'
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

cf2py real dimension(N,N),intent(in) :: A
cf2py real intent(out) :: D
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

      FUNCTION gammln(xx) 
C Returns the value ln[gamma(xx)] for xx > 0. 

      REAL gammln,xx         
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

      FUNCTION factrl(n) 
C Returns the value n! as a floating-point number. 


      INTEGER n 
      REAL factrl 

      INTEGER j,ntop 
C Table to be filled in only as required. 
      REAL a(33),gammln 
      SAVE ntop,a 
C Table initialized with 0! only. 
      DATA ntop,a(1)/0,1./

      if (n.lt.0) then 
        pause 'negative factorial in factrl' 
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

      FUNCTION factln(n) 
C USES gammln Returns ln(n!). 

      INTEGER n 
      REAL factln 
      REAL a(100),gammln 
      SAVE a 
C Initialize the table to negative values. 
      DATA a/100*-1./ 
      if (n.lt.0) pause 'negative factorial in factln' 
C In range of the table. 
      if (n.le.99) then
C If not already in the table, put it in.  
        if (a(n+1).lt.0.) a(n+1)=gammln(n+1.) 
        factln=a(n+1) 
      else 
C Out of range of the table. 
        factln=gammln(n+1.) 
      endif 
      return 
      END

      FUNCTION bico(n,k) 
C USES factln Returns the binomial coefficient as a 
C floating point number.
      INTEGER k,n
      REAL bico
C The nearest-integer function cleans up roundoff error
C for smaller values of n and k. 
      bico=nint(exp(factln(n)-factln(k)-factln(n-k))) 
      return 
      END

      subroutine chol(n,a,c)
c...perform a Cholesky decomposition of matrix a, returned as c
      implicit real*8 (a-h,o-z)
      real c(n,n),a(n,n)

cf2py real dimension(n,n),intent(in) :: a
cf2py real dimension(n,n),intent(out) :: c
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
      
      SUBROUTINE rbin(n,pp,x) 
  
cf2py real intent(in) :: pp
cf2py integer intent(in) :: n
cf2py integer intent(out) :: x  
      
      INTEGER n,x 
      REAL pp,PI 
C USES gammln,rand 
      PARAMETER (PI=3.141592654) 
C Returns as a floating-point number an integer value that is a random deviate drawn from 
C a binomial distribution of n trials each of probability pp, using rand as a source 
C of uniform random deviates. 
      INTEGER j,nold
      REAL am,em,en,g,oldg,p,pc,pclog,plog,pold,sq,t,y,gammln,rand
      SAVE nold,pold,pc,plog,pclog,en,oldg 
C     Arguments from previous calls.
      DATA nold /-1/, pold /-1./  
      if(pp.le.0.5)then 
C       The binomial distribution is invariant under changing pp to 
C       1.-pp, if we also change the answer to n minus itself; 
C       we’ll remember to do this below. 
        p=pp 
      else 
        p=1.-pp 
      endif 
C     This is the mean of the deviate to be produced. 
      am=n*p
      if (n.lt.25) then 
C       Use the direct method while n is not too large. This can 
C       require up to 25 calls to ran1.
        x=0. 
        do 11 j=1,n 
          if(rand().lt.p) x=x+1. 
   11   enddo 
      else if (am.lt.1.) then 
C       If fewer than one event is expected out of 25 or more tri- 
C       als, then the distribution is quite accurately Poisson. Use 
C       direct Poisson method. 
        g=exp(-am) 
        t=1. 
        do 12 j=0,n 
        t=t*rand() 
        if (t.lt.g) goto 1 
   12   enddo  
        j=n 
    1   x=j 
      else 
C       Use the rejection method. 
        if (n.ne.nold) then 
C         If n has changed, then compute useful quantities. 
          en=n 
          oldg=gammln(en+1.) 
          nold=n 
        endif 
        if (p.ne.pold) then 
C         If p has changed, then compute useful quantities. 
          pc=1.-p 
          plog=log(p) 
          pclog=log(pc) 
          pold=p 
        endif 
        sq=sqrt(2.*am*pc) 
C       The following code should by now seem familiar: rejection 
C       method with a Lorentzian comparison function.
    2   y=tan(PI*rand()) 
        em=sq*y+am 
C       Reject.
        if (em.lt.0..or.em.ge.en+1.) goto 2  
         
C       Trick for integer-valued distribution.
        em=int(em)
        t=1.2*sq*(1.+y**2)*exp(oldg-gammln(em+1.) 
     +-gammln(en-em+1.)+em*plog+(en-em)*pclog) 
C       Reject. This happens about 1.5 times per deviate, on average.
        if (rand().gt.t) goto 2 
        x=em 
        endif 
C     Remember to undo the symmetry transformation. 
      if (p.ne.pp) x=n-x 
      return 
      END 
      
      
      SUBROUTINE RNORM(U1, U2)
C
C     ALGORITHM AS 53.1  APPL. STATIST. (1972) VOL.21, NO.3
C
C     Sets U1 and U2 to two independent standardized random normal
C     deviates.   This is a Fortran version of the method given in
C     Knuth(1969).
C
C     Function RAND must give a result randomly and rectangularly
C     distributed between the limits 0 and 1 exclusive.
C
      REAL U1, U2
      REAL RAND
C
C     Local variables
C
      REAL X, Y, S, ONE, TWO
      DATA ONE /1.0/, TWO /2.0/
C
    1 X = RAND()
      Y = RAND()
      X = TWO * X - ONE
      Y = TWO * Y - ONE
      S = X * X + Y * Y
      IF (S .GT. ONE) GO TO 1
      S = SQRT(- TWO * LOG(S) / S)
      U1 = X * S
      U2 = Y * S
      RETURN
      END


      SUBROUTINE WSHRT(D, N, NP, NNP, SB, SA)
C
C     ALGORITHM AS 53  APPL. STATIST. (1972) VOL.21, NO.3
C
C     Wishart variate generator.  On output, SA is an upper-triangular
C     matrix of size NP * NP (written in linear form, column ordered)
C     whose elements have a Wishart(N, SIGMA) distribution.
C
C     D is an upper-triangular array such that SIGMA = D'D (see AS 6)
C
C     Auxiliary function required: a random no. generator called RAND.
C     The Wichmann & Hill generator is included here.   It should be
C     initialized in the calling program.


cf2py real dimension(NNP),intent(in) :: D
cf2py real dimension(NNP),intent(out) :: SA
cf2py real dimension(NNP),intent(hide) :: SB
cf2py integer intent(hide),depend(D) :: NNP=len(D)
cf2py integer intent(in) :: NP
cf2py integer intent(in) :: N

      INTEGER N, NP, NNP
      REAL D(NNP), SB(NNP), SA(NNP)
C
C     Local variables
C
      INTEGER K, NS, I, J, NR, IP, NQ, II
      REAL DF, U1, U2, RN, C
      REAL ZERO, ONE, TWO, NINE
      DATA ZERO /0.0/, ONE /1.0/, TWO /2.0/, NINE /9.0/
C
      K = 1
    1 CALL RNORM(U1, U2)
C
C     Load SB with independent normal (0, 1) variates
C
      SB(K) = U1
      K = K + 1
      IF (K .GT. NNP) GO TO 2
      SB(K) = U2
      K = K + 1
      IF (K .LE. NNP) GO TO 1
    2 NS = 0
C
C     Load diagonal elements with square root of chi-square variates
C
      DO 3 I = 1, NP
        DF = N - I + 1
        NS = NS + I
        U1 = TWO / (NINE * DF)
        U2 = ONE - U1
        U1 = SQRT(U1)
C
C     Wilson-Hilferty formula for approximating chi-square variates
C
        SB(NS) = SQRT(DF * (U2 + SB(NS) * U1)**3)
    3 CONTINUE
C
      RN = N
      NR = 1
      DO 5 I = 1, NP
        NR = NR + I - 1
        DO 5 J = I, NP
          IP = NR
          NQ = (J*J - J) / 2 + I - 1
          C = ZERO
          DO 4 K = I, J
            IP = IP + K - 1
            NQ = NQ + 1
            C = C + SB(IP) * D(NQ)
    4     CONTINUE
          SA(IP) = C
    5 CONTINUE
C
      DO 7 I = 1, NP
        II = NP - I + 1
        NQ = NNP - NP
        DO 7 J = 1, I
          IP = (II*II - II) / 2
          C = ZERO
          DO 6 K = I, NP
            IP = IP + 1
            NQ = NQ + 1
            C = C + SA(IP) * SA(NQ)
    6     CONTINUE
          SA(NQ) = C / RN
          NQ = NQ - 2 * NP + I + J - 1
    7 CONTINUE
C
      RETURN
      END
C

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
C    Input, real ( kind = 8 ) X, the point at which the polynomials are 
C    to be evaluated.
C
C    Output, real ( kind = 8 ) CX(0:N), the values of the first N+1 Hermite
C    polynomials at the point X.
C

cf2py real intent(in) :: x
cf2py integer intent(in) :: n
cf2py real dimension(n+1),intent(out) :: cx
      
      integer n,i
      real cx(n+1)
      real x

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

