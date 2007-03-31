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
C     Note- this seems to be faster than Leva's algorithm from
C     ACM Trans Math Soft, Dec. 1992 - AP
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



      SUBROUTINE fill_stdnormal(array_in,n)

c Fills an input array with standard normals in-place.
c Created 2/4/07, AP

cf2py real dimension(n),intent(inplace) :: array_in
cf2py integer intent(hide),depend(array_in),check(n>0) :: n=len(array_in)

      INTEGER i, n, n_blocks, index
      REAL U1, U2, array_in(n)
      LOGICAL iseven

      iseven = (MOD(n,2) .EQ. 0)

      if(iseven) then
        n_blocks = n/2
      else 
        n_blocks = (n-1)/2
      endif

      do i=1,n_blocks
        call RNORM(U1,U2)
        index = 2*(i-1) + 1
        array_in(index) = U1
        array_in(index+1) = U2
      enddo

      if(.NOT.iseven) then
        call RNORM(U1,U2)
        array_in(n) = U1
      endif


      return
      END
