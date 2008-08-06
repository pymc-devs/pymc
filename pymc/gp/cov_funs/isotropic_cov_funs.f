! Copyright (c) Anand Patil, 2007

c A collection of covariance functions for Gaussian processes.
c By convention, the first dimension of each input array iterates over points, and 
c subsequent dimensions iterate over spatial dimensions.

      SUBROUTINE imul(C,a,nx,ny,cmin,cmax,symm)

cf2py intent(inplace) C
cf2py integer intent(in), optional :: cmin = 0
cf2py integer intent(in), optional :: cmax = -1
cf2py intent(hide) nx,ny
cf2py logical intent(in), optional:: symm=0
cf2py threadsafe

      DOUBLE PRECISION C(nx,ny)
      DOUBLE PRECISION a
      INTEGER nx, ny, i, j, cmin, cmax
      LOGICAL symm

      EXTERNAL DSCAL

      if (cmax.EQ.-1) then
          cmax = ny
      end if

      if (symm) then           
       
        do j=cmin+1,cmax
            do i=1,j
                C(i,j) = C(i,j) * a
            end do
!           CALL DSCAL(j,a,C(1,j),1)
        enddo

      else

        do j=cmin+1,cmax
            do i=1,nx
                C(i,j) = C(i,j) * a
            end do
 !          CALL DSCAL(nx,a,C(1,j),1)
        enddo
      endif  


      RETURN
      END



      SUBROUTINE symmetrize(C,nx,cmin,cmax)

cf2py intent(inplace) C
cf2py intent(hide) nx
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe


      DOUBLE PRECISION C(nx,nx)
      INTEGER nx, i, j, cmin, cmax
      
      if (cmax.EQ.-1) then
          cmax = nx
      end if

      do j=cmin,cmax
          do i=1,j-1
              C(j,i) = C(i,j)
          end do
      end do

      RETURN
      END



      SUBROUTINE stein_spatiotemporal
     *(C,Gt,origin_val,Bk,
     * cmin,cmax,nx,ny,symm)

cf2py threadsafe
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py intent(inplace) C
cf2py intent(hide) nx, ny, Bk
cf2py logical intent(in), optional:: symm=0
cf2py double precision intent(in),check(origin_val>0)::origin_val

      DOUBLE PRECISION C(nx,ny), Gt(nx,ny)
      DOUBLE PRECISION origin_val
      DOUBLE PRECISION rem, dd_here
      DOUBLE PRECISION GA, prefac, snu
      INTEGER nx, ny, i, j, fl, N, nd, cmin, cmax
      DOUBLE PRECISION BK(50), DGAMMA
      DOUBLE PRECISION sp_x(nx,2), sp_y(ny,2)
      LOGICAL symm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if
      
      if (symm) then           
       
        do j=cmin+1,cmax
          
          C(j,j) = 1.0D0
          
          do i=1,j-1
                        
! ================================
! = gamma(t) can be changed here =
! ================================
            dd_here=Gt(i,j)
            


            if (C(i,j) .EQ. 0.0D0) then
              C(i,j)=origin_val / dd_here
            else
              if (dd_here .GT. 5.0D0) then
                  C(i,j)=dexp(-C(i,j)**2)/dd_here*origin_val
                  goto 1
              endif      

              GA = DGAMMA(dd_here+1.0D0)
              prefac = 0.5D0 ** (dd_here-1.0D0) / GA
              prefac = prefac * origin_val
              snu = DSQRT(dd_here) * 2.0D0
              fl = INT(dd_here)
              rem = dd_here - fl
              N = fl
                            
              C(i,j) = C(i,j) * snu
              CALL RKBESL(C(i,j),rem,fl+1,1,BK,N)              
              C(i,j)=prefac * (C(i,j) ** dd_here) * BK(fl+1)
                            

            endif
    1 continue        
!     1       C(j,i)=C(i,j)
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
              
! ================================
! = gamma(t) can be changed here =
! ================================
            dd_here=Gt(i,j)            
            
            if (C(i,j) .EQ. 0.0D0) then
              C(i,j)=origin_val / dd_here
            else
              if (dd_here .GT. 5.0D0) then
                C(i,j)=dexp(-C(i,j)**2)/dd_here*origin_val
                goto 2
              endif      

              GA = DGAMMA(dd_here+1.0D0)
              prefac = 0.5D0 ** (dd_here-1.0D0) / GA
              prefac = prefac * origin_val
              snu = DSQRT(dd_here) * 2.0D0
              fl = INT(dd_here)
              rem = dd_here - fl
              N=fl
              
              C(i,j) = C(i,j) * snu
              CALL RKBESL(C(i,j),rem,fl+1,1,BK,N)
              C(i,j)=prefac * (C(i,j) ** dd_here) * BK(fl+1)
              
            endif
    2     enddo
        enddo
      endif     


      RETURN
      END


      SUBROUTINE matern(C,diff_degree,nx,ny,cmin,cmax,symm,BK,N)

cf2py intent(inplace) C
cf2py integer intent(in), optional :: cmin = 0
cf2py integer intent(in), optional :: cmax = -1
cf2py intent(hide) nx,ny,Bk
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(hide), depend(diff_degree):: N = floor(diff_degree)
cf2py double precision intent(in),check(diff_degree>0)::diff_degree
cf2py threadsafe

      DOUBLE PRECISION C(nx,ny)
      DOUBLE PRECISION diff_degree, rem
      DOUBLE PRECISION GA, prefac, snu
      INTEGER nx, ny, i, j, fl, N, cmin, cmax
      DOUBLE PRECISION BK(N+1), DGAMMA
      LOGICAL symm

!       print *,diff_degree,nx,ny,cmin,cmax,symm,N      
      if (cmax.EQ.-1) then
          cmax = ny
      end if
      

      
      if (diff_degree .GT. 10.0D0) then
        call gaussian(C,nx,ny,symm)
        return
      endif
      

      if (diff_degree .EQ. 1.0D0) then
          prefac = 1.0D0
      else
          GA = DGAMMA(diff_degree)
          prefac = 0.5D0 ** (diff_degree-1.0D0) / GA
      endif
      snu = DSQRT(diff_degree) * 2.0D0
      fl = DINT(diff_degree)
      rem = diff_degree - fl

      if (symm) then           
       
        do j=cmin+1,cmax
          C(j,j) = 1.0D0
          do i=1,j-1
            if (C(i,j) .EQ. 0.0D0) then
              C(i,j)=1.0D0
            else              
              C(i,j) = C(i,j) * snu
              CALL RKBESL(C(i,j),rem,fl+1,1,BK,N)
              C(i,j)=prefac*(C(i,j)**diff_degree)*BK(fl+1)
            endif
!           C(j,i)=C(i,j)
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
            if (C(i,j) .EQ. 0.0D0) then
              C(i,j)=1.0D0
            else
              C(i,j) = C(i,j) * snu
              CALL RKBESL(C(i,j),rem,fl+1,1,BK,N)
              C(i,j)=prefac * (C(i,j) ** diff_degree) * BK(fl+1)
            endif
          enddo
        enddo
      endif  


      RETURN
      END


            
      SUBROUTINE gaussian(C,nx,ny,cmin,cmax,symm)

cf2py intent(inplace) C
cf2py intent(hide) nx,ny
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe

      INTEGER nx,ny,i,j,cmin,cmax
      DOUBLE PRECISION C(nx,ny)
      LOGICAL symm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      if(symm) then

        do j=cmin+1,cmax
          C(j,j) = 1.0D0
          do i=0,j-1
            C(i,j) = dexp(-C(i,j)*C(i,j)) 
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
            C(i,j) = dexp(-C(i,j)*C(i,j)) 
          enddo
        enddo

      endif


      return
      END


      SUBROUTINE pow_exp(C,pow,nx,ny,cmin,cmax,symm)

cf2py intent(inplace) C
cf2py intent(hide) nx,ny
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe

      INTEGER nx,ny,i,j,cmin,cmax
      DOUBLE PRECISION C(nx,ny)
      DOUBLE PRECISION pow
      LOGICAL symm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if


      if(symm) then
          
        do j=cmin+1,cmax
          C(j,j)=1.0D0         
          do i=1,j-1
            C(i,j) = dexp(-dabs(C(i,j))**pow)
!             C(j,i) = C(i,j)
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
            C(i,j) = dexp(-dabs(C(i,j))**pow) 
          enddo
        enddo

      endif
      

      return
      END
      

      SUBROUTINE sphere(C,nx,ny,cmin,cmax,symm)

cf2py intent(inplace) C
cf2py intent(hide) nx,ny
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe

      INTEGER nx,ny,i,j,cmin,cmax
      DOUBLE PRECISION C(nx,ny), t
      LOGICAL symm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if


      if(symm) then
        do j=cmin+1,cmax
          C(j,j)=1.0D0         
          do i=1,j-1
            t = C(i,j)
            if (t .LT. 1.0D0) then
              C(i,j) = 1.0D0-1.5D0*t+0.5D0*t**3
            else
              C(i,j) = 0.0D0
            endif
!             C(j,i) = C(i,j)
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
            t = C(i,j)
            if (t .LT. 1.0D0) then
              C(i,j) = 1.0D0-1.5D0*t+0.5D0*t**3
            else
              C(i,j) = 0.0D0
            endif
          enddo
        enddo

      endif


      return
      END
            

      SUBROUTINE quadratic(C,phi,nx,ny,cmin,cmax,symm)

cf2py intent(inplace) C
cf2py double precision intent(in),check(phi>0)::phi
cf2py intent(hide) nx,ny
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe

      INTEGER nx,ny,i,j,cmin,cmax
      DOUBLE PRECISION C(nx,ny)
      DOUBLE PRECISION phi, t
      LOGICAL symm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      if(symm) then
          
        do j=cmin+1,cmax
          C(j,j)=1.0D0         
          do i=1,j-2
            t=C(i,j)**2
            C(i,j) = 1.0D0-t/(1.0D0+phi*t)
!             C(j,i) = C(i,j)
          enddo
        enddo

      else

        do j=cmin+1,cmax
          do i=1,nx
            t=C(i,j)**2
            C(i,j) = 1.0D0-t/(1.0D0+phi*t)
          enddo
        enddo

      endif


      return
      END


CS    REAL FUNCTION GAMMA(X)
      DOUBLE PRECISION FUNCTION DGAMMA(X)
C----------------------------------------------------------------------
C
C This routine calculates the GAMMA function for a real argument X.
C   Computation is based on an algorithm outlined in reference 1.
C   The program uses rational functions that approximate the GAMMA
C   function to at least 20 significant decimal digits.  Coefficients
C   for the approximation over the interval (1,2) are unpublished.
C   Those for the approximation for X .GE. 12 are from reference 2.
C   The accuracy achieved depends on the arithmetic system, the
C   compiler, the intrinsic functions, and proper selection of the
C   machine-dependent constants.
C
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C beta   - radix for the floating-point representation
C maxexp - the smallest positive power of beta that overflows
C XBIG   - the largest argument for which GAMMA(X) is representable
C          in the machine, i.e., the solution to the equation
C                  GAMMA(XBIG) = beta**maxexp
C XINF   - the largest machine representable floating-point number;
C          approximately beta**maxexp
C EPS    - the smallest positive floating-point number such that
C          1.0+EPS .GT. 1.0
C XMININ - the smallest positive floating-point number such that
C          1/XMININ is machine representable
C
C     Approximate values for some important machines are:
C
C                            beta       maxexp        XBIG
C
C CRAY-1         (S.P.)        2         8191        966.961
C Cyber 180/855
C   under NOS    (S.P.)        2         1070        177.803
C IEEE (IBM/XT,
C   SUN, etc.)   (S.P.)        2          128        35.040
C IEEE (IBM/XT,
C   SUN, etc.)   (D.P.)        2         1024        171.624
C IBM 3033       (D.P.)       16           63        57.574
C VAX D-Format   (D.P.)        2          127        34.844
C VAX G-Format   (D.P.)        2         1023        171.489
C
C                            XINF         EPS        XMININ
C
C CRAY-1         (S.P.)   5.45E+2465   7.11E-15    1.84E-2466
C Cyber 180/855
C   under NOS    (S.P.)   1.26E+322    3.55E-15    3.14E-294
C IEEE (IBM/XT,
C   SUN, etc.)   (S.P.)   3.40E+38     1.19E-7     1.18E-38
C IEEE (IBM/XT,
C   SUN, etc.)   (D.P.)   1.79D+308    2.22D-16    2.23D-308
C IBM 3033       (D.P.)   7.23D+75     2.22D-16    1.39D-76
C VAX D-Format   (D.P.)   1.70D+38     1.39D-17    5.88D-39
C VAX G-Format   (D.P.)   8.98D+307    1.11D-16    1.12D-308
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  The program returns the value XINF for singularities or
C     when overflow would occur.  The computation is believed
C     to be free of underflow and overflow.
C
C
C  Intrinsic functions required are:
C
C     INT, DBLE, EXP, LOG, REAL, SIN
C
C
C References: "An Overview of Software Development for Special
C              Functions", W. J. Cody, Lecture Notes in Mathematics,
C              506, Numerical Analysis Dundee, 1975, G. A. Watson
C              (ed.), Springer Verlag, Berlin, 1976.
C
C              Computer Approximations, Hart, Et. Al., Wiley and
C              sons, New York, 1968.
C
C  Latest modification: October 12, 1989
C
C  Authors: W. J. Cody and L. Stoltz
C           Applied Mathematics Division
C           Argonne National Laboratory
C           Argonne, IL 60439
C
C----------------------------------------------------------------------
      INTEGER I,N
      LOGICAL PARITY
CS    REAL 
      DOUBLE PRECISION 
     1    C,CONV,EPS,FACT,HALF,ONE,P,PI,Q,RES,SQRTPI,SUM,TWELVE,
     2    TWO,X,XBIG,XDEN,XINF,XMININ,XNUM,Y,Y1,YSQ,Z,ZERO
      DIMENSION C(7),P(8),Q(8)
C----------------------------------------------------------------------
C  Mathematical constants
C----------------------------------------------------------------------
CS    DATA ONE,HALF,TWELVE,TWO,ZERO/1.0E0,0.5E0,12.0E0,2.0E0,0.0E0/,
CS   1     SQRTPI/0.9189385332046727417803297E0/,
CS   2     PI/3.1415926535897932384626434E0/
      DATA ONE,HALF,TWELVE,TWO,ZERO/1.0D0,0.5D0,12.0D0,2.0D0,0.0D0/,
     1     SQRTPI/0.9189385332046727417803297D0/,
     2     PI/3.1415926535897932384626434D0/
C----------------------------------------------------------------------
C  Machine dependent parameters
C----------------------------------------------------------------------
CS    DATA XBIG,XMININ,EPS/35.040E0,1.18E-38,1.19E-7/,
CS   1     XINF/3.4E38/
      DATA XBIG,XMININ,EPS/171.624D0,2.23D-308,2.22D-16/,
     1     XINF/1.79D308/
C----------------------------------------------------------------------
C  Numerator and denominator coefficients for rational minimax
C     approximation over (1,2).
C----------------------------------------------------------------------
CS    DATA P/-1.71618513886549492533811E+0,2.47656508055759199108314E+1,
CS   1       -3.79804256470945635097577E+2,6.29331155312818442661052E+2,
CS   2       8.66966202790413211295064E+2,-3.14512729688483675254357E+4,
CS   3       -3.61444134186911729807069E+4,6.64561438202405440627855E+4/
CS    DATA Q/-3.08402300119738975254353E+1,3.15350626979604161529144E+2,
CS   1      -1.01515636749021914166146E+3,-3.10777167157231109440444E+3,
CS   2        2.25381184209801510330112E+4,4.75584627752788110767815E+3,
CS   3      -1.34659959864969306392456E+5,-1.15132259675553483497211E+5/
      DATA P/-1.71618513886549492533811D+0,2.47656508055759199108314D+1,
     1       -3.79804256470945635097577D+2,6.29331155312818442661052D+2,
     2       8.66966202790413211295064D+2,-3.14512729688483675254357D+4,
     3       -3.61444134186911729807069D+4,6.64561438202405440627855D+4/
      DATA Q/-3.08402300119738975254353D+1,3.15350626979604161529144D+2,
     1      -1.01515636749021914166146D+3,-3.10777167157231109440444D+3,
     2        2.25381184209801510330112D+4,4.75584627752788110767815D+3,
     3      -1.34659959864969306392456D+5,-1.15132259675553483497211D+5/
C----------------------------------------------------------------------
C  Coefficients for minimax approximation over (12, INF).
C----------------------------------------------------------------------
CS    DATA C/-1.910444077728E-03,8.4171387781295E-04,
CS   1     -5.952379913043012E-04,7.93650793500350248E-04,
CS   2     -2.777777777777681622553E-03,8.333333333333333331554247E-02,
CS   3      5.7083835261E-03/
      DATA C/-1.910444077728D-03,8.4171387781295D-04,
     1     -5.952379913043012D-04,7.93650793500350248D-04,
     2     -2.777777777777681622553D-03,8.333333333333333331554247D-02,
     3      5.7083835261D-03/
C----------------------------------------------------------------------
C  Statement functions for conversion between integer and float
C----------------------------------------------------------------------
CS    CONV(I) = REAL(I)
      CONV(I) = DBLE(I)
      PARITY = .FALSE.
      FACT = ONE
      N = 0
      Y = X
      IF (Y .LE. ZERO) THEN
C----------------------------------------------------------------------
C  Argument is negative
C----------------------------------------------------------------------
            Y = -X
            Y1 = AINT(Y)
            RES = Y - Y1
            IF (RES .NE. ZERO) THEN
                  IF (Y1 .NE. AINT(Y1*HALF)*TWO) PARITY = .TRUE.
                  FACT = -PI / SIN(PI*RES)
                  Y = Y + ONE
               ELSE
                  RES = XINF
                  GO TO 900
            END IF
      END IF
C----------------------------------------------------------------------
C  Argument is positive
C----------------------------------------------------------------------
      IF (Y .LT. EPS) THEN
C----------------------------------------------------------------------
C  Argument .LT. EPS
C----------------------------------------------------------------------
            IF (Y .GE. XMININ) THEN
                  RES = ONE / Y
               ELSE
                  RES = XINF
                  GO TO 900
            END IF
         ELSE IF (Y .LT. TWELVE) THEN
            Y1 = Y
            IF (Y .LT. ONE) THEN
C----------------------------------------------------------------------
C  0.0 .LT. argument .LT. 1.0
C----------------------------------------------------------------------
                  Z = Y
                  Y = Y + ONE
               ELSE
C----------------------------------------------------------------------
C  1.0 .LT. argument .LT. 12.0, reduce argument if necessary
C----------------------------------------------------------------------
                  N = INT(Y) - 1
                  Y = Y - CONV(N)
                  Z = Y - ONE
            END IF
C----------------------------------------------------------------------
C  Evaluate approximation for 1.0 .LT. argument .LT. 2.0
C----------------------------------------------------------------------
            XNUM = ZERO
            XDEN = ONE
            DO 260 I = 1, 8
               XNUM = (XNUM + P(I)) * Z
               XDEN = XDEN * Z + Q(I)
  260       CONTINUE
            RES = XNUM / XDEN + ONE
            IF (Y1 .LT. Y) THEN
C----------------------------------------------------------------------
C  Adjust result for case  0.0 .LT. argument .LT. 1.0
C----------------------------------------------------------------------
                  RES = RES / Y1
               ELSE IF (Y1 .GT. Y) THEN
C----------------------------------------------------------------------
C  Adjust result for case  2.0 .LT. argument .LT. 12.0
C----------------------------------------------------------------------
                  DO 290 I = 1, N
                     RES = RES * Y
                     Y = Y + ONE
  290             CONTINUE
            END IF
         ELSE
C----------------------------------------------------------------------
C  Evaluate for argument .GE. 12.0,
C----------------------------------------------------------------------
            IF (Y .LE. XBIG) THEN
                  YSQ = Y * Y
                  SUM = C(7)
                  DO 350 I = 1, 6
                     SUM = SUM / YSQ + C(I)
  350             CONTINUE
                  SUM = SUM/Y - Y + SQRTPI
                  SUM = SUM + (Y-HALF)*LOG(Y)
                  RES = EXP(SUM)
               ELSE
                  RES = XINF
                  GO TO 900
            END IF
      END IF
C----------------------------------------------------------------------
C  Final adjustments and return
C----------------------------------------------------------------------
      IF (PARITY) RES = -RES
      IF (FACT .NE. ONE) RES = FACT / RES
CS900 GAMMA = RES
  900 DGAMMA = RES
      RETURN
C ---------- Last line of GAMMA ----------
      END



      SUBROUTINE RKBESL(X,ALPHA,NB,IZE,BK,NCALC)
C-------------------------------------------------------------------
C
C  This FORTRAN 77 routine calculates modified Bessel functions
C  of the second kind, K SUB(N+ALPHA) (X), for non-negative
C  argument X, and non-negative order N+ALPHA, with or without
C  exponential scaling.
C
C  Explanation of variables in the calling sequence
C
C  Description of output values ..
C
C X     - Working precision non-negative real argument for which
C         K's or exponentially scaled K's (K*EXP(X))
C         are to be calculated.  If K's are to be calculated,
C         X must not be greater than XMAX (see below).
C ALPHA - Working precision fractional part of order for which 
C         K's or exponentially scaled K's (K*EXP(X)) are
C         to be calculated.  0 .LE. ALPHA .LT. 1.0.
C NB    - Integer number of functions to be calculated, NB .GT. 0.
C         The first function calculated is of order ALPHA, and the 
C         last is of order (NB - 1 + ALPHA).
C IZE   - Integer type.  IZE = 1 if unscaled K's are to be calculated,
C         and 2 if exponentially scaled K's are to be calculated.
C BK    - Working precision output vector of length NB.  If the
C         routine terminates normally (NCALC=NB), the vector BK
C         contains the functions K(ALPHA,X), ... , K(NB-1+ALPHA,X),
C         or the corresponding exponentially scaled functions.
C         If (0 .LT. NCALC .LT. NB), BK(I) contains correct function
C         values for I .LE. NCALC, and contains the ratios
C         K(ALPHA+I-1,X)/K(ALPHA+I-2,X) for the rest of the array.
C NCALC - Integer output variable indicating possible errors.
C         Before using the vector BK, the user should check that 
C         NCALC=NB, i.e., all orders have been calculated to
C         the desired accuracy.  See error returns below.
C
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C   beta   = Radix for the floating-point system
C   minexp = Smallest representable power of beta
C   maxexp = Smallest power of beta that overflows
C   EPS    = The smallest positive floating-point number such that 
C            1.0+EPS .GT. 1.0
C   XMAX   = Upper limit on the magnitude of X when IZE=1;  Solution 
C            to equation:
C               W(X) * (1-1/8X+9/128X**2) = beta**minexp
C            where  W(X) = EXP(-X)*SQRT(PI/2X)
C   SQXMIN = Square root of beta**minexp
C   XINF   = Largest positive machine number; approximately
C            beta**maxexp
C   XMIN   = Smallest positive machine number; approximately
C            beta**minexp
C
C
C     Approximate values for some important machines are:
C
C                          beta       minexp      maxexp      EPS
C
C  CRAY-1        (S.P.)      2        -8193        8191    7.11E-15
C  Cyber 180/185 
C    under NOS   (S.P.)      2         -975        1070    3.55E-15
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)      2         -126         128    1.19E-7
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)      2        -1022        1024    2.22D-16
C  IBM 3033      (D.P.)     16          -65          63    2.22D-16
C  VAX           (S.P.)      2         -128         127    5.96E-8
C  VAX D-Format  (D.P.)      2         -128         127    1.39D-17
C  VAX G-Format  (D.P.)      2        -1024        1023    1.11D-16
C
C
C                         SQXMIN       XINF        XMIN      XMAX
C
C CRAY-1        (S.P.)  6.77E-1234  5.45E+2465  4.59E-2467 5674.858
C Cyber 180/855
C   under NOS   (S.P.)  1.77E-147   1.26E+322   3.14E-294   672.788
C IEEE (IBM/XT,
C   SUN, etc.)  (S.P.)  1.08E-19    3.40E+38    1.18E-38     85.337
C IEEE (IBM/XT,
C   SUN, etc.)  (D.P.)  1.49D-154   1.79D+308   2.23D-308   705.342
C IBM 3033      (D.P.)  7.35D-40    7.23D+75    5.40D-79    177.852
C VAX           (S.P.)  5.42E-20    1.70E+38    2.94E-39     86.715
C VAX D-Format  (D.P.)  5.42D-20    1.70D+38    2.94D-39     86.715
C VAX G-Format  (D.P.)  7.46D-155   8.98D+307   5.57D-309   706.728
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  In case of an error, NCALC .NE. NB, and not all K's are
C  calculated to the desired accuracy.
C
C  NCALC .LT. -1:  An argument is out of range. For example,
C       NB .LE. 0, IZE is not 1 or 2, or IZE=1 and ABS(X) .GE.
C       XMAX.  In this case, the B-vector is not calculated,
C       and NCALC is set to MIN0(NB,0)-2  so that NCALC .NE. NB.
C  NCALC = -1:  Either  K(ALPHA,X) .GE. XINF  or
C       K(ALPHA+NB-1,X)/K(ALPHA+NB-2,X) .GE. XINF.  In this case,
C       the B-vector is not calculated.  Note that again 
C       NCALC .NE. NB.
C
C  0 .LT. NCALC .LT. NB: Not all requested function values could
C       be calculated accurately.  BK(I) contains correct function
C       values for I .LE. NCALC, and contains the ratios
C       K(ALPHA+I-1,X)/K(ALPHA+I-2,X) for the rest of the array.
C
C
C Intrinsic functions required are:
C
C     ABS, AINT, EXP, INT, LOG, MAX, MIN, SINH, SQRT
C
C
C Acknowledgement
C
C  This program is based on a program written by J. B. Campbell
C  (2) that computes values of the Bessel functions K of real
C  argument and real order.  Modifications include the addition
C  of non-scaled functions, parameterization of machine
C  dependencies, and the use of more accurate approximations
C  for SINH and SIN.
C
C References: "On Temme's Algorithm for the Modified Bessel
C              Functions of the Third Kind," Campbell, J. B.,
C              TOMS 6(4), Dec. 1980, pp. 581-586.
C
C             "A FORTRAN IV Subroutine for the Modified Bessel
C              Functions of the Third Kind of Real Order and Real
C              Argument," Campbell, J. B., Report NRC/ERB-925,
C              National Research Council, Canada.
C
C  Latest modification: May 30, 1989
C
C  Modified by: W. J. Cody and L. Stoltz
C               Applied Mathematics Division
C               Argonne National Laboratory
C               Argonne, IL  60439
C
C-------------------------------------------------------------------
      INTEGER I,IEND,ITEMP,IZE,J,K,M,MPLUS1,NB,NCALC
CS    REAL
      DOUBLE PRECISION  
     1    A,ALPHA,BLPHA,BK,BK1,BK2,C,D,DM,D1,D2,D3,ENU,EPS,ESTF,ESTM,
     2    EX,FOUR,F0,F1,F2,HALF,ONE,P,P0,Q,Q0,R,RATIO,S,SQXMIN,T,TINYX,
     3    TWO,TWONU,TWOX,T1,T2,WMINF,X,XINF,XMAX,XMIN,X2BY4,ZERO
      DIMENSION BK(1),P(8),Q(7),R(5),S(4),T(6),ESTM(6),ESTF(7)
C---------------------------------------------------------------------
C  Mathematical constants
C    A = LOG(2.D0) - Euler's constant
C    D = SQRT(2.D0/PI)
C---------------------------------------------------------------------
CS    DATA HALF,ONE,TWO,ZERO/0.5E0,1.0E0,2.0E0,0.0E0/
CS    DATA FOUR,TINYX/4.0E0,1.0E-10/
CS    DATA A/ 0.11593151565841244881E0/,D/0.797884560802865364E0/
      DATA HALF,ONE,TWO,ZERO/0.5D0,1.0D0,2.0D0,0.0D0/
      DATA FOUR,TINYX/4.0D0,1.0D-10/
      DATA A/ 0.11593151565841244881D0/,D/0.797884560802865364D0/
C---------------------------------------------------------------------
C  Machine dependent parameters
C---------------------------------------------------------------------
CS    DATA EPS/1.19E-7/,SQXMIN/1.08E-19/,XINF/3.40E+38/
CS    DATA XMIN/1.18E-38/,XMAX/85.337E0/
      DATA EPS/2.22D-16/,SQXMIN/1.49D-154/,XINF/1.79D+308/
      DATA XMIN/2.23D-308/,XMAX/705.342D0/
C---------------------------------------------------------------------
C  P, Q - Approximation for LOG(GAMMA(1+ALPHA))/ALPHA
C                                         + Euler's constant
C         Coefficients converted from hex to decimal and modified
C         by W. J. Cody, 2/26/82
C  R, S - Approximation for (1-ALPHA*PI/SIN(ALPHA*PI))/(2.D0*ALPHA)
C  T    - Approximation for SINH(Y)/Y
C---------------------------------------------------------------------
CS    DATA P/ 0.805629875690432845E00,    0.204045500205365151E02,
CS   1        0.157705605106676174E03,    0.536671116469207504E03,
CS   2        0.900382759291288778E03,    0.730923886650660393E03,
CS   3        0.229299301509425145E03,    0.822467033424113231E00/
CS    DATA Q/ 0.294601986247850434E02,    0.277577868510221208E03,
CS   1        0.120670325591027438E04,    0.276291444159791519E04,
CS   2        0.344374050506564618E04,    0.221063190113378647E04,
CS   3        0.572267338359892221E03/
CS    DATA R/-0.48672575865218401848E+0,  0.13079485869097804016E+2,
CS   1       -0.10196490580880537526E+3,  0.34765409106507813131E+3,
CS   2        0.34958981245219347820E-3/
CS    DATA S/-0.25579105509976461286E+2,  0.21257260432226544008E+3,
CS   1       -0.61069018684944109624E+3,  0.42269668805777760407E+3/
CS    DATA T/ 0.16125990452916363814E-9, 0.25051878502858255354E-7,
CS   1        0.27557319615147964774E-5, 0.19841269840928373686E-3,
CS   2        0.83333333333334751799E-2, 0.16666666666666666446E+0/
CS    DATA ESTM/5.20583E1, 5.7607E0, 2.7782E0, 1.44303E1, 1.853004E2,
CS   1          9.3715E0/
CS    DATA ESTF/4.18341E1, 7.1075E0, 6.4306E0, 4.25110E1, 1.35633E0,
CS   1          8.45096E1, 2.0E1/
      DATA P/ 0.805629875690432845D00,    0.204045500205365151D02,
     1        0.157705605106676174D03,    0.536671116469207504D03,
     2        0.900382759291288778D03,    0.730923886650660393D03,
     3        0.229299301509425145D03,    0.822467033424113231D00/
      DATA Q/ 0.294601986247850434D02,    0.277577868510221208D03,
     1        0.120670325591027438D04,    0.276291444159791519D04,
     2        0.344374050506564618D04,    0.221063190113378647D04,
     3        0.572267338359892221D03/
      DATA R/-0.48672575865218401848D+0,  0.13079485869097804016D+2,
     1       -0.10196490580880537526D+3,  0.34765409106507813131D+3,
     2        0.34958981245219347820D-3/
      DATA S/-0.25579105509976461286D+2,  0.21257260432226544008D+3,
     1       -0.61069018684944109624D+3,  0.42269668805777760407D+3/
      DATA T/ 0.16125990452916363814D-9, 0.25051878502858255354D-7,
     1        0.27557319615147964774D-5, 0.19841269840928373686D-3,
     2        0.83333333333334751799D-2, 0.16666666666666666446D+0/
      DATA ESTM/5.20583D1, 5.7607D0, 2.7782D0, 1.44303D1, 1.853004D2,
     1          9.3715D0/
      DATA ESTF/4.18341D1, 7.1075D0, 6.4306D0, 4.25110D1, 1.35633D0,
     1          8.45096D1, 2.0D1/
C---------------------------------------------------------------------
      EX = X
      ENU = ALPHA
      NCALC = MIN(NB,0)-2
      IF ((NB .GT. 0) .AND. ((ENU .GE. ZERO) .AND. (ENU .LT. ONE))
     1     .AND. ((IZE .GE. 1) .AND. (IZE .LE. 2)) .AND.
     2     ((IZE .NE. 1) .OR. (EX .LE. XMAX)) .AND.
     3     (EX .GT. ZERO))  THEN
            K = 0
            IF (ENU .LT. SQXMIN) ENU = ZERO
            IF (ENU .GT. HALF) THEN
                  K = 1
                  ENU = ENU - ONE
            END IF
            TWONU = ENU+ENU
            IEND = NB+K-1
            C = ENU*ENU
            D3 = -C
            IF (EX .LE. ONE) THEN
C---------------------------------------------------------------------
C  Calculation of P0 = GAMMA(1+ALPHA) * (2/X)**ALPHA
C                 Q0 = GAMMA(1-ALPHA) * (X/2)**ALPHA
C---------------------------------------------------------------------
                  D1 = ZERO
                  D2 = P(1)
                  T1 = ONE
                  T2 = Q(1)
                  DO 10 I = 2,7,2
                     D1 = C*D1+P(I)
                     D2 = C*D2+P(I+1)
                     T1 = C*T1+Q(I)
                     T2 = C*T2+Q(I+1)
   10             CONTINUE
                  D1 = ENU*D1
                  T1 = ENU*T1
                  F1 = LOG(EX)
                  F0 = A+ENU*(P(8)-ENU*(D1+D2)/(T1+T2))-F1
                  Q0 = EXP(-ENU*(A-ENU*(P(8)+ENU*(D1-D2)/(T1-T2))-F1))
                  F1 = ENU*F0
                  P0 = EXP(F1)
C---------------------------------------------------------------------
C  Calculation of F0 = 
C---------------------------------------------------------------------
                  D1 = R(5)
                  T1 = ONE
                  DO 20 I = 1,4
                     D1 = C*D1+R(I)
                     T1 = C*T1+S(I)
   20             CONTINUE
                  IF (ABS(F1) .LE. HALF) THEN
                        F1 = F1*F1
                        D2 = ZERO
                        DO 30 I = 1,6
                           D2 = F1*D2+T(I)
   30                   CONTINUE
                        D2 = F0+F0*F1*D2
                     ELSE
                        D2 = SINH(F1)/ENU
                  END IF
                  F0 = D2-ENU*D1/(T1*P0)
                  IF (EX .LE. TINYX) THEN
C--------------------------------------------------------------------
C  X.LE.1.0E-10
C  Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X)
C--------------------------------------------------------------------
                        BK(1) = F0+EX*F0
                        IF (IZE .EQ. 1) BK(1) = BK(1)-EX*BK(1)
                        RATIO = P0/F0
                        C = EX*XINF
                        IF (K .NE. 0) THEN
C--------------------------------------------------------------------
C  Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X),
C  ALPHA .GE. 1/2
C--------------------------------------------------------------------
                              NCALC = -1
                              IF (BK(1) .GE. C/RATIO) GO TO 500
                              BK(1) = RATIO*BK(1)/EX
                              TWONU = TWONU+TWO
                              RATIO = TWONU
                        END IF
                        NCALC = 1
                        IF (NB .EQ. 1) GO TO 500
C--------------------------------------------------------------------
C  Calculate  K(ALPHA+L,X)/K(ALPHA+L-1,X),  L  =  1, 2, ... , NB-1
C--------------------------------------------------------------------
                        NCALC = -1
                        DO 80 I = 2,NB
                           IF (RATIO .GE. C) GO TO 500
                           BK(I) = RATIO/EX
                           TWONU = TWONU+TWO
                           RATIO = TWONU
   80                   CONTINUE
                        NCALC = 1
                        GO TO 420
                     ELSE
C--------------------------------------------------------------------
C  1.0E-10 .LT. X .LE. 1.0
C--------------------------------------------------------------------
                        C = ONE
                        X2BY4 = EX*EX/FOUR
                        P0 = HALF*P0
                        Q0 = HALF*Q0
                        D1 = -ONE
                        D2 = ZERO
                        BK1 = ZERO
                        BK2 = ZERO
                        F1 = F0
                        F2 = P0
  100                   D1 = D1+TWO
                        D2 = D2+ONE
                        D3 = D1+D3
                        C = X2BY4*C/D2
                        F0 = (D2*F0+P0+Q0)/D3
                        P0 = P0/(D2-ENU)
                        Q0 = Q0/(D2+ENU)
                        T1 = C*F0
                        T2 = C*(P0-D2*F0)
                        BK1 = BK1+T1
                        BK2 = BK2+T2
                        IF ((ABS(T1/(F1+BK1)) .GT. EPS) .OR.
     1                     (ABS(T2/(F2+BK2)) .GT. EPS))  GO TO 100
                        BK1 = F1+BK1
                        BK2 = TWO*(F2+BK2)/EX
                        IF (IZE .EQ. 2) THEN
                              D1 = EXP(EX)
                              BK1 = BK1*D1
                              BK2 = BK2*D1
                        END IF
                        WMINF = ESTF(1)*EX+ESTF(2)
                  END IF
               ELSE IF (EPS*EX .GT. ONE) THEN
C--------------------------------------------------------------------
C  X .GT. ONE/EPS
C--------------------------------------------------------------------
                  NCALC = NB
                  BK1 = ONE / (D*SQRT(EX))
                  DO 110 I = 1, NB
                     BK(I) = BK1
  110             CONTINUE
                  GO TO 500
               ELSE
C--------------------------------------------------------------------
C  X .GT. 1.0
C--------------------------------------------------------------------
                  TWOX = EX+EX
                  BLPHA = ZERO
                  RATIO = ZERO
                  IF (EX .LE. FOUR) THEN
C--------------------------------------------------------------------
C  Calculation of K(ALPHA+1,X)/K(ALPHA,X),  1.0 .LE. X .LE. 4.0
C--------------------------------------------------------------------
                        D2 = AINT(ESTM(1)/EX+ESTM(2))
                        M = INT(D2)
                        D1 = D2+D2
                        D2 = D2-HALF
                        D2 = D2*D2
                        DO 120 I = 2,M
                           D1 = D1-TWO
                           D2 = D2-D1
                           RATIO = (D3+D2)/(TWOX+D1-RATIO)
  120                   CONTINUE
C--------------------------------------------------------------------
C  Calculation of I(|ALPHA|,X) and I(|ALPHA|+1,X) by backward
C    recurrence and K(ALPHA,X) from the wronskian
C--------------------------------------------------------------------
                        D2 = AINT(ESTM(3)*EX+ESTM(4))
                        M = INT(D2)
                        C = ABS(ENU)
                        D3 = C+C
                        D1 = D3-ONE
                        F1 = XMIN
                        F0 = (TWO*(C+D2)/EX+HALF*EX/(C+D2+ONE))*XMIN
                        DO 130 I = 3,M
                           D2 = D2-ONE
                           F2 = (D3+D2+D2)*F0
                           BLPHA = (ONE+D1/D2)*(F2+BLPHA)
                           F2 = F2/EX+F1
                           F1 = F0
                           F0 = F2
  130                   CONTINUE
                        F1 = (D3+TWO)*F0/EX+F1
                        D1 = ZERO
                        T1 = ONE
                        DO 140 I = 1,7
                           D1 = C*D1+P(I)
                           T1 = C*T1+Q(I)
  140                   CONTINUE
                        P0 = EXP(C*(A+C*(P(8)-C*D1/T1)-LOG(EX)))/EX
                        F2 = (C+HALF-RATIO)*F1/EX
                        BK1 = P0+(D3*F0-F2+F0+BLPHA)/(F2+F1+F0)*P0
                        IF (IZE .EQ. 1) BK1 = BK1*EXP(-EX)
                        WMINF = ESTF(3)*EX+ESTF(4)
                     ELSE
C--------------------------------------------------------------------
C  Calculation of K(ALPHA,X) and K(ALPHA+1,X)/K(ALPHA,X), by backward
C  recurrence, for  X .GT. 4.0
C--------------------------------------------------------------------
                        DM = AINT(ESTM(5)/EX+ESTM(6))
                        M = INT(DM)
                        D2 = DM-HALF
                        D2 = D2*D2
                        D1 = DM+DM
                        DO 160 I = 2,M
                           DM = DM-ONE
                           D1 = D1-TWO
                           D2 = D2-D1
                           RATIO = (D3+D2)/(TWOX+D1-RATIO)
                           BLPHA = (RATIO+RATIO*BLPHA)/DM
  160                   CONTINUE
                        BK1 = ONE/((D+D*BLPHA)*SQRT(EX))
                        IF (IZE .EQ. 1) BK1 = BK1*EXP(-EX)
                        WMINF = ESTF(5)*(EX-ABS(EX-ESTF(7)))+ESTF(6)
                  END IF
C--------------------------------------------------------------------
C  Calculation of K(ALPHA+1,X) from K(ALPHA,X) and
C    K(ALPHA+1,X)/K(ALPHA,X)
C--------------------------------------------------------------------
                  BK2 = BK1+BK1*(ENU+HALF-RATIO)/EX
            END IF
C--------------------------------------------------------------------
C  Calculation of 'NCALC', K(ALPHA+I,X), I  =  0, 1, ... , NCALC-1,
C  K(ALPHA+I,X)/K(ALPHA+I-1,X), I  =  NCALC, NCALC+1, ... , NB-1
C--------------------------------------------------------------------
            NCALC = NB
            BK(1) = BK1
            IF (IEND .EQ. 0) GO TO 500
            J = 2-K
            IF (J .GT. 0) BK(J) = BK2
            IF (IEND .EQ. 1) GO TO 500
            M = MIN(INT(WMINF-ENU),IEND)
            DO 190 I = 2,M
               T1 = BK1
               BK1 = BK2
               TWONU = TWONU+TWO
               IF (EX .LT. ONE) THEN
                     IF (BK1 .GE. (XINF/TWONU)*EX) GO TO 195
                     GO TO 187
                  ELSE 
                     IF (BK1/EX .GE. XINF/TWONU) GO TO 195
               END IF
  187          CONTINUE
               BK2 = TWONU/EX*BK1+T1
               ITEMP = I
               J = J+1
               IF (J .GT. 0) BK(J) = BK2
  190       CONTINUE
  195       M = ITEMP
            IF (M .EQ. IEND) GO TO 500
            RATIO = BK2/BK1
            MPLUS1 = M+1
            NCALC = -1
            DO 410 I = MPLUS1,IEND
               TWONU = TWONU+TWO
               RATIO = TWONU/EX+ONE/RATIO
               J = J+1
               IF (J .GT. 1) THEN
                     BK(J) = RATIO
                  ELSE
                     IF (BK2 .GE. XINF/RATIO) GO TO 500
                     BK2 = RATIO*BK2
               END IF
  410       CONTINUE
            NCALC = MAX(MPLUS1-K,1)
            IF (NCALC .EQ. 1) BK(1) = BK2
            IF (NB .EQ. 1) GO TO 500
  420       J = NCALC+1
            DO 430 I = J,NB
               IF (BK(NCALC) .GE. XINF/BK(I)) GO TO 500
               BK(I) = BK(NCALC)*BK(I)
               NCALC = I
  430       CONTINUE
      END IF
  500 RETURN
C---------- Last line of RKBESL ----------
      END
