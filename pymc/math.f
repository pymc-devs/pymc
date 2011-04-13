      DOUBLE PRECISION FUNCTION PPND16 (P, IFAULT)
C
C      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
C
C      Produces the normal deviate Z corresponding to a given lower
C      tail area of P; Z is accurate to about 1 part in 10**16.
C
C      The hash sums below are the sums of the mantissas of the
C      coefficients.   They are included for use in checking
C      transcription.
C
      INTEGER IFAULT
      DOUBLE PRECISION ZERO, ONE, HALF, SPLIT1, SPLIT2, CONST1,
     *           CONST2, A0, A1, A2, A3, A4, A5, A6, A7, 
     *           B1, B2, B3, B4, B5, B6, B7,
     *           C0, C1, C2, C3, C4, C5, C6, C7,
     *           D1, D2, D3, D4, D5, D6, D7, 
     *           E0, E1, E2, E3, E4, E5, E6, E7, 
     *           F1, F2, F3, F4, F5, F6, F7, P, Q, R
      PARAMETER (ZERO = 0.D0, ONE = 1.D0, HALF = 0.5D0,
     *            SPLIT1 = 0.425D0, SPLIT2 = 5.D0,
     *            CONST1 = 0.180625D0, CONST2 = 1.6D0)
C
C      Coefficients for P close to 0.5
C
      PARAMETER (A0 = 3.3871328727963666080D0,
     *           A1 = 1.3314166789178437745D+2,
     *           A2 = 1.9715909503065514427D+3,
     *           A3 = 1.3731693765509461125D+4,
     *           A4 = 4.5921953931549871457D+4,
     *           A5 = 6.7265770927008700853D+4,
     *           A6 = 3.3430575583588128105D+4,
     *           A7 = 2.5090809287301226727D+3,
     *           B1 = 4.2313330701600911252D+1,
     *           B2 = 6.8718700749205790830D+2,
     *           B3 = 5.3941960214247511077D+3,
     *           B4 = 2.1213794301586595867D+4,
     *           B5 = 3.9307895800092710610D+4,
     *           B6 = 2.8729085735721942674D+4,
     *           B7 = 5.2264952788528545610D+3)
C      HASH SUM AB    55.88319 28806 14901 4439
C
C      Coefficients for P not close to 0, 0.5 or 1.
C
      PARAMETER (C0 = 1.42343711074968357734D0,
     *          C1 = 4.63033784615654529590D0,
     *          C2 = 5.76949722146069140550D0,
     *          C3 = 3.64784832476320460504D0,
     *          C4 = 1.27045825245236838258D0,
     *          C5 = 2.41780725177450611770D-1,
     *          C6=2.27238449892691845833D-2,
     *          C7 = 7.74545014278341407640D-4,
     *          D1 = 2.05319162663775882187D0,
     *          D2 = 1.67638483018380384940D0,
     *          D3 = 6.89767334985100004550D-1,
     *          D4 = 1.48103976427480074590D-1,
     *          D5 = 1.51986665636164571966D-2,
     *          D6 = 5.47593808499534494600D-4,
     *          D7 = 1.05075007164441684324D-9)
C      HASH SUM       49.33206 50330 16102 89036
C
C      Coefficients for P near 0 or 1.
C
      PARAMETER (E0 = 6.65790464350110377720D0,
     *          E1 = 5.46378491116411436990D0,
     *          E2 = 1.78482653991729133580D0,
     *          E3 = 2.96560571828504891230D-1,
     *          E4 = 2.65321895265761230930D-2,
     *          E5 = 1.24266094738807843860D-3,
     *          E6 = 2.71155556874348757815D-5,
     *          E7 = 2.01033439929228813265D-7,
     *          F1 = 5.99832206555887937690D-1,
     *          F2 = 1.36929880922735805310D-1,
     *          F3 = 1.48753612908506148525D-2,
     *          F4 = 7.86869131145613259100D-4,
     *          F5 = 1.84631831751005468180D-5,
     *          F6 = 1.42151175831644588870D-7,
     *          F7 = 2.04426310338993978564D-15)
C      HASH SUM EF    47.52583 31754 92896 71629
C
      IFAULT = 0
      Q = P - HALF
      IF (ABS(Q) .LE. SPLIT1) THEN
        R = CONST1 - Q * Q
        PPND16 = Q * (((((((A7 * R + A6) * R + A5) * R + A4) * R + A3)
     *             * R + A2) * R + A1) * R + A0) /
     *             (((((((B7 * R + B6) * R + B5) * R + B4) * R + B3)
     *             * R + B2) * R + B1) * R + ONE)
        RETURN
      ELSE
        IF (Q .LT. ZERO) THEN
          R = P
        ELSE
          R = ONE - P
        END IF
        IF (R .LE. ZERO) THEN
          IFAULT = 1
          PPND16 = ZERO
          RETURN
        END IF
        R = SQRT(-LOG(R))
        IF (R .LE. SPLIT2) THEN
          R = R - CONST2
          PPND16 = (((((((C7 * R + C6) * R + C5) * R + C4) * R + C3)
     *              * R + C2) * R + C1) * R + C0) /
     *              (((((((D7 * R + D6) * R + D5) * R + D4) * R + D3)
     *              * R + D2) * R + D1) * R + ONE)
        ELSE
          R = R - SPLIT2
          PPND16 = (((((((E7 * R + E6) * R + E5) * R + E4) * R + E3)
     *             * R + E2) * R + E1) * R + E0) /
     *             (((((((F7 * R + F6) * R + F5) * R + F4) * R + F3)
     *             * R + F2) * R + F1) * R + ONE)
        END IF
        IF (Q .LT. ZERO) PPND16 = - PPND16
        RETURN
      END IF
      END


      SUBROUTINE CALERF(ARG,RESULT,JINT)
C------------------------------------------------------------------
C
C This packet evaluates  erf(x),  erfc(x),  and  exp(x*x)*erfc(x)
C   for a real argument  x.  It contains three FUNCTION type
C   subprograms: ERF, ERFC, and ERFCX (or DERF, DERFC, and DERFCX),
C   and one SUBROUTINE type subprogram, CALERF.  The calling
C   statements for the primary entries are:
C
C                   Y=ERF(X)     (or   Y=DERF(X)),
C
C                   Y=ERFC(X)    (or   Y=DERFC(X)),
C   and
C                   Y=ERFCX(X)   (or   Y=DERFCX(X)).
C
C   The routine  CALERF  is intended for internal packet use only,
C   all computations within the packet being concentrated in this
C   routine.  The function subprograms invoke  CALERF  with the
C   statement
C
C          CALL CALERF(ARG,RESULT,JINT)
C
C   where the parameter usage is as follows
C
C      Function                     Parameters for CALERF
C       call              ARG                  Result          JINT
C
C     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0
C     ERFC(ARG)     ABS(ARG) .LT. XBIG        ERFC(ARG)         1
C     ERFCX(ARG)    XNEG .LT. ARG .LT. XMAX   ERFCX(ARG)        2
C
C   The main computation evaluates near-minimax approximations
C   from "Rational Chebyshev approximations for the error function"
C   by W. J. Cody, Math. Comp., 1969, PP. 631-638.  This
C   transportable program uses rational functions that theoretically
C   approximate  erf(x)  and  erfc(x)  to at least 18 significant
C   decimal digits.  The accuracy achieved depends on the arithmetic
C   system, the compiler, the intrinsic functions, and proper
C   selection of the machine-dependent constants.
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C   XMIN   = the smallest positive floating-point number.
C   XINF   = the largest positive finite floating-point number.
C   XNEG   = the largest negative argument acceptable to ERFCX;
C            the negative of the solution to the equation
C            2*exp(x*x) = XINF.
C   XSMALL = argument below which erf(x) may be represented by
C            2*x/sqrt(pi)  and above which  x*x  will not underflow.
C            A conservative value is the largest machine number X
C            such that   1.0 + X = 1.0   to machine precision.
C   XBIG   = largest argument acceptable to ERFC;  solution to
C            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where
C            W(x) = exp(-x*x)/[x*sqrt(pi)].
C   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to
C            machine precision.  A conservative value is
C            1/[2*sqrt(XSMALL)]
C   XMAX   = largest acceptable argument to ERFCX; the minimum
C            of XINF and 1/[sqrt(pi)*XMIN].
C
C   Approximate values for some important machines are:
C
C                          XMIN       XINF        XNEG     XSMALL
C
C    C 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15
C  CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16
C  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17
C  UNIVAC 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18
C  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17
C  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16
C
C
C                          XBIG       XHUGE       XMAX
C
C    C 7600      (S.P.)  25.922      8.39E+6     1.80X+293
C  CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)   9.194      2.90E+3     4.79E+37
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  26.543      6.71D+7     2.53D+307
C  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75
C  UNIVAC 1108   (D.P.)  26.582      5.37D+8     8.98D+307
C  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38
C  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  The program returns  ERFC = 0      for  ARG .GE. XBIG;
C
C                       ERFCX = XINF  for  ARG .LT. XNEG;
C      and
C                       ERFCX = 0     for  ARG .GE. XMAX.
C
C
C Intrinsic functions required are:
C
C     ABS, AINT, EXP
C
C
C  Author: W. J. Cody
C          Mathematics and Computer Science Division
C          Argonne National Laboratory
C          Argonne, IL 60439
C
C  Latest modification: March 19, 1990
C
C------------------------------------------------------------------
      INTEGER I,JINT
CS    REAL
      DOUBLE PRECISION
     1     A,ARG,B,C,D,DEL,FOUR,HALF,P,ONE,Q,RESULT,SIXTEN,SQRPI,
     2     TWO,THRESH,X,XBIG,XDEN,XHUGE,XINF,XMAX,XNEG,XNUM,XSMALL,
     3     Y,YSQ,ZERO
      DIMENSION A(5),B(4),C(9),D(8),P(6),Q(5)
C------------------------------------------------------------------
C  Mathematical constants
C------------------------------------------------------------------
CS    DATA FOUR,ONE,HALF,TWO,ZERO/4.0E0,1.0E0,0.5E0,2.0E0,0.0E0/,
CS   1     SQRPI/5.6418958354775628695E-1/,THRESH/0.46875E0/,
CS   2     SIXTEN/16.0E0/
      DATA FOUR,ONE,HALF,TWO,ZERO/4.0D0,1.0D0,0.5D0,2.0D0,0.0D0/,
     1     SQRPI/5.6418958354775628695D-1/,THRESH/0.46875D0/,
     2     SIXTEN/16.0D0/
C------------------------------------------------------------------
C  Machine-dependent constants
C------------------------------------------------------------------
CS    DATA XINF,XNEG,XSMALL/3.40E+38,-9.382E0,5.96E-8/,
CS   1     XBIG,XHUGE,XMAX/9.194E0,2.90E3,4.79E37/
      DATA XINF,XNEG,XSMALL/1.79D308,-26.628D0,1.11D-16/,
     1     XBIG,XHUGE,XMAX/26.543D0,6.71D7,2.53D307/
C------------------------------------------------------------------
C  Coefficients for approximation to  erf  in first interval
C------------------------------------------------------------------
CS    DATA A/3.16112374387056560E00,1.13864154151050156E02,
CS   1       3.77485237685302021E02,3.20937758913846947E03,
CS   2       1.85777706184603153E-1/
CS    DATA B/2.36012909523441209E01,2.44024637934444173E02,
CS   1       1.28261652607737228E03,2.84423683343917062E03/
      DATA A/3.16112374387056560D00,1.13864154151050156D02,
     1       3.77485237685302021D02,3.20937758913846947D03,
     2       1.85777706184603153D-1/
      DATA B/2.36012909523441209D01,2.44024637934444173D02,
     1       1.28261652607737228D03,2.84423683343917062D03/
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in second interval
C------------------------------------------------------------------
CS    DATA C/5.64188496988670089E-1,8.88314979438837594E0,
CS   1       6.61191906371416295E01,2.98635138197400131E02,
CS   2       8.81952221241769090E02,1.71204761263407058E03,
CS   3       2.05107837782607147E03,1.23033935479799725E03,
CS   4       2.15311535474403846E-8/
CS    DATA D/1.57449261107098347E01,1.17693950891312499E02,
CS   1       5.37181101862009858E02,1.62138957456669019E03,
CS   2       3.29079923573345963E03,4.36261909014324716E03,
CS   3       3.43936767414372164E03,1.23033935480374942E03/
      DATA C/5.64188496988670089D-1,8.88314979438837594D0,
     1       6.61191906371416295D01,2.98635138197400131D02,
     2       8.81952221241769090D02,1.71204761263407058D03,
     3       2.05107837782607147D03,1.23033935479799725D03,
     4       2.15311535474403846D-8/
      DATA D/1.57449261107098347D01,1.17693950891312499D02,
     1       5.37181101862009858D02,1.62138957456669019D03,
     2       3.29079923573345963D03,4.36261909014324716D03,
     3       3.43936767414372164D03,1.23033935480374942D03/
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in third interval
C------------------------------------------------------------------
CS    DATA P/3.05326634961232344E-1,3.60344899949804439E-1,
CS   1       1.25781726111229246E-1,1.60837851487422766E-2,
CS   2       6.58749161529837803E-4,1.63153871373020978E-2/
CS    DATA Q/2.56852019228982242E00,1.87295284992346047E00,
CS   1       5.27905102951428412E-1,6.05183413124413191E-2,
CS   2       2.33520497626869185E-3/
      DATA P/3.05326634961232344D-1,3.60344899949804439D-1,
     1       1.25781726111229246D-1,1.60837851487422766D-2,
     2       6.58749161529837803D-4,1.63153871373020978D-2/
      DATA Q/2.56852019228982242D00,1.87295284992346047D00,
     1       5.27905102951428412D-1,6.05183413124413191D-2,
     2       2.33520497626869185D-3/
C------------------------------------------------------------------
      X = ARG
      Y = ABS(X)
      IF (Y .LE. THRESH) THEN
C------------------------------------------------------------------
C  Evaluate  erf  for  |X| <= 0.46875
C------------------------------------------------------------------
            YSQ = ZERO
            IF (Y .GT. XSMALL) YSQ = Y * Y
            XNUM = A(5)*YSQ
            XDEN = YSQ
            DO 20 I = 1, 3
               XNUM = (XNUM + A(I)) * YSQ
               XDEN = (XDEN + B(I)) * YSQ
   20       CONTINUE
            RESULT = X * (XNUM + A(4)) / (XDEN + B(4))
            IF (JINT .NE. 0) RESULT = ONE - RESULT
            IF (JINT .EQ. 2) RESULT = EXP(YSQ) * RESULT
            GO TO 800
C------------------------------------------------------------------
C  Evaluate  erfc  for 0.46875 <= |X| <= 4.0
C------------------------------------------------------------------
         ELSE IF (Y .LE. FOUR) THEN
            XNUM = C(9)*Y
            XDEN = Y
            DO 120 I = 1, 7
               XNUM = (XNUM + C(I)) * Y
               XDEN = (XDEN + D(I)) * Y
  120       CONTINUE
            RESULT = (XNUM + C(8)) / (XDEN + D(8))
            IF (JINT .NE. 2) THEN
               YSQ = AINT(Y*SIXTEN)/SIXTEN
               DEL = (Y-YSQ)*(Y+YSQ)
               RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
            END IF
C------------------------------------------------------------------
C  Evaluate  erfc  for |X| > 4.0
C------------------------------------------------------------------
         ELSE
            RESULT = ZERO
            IF (Y .GE. XBIG) THEN
               IF ((JINT .NE. 2) .OR. (Y .GE. XMAX)) GO TO 300
               IF (Y .GE. XHUGE) THEN
                  RESULT = SQRPI / Y
                  GO TO 300
               END IF
            END IF
            YSQ = ONE / (Y * Y)
            XNUM = P(6)*YSQ
            XDEN = YSQ
            DO 240 I = 1, 4
               XNUM = (XNUM + P(I)) * YSQ
               XDEN = (XDEN + Q(I)) * YSQ
  240       CONTINUE
            RESULT = YSQ *(XNUM + P(5)) / (XDEN + Q(5))
            RESULT = (SQRPI -  RESULT) / Y
            IF (JINT .NE. 2) THEN
               YSQ = AINT(Y*SIXTEN)/SIXTEN
               DEL = (Y-YSQ)*(Y+YSQ)
               RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
            END IF
      END IF
C------------------------------------------------------------------
C  Fix up for negative argument, erf, etc.
C------------------------------------------------------------------
  300 IF (JINT .EQ. 0) THEN
            RESULT = (HALF - RESULT) + HALF
            IF (X .LT. ZERO) RESULT = -RESULT
         ELSE IF (JINT .EQ. 1) THEN
            IF (X .LT. ZERO) RESULT = TWO - RESULT
         ELSE
            IF (X .LT. ZERO) THEN
               IF (X .LT. XNEG) THEN
                     RESULT = XINF
                  ELSE
                     YSQ = AINT(X*SIXTEN)/SIXTEN
                     DEL = (X-YSQ)*(X+YSQ)
                     Y = EXP(YSQ*YSQ) * EXP(DEL)
                     RESULT = (Y+Y) - RESULT
               END IF
            END IF
      END IF
  800 RETURN
C---------- Last card of CALERF ----------
      END
CS    REAL FUNCTION ERF(X)
      DOUBLE PRECISION FUNCTION DERF(X)
C--------------------------------------------------------------------
C
C This subprogram computes approximate values for erf(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, January 8, 1985
C
C--------------------------------------------------------------------
      INTEGER JINT
CS    REAL             X, RESULT
      DOUBLE PRECISION X, RESULT
C------------------------------------------------------------------
      JINT = 0
      CALL CALERF(X,RESULT,JINT)
CS    ERF = RESULT
      DERF = RESULT
      RETURN
C---------- Last card of DERF ----------
      END
CS    REAL FUNCTION ERFC(X)
      DOUBLE PRECISION FUNCTION DERFC(X)
C--------------------------------------------------------------------
C
C This subprogram computes approximate values for erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, January 8, 1985
C
C--------------------------------------------------------------------
      INTEGER JINT
CS    REAL             X, RESULT
      DOUBLE PRECISION X, RESULT
C------------------------------------------------------------------
      JINT = 1
      CALL CALERF(X,RESULT,JINT)
CS    ERFC = RESULT
      DERFC = RESULT
      RETURN
C---------- Last card of DERFC ----------
      END
CS    REAL FUNCTION ERFCX(X)
      DOUBLE PRECISION FUNCTION DERFCX(X)
C------------------------------------------------------------------
C
C This subprogram computes approximate values for exp(x*x) * erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, March 30, 1987
C
C------------------------------------------------------------------
      INTEGER JINT
CS    REAL             X, RESULT
      DOUBLE PRECISION X, RESULT
C------------------------------------------------------------------
      JINT = 2
      CALL CALERF(X,RESULT,JINT)
CS    ERFCX = RESULT
      DERFCX = RESULT
      RETURN
C---------- Last card of DERFCX ----------
      END
