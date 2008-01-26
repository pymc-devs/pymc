      SUBROUTINE DLAE2( A, B, C, RT1, RT2 )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   A, B, C, RT1, RT2
*     ..
*
*  Purpose
*  =======
*
*  DLAE2  computes the eigenvalues of a 2-by-2 symmetric matrix
*     [  A   B  ]
*     [  B   C  ].
*  On return, RT1 is the eigenvalue of larger absolute value, and RT2
*  is the eigenvalue of smaller absolute value.
*
*  Arguments
*  =========
*
*  A       (input) DOUBLE PRECISION
*          The (1,1) element of the 2-by-2 matrix.
*
*  B       (input) DOUBLE PRECISION
*          The (1,2) and (2,1) elements of the 2-by-2 matrix.
*
*  C       (input) DOUBLE PRECISION
*          The (2,2) element of the 2-by-2 matrix.
*
*  RT1     (output) DOUBLE PRECISION
*          The eigenvalue of larger absolute value.
*
*  RT2     (output) DOUBLE PRECISION
*          The eigenvalue of smaller absolute value.
*
*  Further Details
*  ===============
*
*  RT1 is accurate to a few ulps barring over/underflow.
*
*  RT2 may be inaccurate if there is massive cancellation in the
*  determinant A*C-B*B; higher precision or correctly rounded or
*  correctly truncated arithmetic would be needed to compute RT2
*  accurately in all cases.
*
*  Overflow is possible only if RT1 is within a factor of 5 of overflow.
*  Underflow is harmless if the input data is 0 or exceeds
*     underflow_threshold / macheps.
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D0 )
      DOUBLE PRECISION   TWO
      PARAMETER          ( TWO = 2.0D0 )
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D0 )
      DOUBLE PRECISION   HALF
      PARAMETER          ( HALF = 0.5D0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   AB, ACMN, ACMX, ADF, DF, RT, SM, TB
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, SQRT
*     ..
*     .. Executable Statements ..
*
*     Compute the eigenvalues
*
      SM = A + C
      DF = A - C
      ADF = ABS( DF )
      TB = B + B
      AB = ABS( TB )
      IF( ABS( A ).GT.ABS( C ) ) THEN
         ACMX = A
         ACMN = C
      ELSE
         ACMX = C
         ACMN = A
      END IF
      IF( ADF.GT.AB ) THEN
         RT = ADF*SQRT( ONE+( AB / ADF )**2 )
      ELSE IF( ADF.LT.AB ) THEN
         RT = AB*SQRT( ONE+( ADF / AB )**2 )
      ELSE
*
*        Includes case AB=ADF=0
*
         RT = AB*SQRT( TWO )
      END IF
      IF( SM.LT.ZERO ) THEN
         RT1 = HALF*( SM-RT )
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
      ELSE IF( SM.GT.ZERO ) THEN
         RT1 = HALF*( SM+RT )
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
      ELSE
*
*        Includes case RT1 = RT2 = 0
*
         RT1 = HALF*RT
         RT2 = -HALF*RT
      END IF
      RETURN
*
*     End of DLAE2
*
      END
