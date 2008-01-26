      DOUBLE PRECISION FUNCTION DLAPY3( X, Y, Z )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   X, Y, Z
*     ..
*
*  Purpose
*  =======
*
*  DLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause
*  unnecessary overflow.
*
*  Arguments
*  =========
*
*  X       (input) DOUBLE PRECISION
*  Y       (input) DOUBLE PRECISION
*  Z       (input) DOUBLE PRECISION
*          X, Y and Z specify the values x, y and z.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   W, XABS, YABS, ZABS
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, SQRT
*     ..
*     .. Executable Statements ..
*
      XABS = ABS( X )
      YABS = ABS( Y )
      ZABS = ABS( Z )
      W = MAX( XABS, YABS, ZABS )
      IF( W.EQ.ZERO ) THEN
*     W can be zero for max(0,nan,0)
*     adding all three entries together will make sure
*     NaN will not disappear.
         DLAPY3 =  XABS + YABS + ZABS
      ELSE
         DLAPY3 = W*SQRT( ( XABS / W )**2+( YABS / W )**2+
     $            ( ZABS / W )**2 )
      END IF
      RETURN
*
*     End of DLAPY3
*
      END
