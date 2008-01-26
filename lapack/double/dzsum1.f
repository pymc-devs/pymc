      DOUBLE PRECISION FUNCTION DZSUM1( N, CX, INCX )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INCX, N
*     ..
*     .. Array Arguments ..
      COMPLEX*16         CX( * )
*     ..
*
*  Purpose
*  =======
*
*  DZSUM1 takes the sum of the absolute values of a complex
*  vector and returns a double precision result.
*
*  Based on DZASUM from the Level 1 BLAS.
*  The change is to use the 'genuine' absolute value.
*
*  Contributed by Nick Higham for use with ZLACON.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The number of elements in the vector CX.
*
*  CX      (input) COMPLEX*16 array, dimension (N)
*          The vector whose elements will be summed.
*
*  INCX    (input) INTEGER
*          The spacing between successive values of CX.  INCX > 0.
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, NINCX
      DOUBLE PRECISION   STEMP
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
*     ..
*     .. Executable Statements ..
*
      DZSUM1 = 0.0D0
      STEMP = 0.0D0
      IF( N.LE.0 )
     $   RETURN
      IF( INCX.EQ.1 )
     $   GO TO 20
*
*     CODE FOR INCREMENT NOT EQUAL TO 1
*
      NINCX = N*INCX
      DO 10 I = 1, NINCX, INCX
*
*        NEXT LINE MODIFIED.
*
         STEMP = STEMP + ABS( CX( I ) )
   10 CONTINUE
      DZSUM1 = STEMP
      RETURN
*
*     CODE FOR INCREMENT EQUAL TO 1
*
   20 CONTINUE
      DO 30 I = 1, N
*
*        NEXT LINE MODIFIED.
*
         STEMP = STEMP + ABS( CX( I ) )
   30 CONTINUE
      DZSUM1 = STEMP
      RETURN
*
*     End of DZSUM1
*
      END
