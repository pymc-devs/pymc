      DOUBLE PRECISION FUNCTION DZASUM(N,ZX,INCX)
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE COMPLEX ZX(*)
*     ..
*
*  Purpose
*  =======
*
*     takes the sum of the absolute values.
*     jack dongarra, 3/11/78.
*     modified 3/93 to return if incx .le. 0.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      DOUBLE PRECISION STEMP
      INTEGER I,IX
*     ..
*     .. External Functions ..
      DOUBLE PRECISION DCABS1
      EXTERNAL DCABS1
*     ..
      DZASUM = 0.0d0
      STEMP = 0.0d0
      IF (N.LE.0 .OR. INCX.LE.0) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      IX = 1
      DO 10 I = 1,N
          STEMP = STEMP + DCABS1(ZX(IX))
          IX = IX + INCX
   10 CONTINUE
      DZASUM = STEMP
      RETURN
*
*        code for increment equal to 1
*
   20 DO 30 I = 1,N
          STEMP = STEMP + DCABS1(ZX(I))
   30 CONTINUE
      DZASUM = STEMP
      RETURN
      END
