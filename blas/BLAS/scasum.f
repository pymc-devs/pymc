      REAL FUNCTION SCASUM(N,CX,INCX)
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      COMPLEX CX(*)
*     ..
*
*  Purpose
*  =======
*
*     takes the sum of the absolute values of a complex vector and
*     returns a single precision result.
*     jack dongarra, linpack, 3/11/78.
*     modified 3/93 to return if incx .le. 0.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      REAL STEMP
      INTEGER I,NINCX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS,AIMAG,REAL
*     ..
      SCASUM = 0.0e0
      STEMP = 0.0e0
      IF (N.LE.0 .OR. INCX.LE.0) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      NINCX = N*INCX
      DO 10 I = 1,NINCX,INCX
          STEMP = STEMP + ABS(REAL(CX(I))) + ABS(AIMAG(CX(I)))
   10 CONTINUE
      SCASUM = STEMP
      RETURN
*
*        code for increment equal to 1
*
   20 DO 30 I = 1,N
          STEMP = STEMP + ABS(REAL(CX(I))) + ABS(AIMAG(CX(I)))
   30 CONTINUE
      SCASUM = STEMP
      RETURN
      END
