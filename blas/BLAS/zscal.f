      SUBROUTINE ZSCAL(N,ZA,ZX,INCX)
*     .. Scalar Arguments ..
      DOUBLE COMPLEX ZA
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE COMPLEX ZX(*)
*     ..
*
*  Purpose
*  =======
*
*     scales a vector by a constant.
*     jack dongarra, 3/11/78.
*     modified 3/93 to return if incx .le. 0.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      INTEGER I,IX
*     ..
      IF (N.LE.0 .OR. INCX.LE.0) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      IX = 1
      DO 10 I = 1,N
          ZX(IX) = ZA*ZX(IX)
          IX = IX + INCX
   10 CONTINUE
      RETURN
*
*        code for increment equal to 1
*
   20 DO 30 I = 1,N
          ZX(I) = ZA*ZX(I)
   30 CONTINUE
      RETURN
      END
