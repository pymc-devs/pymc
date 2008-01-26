      SUBROUTINE CAXPY(N,CA,CX,INCX,CY,INCY)
*     .. Scalar Arguments ..
      COMPLEX CA
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      COMPLEX CX(*),CY(*)
*     ..
*
*  Purpose
*  =======
*
*     CAXPY constant times a vector plus a vector.
*
*  Further Details
*  ===============
*
*     jack dongarra, linpack, 3/11/78.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*     .. Local Scalars ..
      INTEGER I,IX,IY
*     ..
*     .. External Functions ..
      REAL SCABS1
      EXTERNAL SCABS1
*     ..
      IF (N.LE.0) RETURN
      IF (SCABS1(CA).EQ.0.0E+0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) GO TO 20
*
*        code for unequal increments or equal increments
*          not equal to 1
*
      IX = 1
      IY = 1
      IF (INCX.LT.0) IX = (-N+1)*INCX + 1
      IF (INCY.LT.0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
          CY(IY) = CY(IY) + CA*CX(IX)
          IX = IX + INCX
          IY = IY + INCY
   10 CONTINUE
      RETURN
*
*        code for both increments equal to 1
*
   20 DO 30 I = 1,N
          CY(I) = CY(I) + CA*CX(I)
   30 CONTINUE
      RETURN
      END
