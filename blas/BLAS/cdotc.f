      COMPLEX FUNCTION CDOTC(N,CX,INCX,CY,INCY)
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      COMPLEX CX(*),CY(*)
*     ..
*
*  Purpose
*  =======
*
*     forms the dot product of two vectors, conjugating the first
*     vector.
*
*  Further Details
*  ===============
*
*     jack dongarra, linpack,  3/11/78.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*     .. Local Scalars ..
      COMPLEX CTEMP
      INTEGER I,IX,IY
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC CONJG
*     ..
      CTEMP = (0.0,0.0)
      CDOTC = (0.0,0.0)
      IF (N.LE.0) RETURN
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
          CTEMP = CTEMP + CONJG(CX(IX))*CY(IY)
          IX = IX + INCX
          IY = IY + INCY
   10 CONTINUE
      CDOTC = CTEMP
      RETURN
*
*        code for both increments equal to 1
*
   20 DO 30 I = 1,N
          CTEMP = CTEMP + CONJG(CX(I))*CY(I)
   30 CONTINUE
      CDOTC = CTEMP
      RETURN
      END
