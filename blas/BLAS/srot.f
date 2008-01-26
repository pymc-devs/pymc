      SUBROUTINE SROT(N,SX,INCX,SY,INCY,C,S)
*     .. Scalar Arguments ..
      REAL C,S
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      REAL SX(*),SY(*)
*     ..
*
*  Purpose
*  =======
*
*     applies a plane rotation.
*
*  Further Details
*  ===============
*
*     jack dongarra, linpack, 3/11/78.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*

*     .. Local Scalars ..
      REAL STEMP
      INTEGER I,IX,IY
*     ..
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) GO TO 20
*
*       code for unequal increments or equal increments not equal
*         to 1
*
      IX = 1
      IY = 1
      IF (INCX.LT.0) IX = (-N+1)*INCX + 1
      IF (INCY.LT.0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
          STEMP = C*SX(IX) + S*SY(IY)
          SY(IY) = C*SY(IY) - S*SX(IX)
          SX(IX) = STEMP
          IX = IX + INCX
          IY = IY + INCY
   10 CONTINUE
      RETURN
*
*       code for both increments equal to 1
*
   20 DO 30 I = 1,N
          STEMP = C*SX(I) + S*SY(I)
          SY(I) = C*SY(I) - S*SX(I)
          SX(I) = STEMP
   30 CONTINUE
      RETURN
      END
