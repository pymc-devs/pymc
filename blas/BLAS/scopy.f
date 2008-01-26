      SUBROUTINE SCOPY(N,SX,INCX,SY,INCY)
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      REAL SX(*),SY(*)
*     ..
*
*  Purpose
*  =======
*
*     copies a vector, x, to a vector, y.
*     uses unrolled loops for increments equal to 1.
*     jack dongarra, linpack, 3/11/78.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
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
          SY(IY) = SX(IX)
          IX = IX + INCX
          IY = IY + INCY
   10 CONTINUE
      RETURN
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
   20 M = MOD(N,7)
      IF (M.EQ.0) GO TO 40
      DO 30 I = 1,M
          SY(I) = SX(I)
   30 CONTINUE
      IF (N.LT.7) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,7
          SY(I) = SX(I)
          SY(I+1) = SX(I+1)
          SY(I+2) = SX(I+2)
          SY(I+3) = SX(I+3)
          SY(I+4) = SX(I+4)
          SY(I+5) = SX(I+5)
          SY(I+6) = SX(I+6)
   50 CONTINUE
      RETURN
      END
