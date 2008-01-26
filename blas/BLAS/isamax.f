      INTEGER FUNCTION ISAMAX(N,SX,INCX)
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      REAL SX(*)
*     ..
*
*  Purpose
*  =======
*
*     finds the index of element having max. absolute value.
*     jack dongarra, linpack, 3/11/78.
*     modified 3/93 to return if incx .le. 0.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      REAL SMAX
      INTEGER I,IX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS
*     ..
      ISAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      ISAMAX = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      IX = 1
      SMAX = ABS(SX(1))
      IX = IX + INCX
      DO 10 I = 2,N
          IF (ABS(SX(IX)).LE.SMAX) GO TO 5
          ISAMAX = I
          SMAX = ABS(SX(IX))
    5     IX = IX + INCX
   10 CONTINUE
      RETURN
*
*        code for increment equal to 1
*
   20 SMAX = ABS(SX(1))
      DO 30 I = 2,N
          IF (ABS(SX(I)).LE.SMAX) GO TO 30
          ISAMAX = I
          SMAX = ABS(SX(I))
   30 CONTINUE
      RETURN
      END
