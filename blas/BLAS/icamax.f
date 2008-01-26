      INTEGER FUNCTION ICAMAX(N,CX,INCX)
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
*     .. External Functions ..
      REAL SCABS1
      EXTERNAL SCABS1
*     ..
      ICAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      ICAMAX = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      IX = 1
      SMAX = SCABS1(CX(1))
      IX = IX + INCX
      DO 10 I = 2,N
          IF (SCABS1(CX(IX)).LE.SMAX) GO TO 5
          ICAMAX = I
          SMAX = SCABS1(CX(IX))
    5     IX = IX + INCX
   10 CONTINUE
      RETURN
*
*        code for increment equal to 1
*
   20 SMAX = SCABS1(CX(1))
      DO 30 I = 2,N
          IF (SCABS1(CX(I)).LE.SMAX) GO TO 30
          ICAMAX = I
          SMAX = SCABS1(CX(I))
   30 CONTINUE
      RETURN
      END
