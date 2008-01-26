      SUBROUTINE CSCAL(N,CA,CX,INCX)
*     .. Scalar Arguments ..
      COMPLEX CA
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      COMPLEX CX(*)
*     ..
*
*  Purpose
*  =======
*
*     scales a vector by a constant.
*     jack dongarra, linpack,  3/11/78.
*     modified 3/93 to return if incx .le. 0.
*     modified 12/3/93, array(1) declarations changed to array(*)
*
*
*     .. Local Scalars ..
      INTEGER I,NINCX
*     ..
      IF (N.LE.0 .OR. INCX.LE.0) RETURN
      IF (INCX.EQ.1) GO TO 20
*
*        code for increment not equal to 1
*
      NINCX = N*INCX
      DO 10 I = 1,NINCX,INCX
          CX(I) = CA*CX(I)
   10 CONTINUE
      RETURN
*
*        code for increment equal to 1
*
   20 DO 30 I = 1,N
          CX(I) = CA*CX(I)
   30 CONTINUE
      RETURN
      END
