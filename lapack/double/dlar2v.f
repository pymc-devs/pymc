      SUBROUTINE DLAR2V( N, X, Y, Z, INCX, C, S, INCC )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INCC, INCX, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   C( * ), S( * ), X( * ), Y( * ), Z( * )
*     ..
*
*  Purpose
*  =======
*
*  DLAR2V applies a vector of real plane rotations from both sides to
*  a sequence of 2-by-2 real symmetric matrices, defined by the elements
*  of the vectors x, y and z. For i = 1,2,...,n
*
*     ( x(i)  z(i) ) := (  c(i)  s(i) ) ( x(i)  z(i) ) ( c(i) -s(i) )
*     ( z(i)  y(i) )    ( -s(i)  c(i) ) ( z(i)  y(i) ) ( s(i)  c(i) )
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The number of plane rotations to be applied.
*
*  X       (input/output) DOUBLE PRECISION array,
*                         dimension (1+(N-1)*INCX)
*          The vector x.
*
*  Y       (input/output) DOUBLE PRECISION array,
*                         dimension (1+(N-1)*INCX)
*          The vector y.
*
*  Z       (input/output) DOUBLE PRECISION array,
*                         dimension (1+(N-1)*INCX)
*          The vector z.
*
*  INCX    (input) INTEGER
*          The increment between elements of X, Y and Z. INCX > 0.
*
*  C       (input) DOUBLE PRECISION array, dimension (1+(N-1)*INCC)
*          The cosines of the plane rotations.
*
*  S       (input) DOUBLE PRECISION array, dimension (1+(N-1)*INCC)
*          The sines of the plane rotations.
*
*  INCC    (input) INTEGER
*          The increment between elements of C and S. INCC > 0.
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, IC, IX
      DOUBLE PRECISION   CI, SI, T1, T2, T3, T4, T5, T6, XI, YI, ZI
*     ..
*     .. Executable Statements ..
*
      IX = 1
      IC = 1
      DO 10 I = 1, N
         XI = X( IX )
         YI = Y( IX )
         ZI = Z( IX )
         CI = C( IC )
         SI = S( IC )
         T1 = SI*ZI
         T2 = CI*ZI
         T3 = T2 - SI*XI
         T4 = T2 + SI*YI
         T5 = CI*XI + T1
         T6 = CI*YI - T1
         X( IX ) = CI*T5 + SI*T4
         Y( IX ) = CI*T6 - SI*T3
         Z( IX ) = CI*T4 - SI*T5
         IX = IX + INCX
         IC = IC + INCC
   10 CONTINUE
*
*     End of DLAR2V
*
      RETURN
      END
