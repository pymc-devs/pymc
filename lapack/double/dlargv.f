      SUBROUTINE DLARGV( N, X, INCX, Y, INCY, C, INCC )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INCC, INCX, INCY, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   C( * ), X( * ), Y( * )
*     ..
*
*  Purpose
*  =======
*
*  DLARGV generates a vector of real plane rotations, determined by
*  elements of the real vectors x and y. For i = 1,2,...,n
*
*     (  c(i)  s(i) ) ( x(i) ) = ( a(i) )
*     ( -s(i)  c(i) ) ( y(i) ) = (   0  )
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The number of plane rotations to be generated.
*
*  X       (input/output) DOUBLE PRECISION array,
*                         dimension (1+(N-1)*INCX)
*          On entry, the vector x.
*          On exit, x(i) is overwritten by a(i), for i = 1,...,n.
*
*  INCX    (input) INTEGER
*          The increment between elements of X. INCX > 0.
*
*  Y       (input/output) DOUBLE PRECISION array,
*                         dimension (1+(N-1)*INCY)
*          On entry, the vector y.
*          On exit, the sines of the plane rotations.
*
*  INCY    (input) INTEGER
*          The increment between elements of Y. INCY > 0.
*
*  C       (output) DOUBLE PRECISION array, dimension (1+(N-1)*INCC)
*          The cosines of the plane rotations.
*
*  INCC    (input) INTEGER
*          The increment between elements of C. INCC > 0.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, IC, IX, IY
      DOUBLE PRECISION   F, G, T, TT
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, SQRT
*     ..
*     .. Executable Statements ..
*
      IX = 1
      IY = 1
      IC = 1
      DO 10 I = 1, N
         F = X( IX )
         G = Y( IY )
         IF( G.EQ.ZERO ) THEN
            C( IC ) = ONE
         ELSE IF( F.EQ.ZERO ) THEN
            C( IC ) = ZERO
            Y( IY ) = ONE
            X( IX ) = G
         ELSE IF( ABS( F ).GT.ABS( G ) ) THEN
            T = G / F
            TT = SQRT( ONE+T*T )
            C( IC ) = ONE / TT
            Y( IY ) = T*C( IC )
            X( IX ) = F*TT
         ELSE
            T = F / G
            TT = SQRT( ONE+T*T )
            Y( IY ) = ONE / TT
            C( IC ) = T*Y( IY )
            X( IX ) = G*TT
         END IF
         IC = IC + INCC
         IY = IY + INCY
         IX = IX + INCX
   10 CONTINUE
      RETURN
*
*     End of DLARGV
*
      END
