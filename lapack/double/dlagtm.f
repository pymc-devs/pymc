      SUBROUTINE DLAGTM( TRANS, N, NRHS, ALPHA, DL, D, DU, X, LDX, BETA,
     $                   B, LDB )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          TRANS
      INTEGER            LDB, LDX, N, NRHS
      DOUBLE PRECISION   ALPHA, BETA
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   B( LDB, * ), D( * ), DL( * ), DU( * ),
     $                   X( LDX, * )
*     ..
*
*  Purpose
*  =======
*
*  DLAGTM performs a matrix-vector product of the form
*
*     B := alpha * A * X + beta * B
*
*  where A is a tridiagonal matrix of order N, B and X are N by NRHS
*  matrices, and alpha and beta are real scalars, each of which may be
*  0., 1., or -1.
*
*  Arguments
*  =========
*
*  TRANS   (input) CHARACTER*1
*          Specifies the operation applied to A.
*          = 'N':  No transpose, B := alpha * A * X + beta * B
*          = 'T':  Transpose,    B := alpha * A'* X + beta * B
*          = 'C':  Conjugate transpose = Transpose
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  NRHS    (input) INTEGER
*          The number of right hand sides, i.e., the number of columns
*          of the matrices X and B.
*
*  ALPHA   (input) DOUBLE PRECISION
*          The scalar alpha.  ALPHA must be 0., 1., or -1.; otherwise,
*          it is assumed to be 0.
*
*  DL      (input) DOUBLE PRECISION array, dimension (N-1)
*          The (n-1) sub-diagonal elements of T.
*
*  D       (input) DOUBLE PRECISION array, dimension (N)
*          The diagonal elements of T.
*
*  DU      (input) DOUBLE PRECISION array, dimension (N-1)
*          The (n-1) super-diagonal elements of T.
*
*  X       (input) DOUBLE PRECISION array, dimension (LDX,NRHS)
*          The N by NRHS matrix X.
*  LDX     (input) INTEGER
*          The leading dimension of the array X.  LDX >= max(N,1).
*
*  BETA    (input) DOUBLE PRECISION
*          The scalar beta.  BETA must be 0., 1., or -1.; otherwise,
*          it is assumed to be 1.
*
*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
*          On entry, the N by NRHS matrix B.
*          On exit, B is overwritten by the matrix expression
*          B := alpha * A * X + beta * B.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(N,1).
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Executable Statements ..
*
      IF( N.EQ.0 )
     $   RETURN
*
*     Multiply B by BETA if BETA.NE.1.
*
      IF( BETA.EQ.ZERO ) THEN
         DO 20 J = 1, NRHS
            DO 10 I = 1, N
               B( I, J ) = ZERO
   10       CONTINUE
   20    CONTINUE
      ELSE IF( BETA.EQ.-ONE ) THEN
         DO 40 J = 1, NRHS
            DO 30 I = 1, N
               B( I, J ) = -B( I, J )
   30       CONTINUE
   40    CONTINUE
      END IF
*
      IF( ALPHA.EQ.ONE ) THEN
         IF( LSAME( TRANS, 'N' ) ) THEN
*
*           Compute B := B + A*X
*
            DO 60 J = 1, NRHS
               IF( N.EQ.1 ) THEN
                  B( 1, J ) = B( 1, J ) + D( 1 )*X( 1, J )
               ELSE
                  B( 1, J ) = B( 1, J ) + D( 1 )*X( 1, J ) +
     $                        DU( 1 )*X( 2, J )
                  B( N, J ) = B( N, J ) + DL( N-1 )*X( N-1, J ) +
     $                        D( N )*X( N, J )
                  DO 50 I = 2, N - 1
                     B( I, J ) = B( I, J ) + DL( I-1 )*X( I-1, J ) +
     $                           D( I )*X( I, J ) + DU( I )*X( I+1, J )
   50             CONTINUE
               END IF
   60       CONTINUE
         ELSE
*
*           Compute B := B + A'*X
*
            DO 80 J = 1, NRHS
               IF( N.EQ.1 ) THEN
                  B( 1, J ) = B( 1, J ) + D( 1 )*X( 1, J )
               ELSE
                  B( 1, J ) = B( 1, J ) + D( 1 )*X( 1, J ) +
     $                        DL( 1 )*X( 2, J )
                  B( N, J ) = B( N, J ) + DU( N-1 )*X( N-1, J ) +
     $                        D( N )*X( N, J )
                  DO 70 I = 2, N - 1
                     B( I, J ) = B( I, J ) + DU( I-1 )*X( I-1, J ) +
     $                           D( I )*X( I, J ) + DL( I )*X( I+1, J )
   70             CONTINUE
               END IF
   80       CONTINUE
         END IF
      ELSE IF( ALPHA.EQ.-ONE ) THEN
         IF( LSAME( TRANS, 'N' ) ) THEN
*
*           Compute B := B - A*X
*
            DO 100 J = 1, NRHS
               IF( N.EQ.1 ) THEN
                  B( 1, J ) = B( 1, J ) - D( 1 )*X( 1, J )
               ELSE
                  B( 1, J ) = B( 1, J ) - D( 1 )*X( 1, J ) -
     $                        DU( 1 )*X( 2, J )
                  B( N, J ) = B( N, J ) - DL( N-1 )*X( N-1, J ) -
     $                        D( N )*X( N, J )
                  DO 90 I = 2, N - 1
                     B( I, J ) = B( I, J ) - DL( I-1 )*X( I-1, J ) -
     $                           D( I )*X( I, J ) - DU( I )*X( I+1, J )
   90             CONTINUE
               END IF
  100       CONTINUE
         ELSE
*
*           Compute B := B - A'*X
*
            DO 120 J = 1, NRHS
               IF( N.EQ.1 ) THEN
                  B( 1, J ) = B( 1, J ) - D( 1 )*X( 1, J )
               ELSE
                  B( 1, J ) = B( 1, J ) - D( 1 )*X( 1, J ) -
     $                        DL( 1 )*X( 2, J )
                  B( N, J ) = B( N, J ) - DU( N-1 )*X( N-1, J ) -
     $                        D( N )*X( N, J )
                  DO 110 I = 2, N - 1
                     B( I, J ) = B( I, J ) - DU( I-1 )*X( I-1, J ) -
     $                           D( I )*X( I, J ) - DL( I )*X( I+1, J )
  110             CONTINUE
               END IF
  120       CONTINUE
         END IF
      END IF
      RETURN
*
*     End of DLAGTM
*
      END
