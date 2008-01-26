      SUBROUTINE DPTTS2( N, NRHS, D, E, B, LDB )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            LDB, N, NRHS
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   B( LDB, * ), D( * ), E( * )
*     ..
*
*  Purpose
*  =======
*
*  DPTTS2 solves a tridiagonal system of the form
*     A * X = B
*  using the L*D*L' factorization of A computed by DPTTRF.  D is a
*  diagonal matrix specified in the vector D, L is a unit bidiagonal
*  matrix whose subdiagonal is specified in the vector E, and X and B
*  are N by NRHS matrices.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The order of the tridiagonal matrix A.  N >= 0.
*
*  NRHS    (input) INTEGER
*          The number of right hand sides, i.e., the number of columns
*          of the matrix B.  NRHS >= 0.
*
*  D       (input) DOUBLE PRECISION array, dimension (N)
*          The n diagonal elements of the diagonal matrix D from the
*          L*D*L' factorization of A.
*
*  E       (input) DOUBLE PRECISION array, dimension (N-1)
*          The (n-1) subdiagonal elements of the unit bidiagonal factor
*          L from the L*D*L' factorization of A.  E can also be regarded
*          as the superdiagonal of the unit bidiagonal factor U from the
*          factorization A = U'*D*U.
*
*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
*          On entry, the right hand side vectors B for the system of
*          linear equations.
*          On exit, the solution vectors, X.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, J
*     ..
*     .. External Subroutines ..
      EXTERNAL           DSCAL
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( N.LE.1 ) THEN
         IF( N.EQ.1 )
     $      CALL DSCAL( NRHS, 1.D0 / D( 1 ), B, LDB )
         RETURN
      END IF
*
*     Solve A * X = B using the factorization A = L*D*L',
*     overwriting each right hand side vector with its solution.
*
      DO 30 J = 1, NRHS
*
*           Solve L * x = b.
*
         DO 10 I = 2, N
            B( I, J ) = B( I, J ) - B( I-1, J )*E( I-1 )
   10    CONTINUE
*
*           Solve D * L' * x = b.
*
         B( N, J ) = B( N, J ) / D( N )
         DO 20 I = N - 1, 1, -1
            B( I, J ) = B( I, J ) / D( I ) - B( I+1, J )*E( I )
   20    CONTINUE
   30 CONTINUE
*
      RETURN
*
*     End of DPTTS2
*
      END
