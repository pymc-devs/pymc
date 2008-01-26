      SUBROUTINE DTBTRS( UPLO, TRANS, DIAG, N, KD, NRHS, AB, LDAB, B,
     $                   LDB, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          DIAG, TRANS, UPLO
      INTEGER            INFO, KD, LDAB, LDB, N, NRHS
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )
*     ..
*
*  Purpose
*  =======
*
*  DTBTRS solves a triangular system of the form
*
*     A * X = B  or  A**T * X = B,
*
*  where A is a triangular band matrix of order N, and B is an
*  N-by NRHS matrix.  A check is made to verify that A is nonsingular.
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  A is upper triangular;
*          = 'L':  A is lower triangular.
*
*  TRANS   (input) CHARACTER*1
*          Specifies the form the system of equations:
*          = 'N':  A * X = B  (No transpose)
*          = 'T':  A**T * X = B  (Transpose)
*          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
*
*  DIAG    (input) CHARACTER*1
*          = 'N':  A is non-unit triangular;
*          = 'U':  A is unit triangular.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  KD      (input) INTEGER
*          The number of superdiagonals or subdiagonals of the
*          triangular band matrix A.  KD >= 0.
*
*  NRHS    (input) INTEGER
*          The number of right hand sides, i.e., the number of columns
*          of the matrix B.  NRHS >= 0.
*
*  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
*          The upper or lower triangular band matrix A, stored in the
*          first kd+1 rows of AB.  The j-th column of A is stored
*          in the j-th column of the array AB as follows:
*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
*          If DIAG = 'U', the diagonal elements of A are not referenced
*          and are assumed to be 1.
*
*  LDAB    (input) INTEGER
*          The leading dimension of the array AB.  LDAB >= KD+1.
*
*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
*          On entry, the right hand side matrix B.
*          On exit, if INFO = 0, the solution matrix X.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  if INFO = i, the i-th diagonal element of A is zero,
*                indicating that the matrix is singular and the
*                solutions X have not been computed.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            NOUNIT, UPPER
      INTEGER            J
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DTBSV, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      NOUNIT = LSAME( DIAG, 'N' )
      UPPER = LSAME( UPLO, 'U' )
      IF( .NOT.UPPER .AND. .NOT.LSAME( UPLO, 'L' ) ) THEN
         INFO = -1
      ELSE IF( .NOT.LSAME( TRANS, 'N' ) .AND. .NOT.
     $         LSAME( TRANS, 'T' ) .AND. .NOT.LSAME( TRANS, 'C' ) ) THEN
         INFO = -2
      ELSE IF( .NOT.NOUNIT .AND. .NOT.LSAME( DIAG, 'U' ) ) THEN
         INFO = -3
      ELSE IF( N.LT.0 ) THEN
         INFO = -4
      ELSE IF( KD.LT.0 ) THEN
         INFO = -5
      ELSE IF( NRHS.LT.0 ) THEN
         INFO = -6
      ELSE IF( LDAB.LT.KD+1 ) THEN
         INFO = -8
      ELSE IF( LDB.LT.MAX( 1, N ) ) THEN
         INFO = -10
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DTBTRS', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
*     Check for singularity.
*
      IF( NOUNIT ) THEN
         IF( UPPER ) THEN
            DO 10 INFO = 1, N
               IF( AB( KD+1, INFO ).EQ.ZERO )
     $            RETURN
   10       CONTINUE
         ELSE
            DO 20 INFO = 1, N
               IF( AB( 1, INFO ).EQ.ZERO )
     $            RETURN
   20       CONTINUE
         END IF
      END IF
      INFO = 0
*
*     Solve A * X = B  or  A' * X = B.
*
      DO 30 J = 1, NRHS
         CALL DTBSV( UPLO, TRANS, DIAG, N, KD, AB, LDAB, B( 1, J ), 1 )
   30 CONTINUE
*
      RETURN
*
*     End of DTBTRS
*
      END
