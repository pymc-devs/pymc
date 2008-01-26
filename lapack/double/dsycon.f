      SUBROUTINE DSYCON( UPLO, N, A, LDA, IPIV, ANORM, RCOND, WORK,
     $                   IWORK, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     Modified to call DLACN2 in place of DLACON, 5 Feb 03, SJH.
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, LDA, N
      DOUBLE PRECISION   ANORM, RCOND
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * ), IWORK( * )
      DOUBLE PRECISION   A( LDA, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DSYCON estimates the reciprocal of the condition number (in the
*  1-norm) of a real symmetric matrix A using the factorization
*  A = U*D*U**T or A = L*D*L**T computed by DSYTRF.
*
*  An estimate is obtained for norm(inv(A)), and the reciprocal of the
*  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          Specifies whether the details of the factorization are stored
*          as an upper or lower triangular matrix.
*          = 'U':  Upper triangular, form is A = U*D*U**T;
*          = 'L':  Lower triangular, form is A = L*D*L**T.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
*          The block diagonal matrix D and the multipliers used to
*          obtain the factor U or L as computed by DSYTRF.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N).
*
*  IPIV    (input) INTEGER array, dimension (N)
*          Details of the interchanges and the block structure of D
*          as determined by DSYTRF.
*
*  ANORM   (input) DOUBLE PRECISION
*          The 1-norm of the original matrix A.
*
*  RCOND   (output) DOUBLE PRECISION
*          The reciprocal of the condition number of the matrix A,
*          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
*          estimate of the 1-norm of inv(A) computed in this routine.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
*
*  IWORK    (workspace) INTEGER array, dimension (N)
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            I, KASE
      DOUBLE PRECISION   AINVNM
*     ..
*     .. Local Arrays ..
      INTEGER            ISAVE( 3 )
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLACN2, DSYTRS, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      UPPER = LSAME( UPLO, 'U' )
      IF( .NOT.UPPER .AND. .NOT.LSAME( UPLO, 'L' ) ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -4
      ELSE IF( ANORM.LT.ZERO ) THEN
         INFO = -6
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSYCON', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      RCOND = ZERO
      IF( N.EQ.0 ) THEN
         RCOND = ONE
         RETURN
      ELSE IF( ANORM.LE.ZERO ) THEN
         RETURN
      END IF
*
*     Check that the diagonal matrix D is nonsingular.
*
      IF( UPPER ) THEN
*
*        Upper triangular storage: examine D from bottom to top
*
         DO 10 I = N, 1, -1
            IF( IPIV( I ).GT.0 .AND. A( I, I ).EQ.ZERO )
     $         RETURN
   10    CONTINUE
      ELSE
*
*        Lower triangular storage: examine D from top to bottom.
*
         DO 20 I = 1, N
            IF( IPIV( I ).GT.0 .AND. A( I, I ).EQ.ZERO )
     $         RETURN
   20    CONTINUE
      END IF
*
*     Estimate the 1-norm of the inverse.
*
      KASE = 0
   30 CONTINUE
      CALL DLACN2( N, WORK( N+1 ), WORK, IWORK, AINVNM, KASE, ISAVE )
      IF( KASE.NE.0 ) THEN
*
*        Multiply by inv(L*D*L') or inv(U*D*U').
*
         CALL DSYTRS( UPLO, N, 1, A, LDA, IPIV, WORK, N, INFO )
         GO TO 30
      END IF
*
*     Compute the estimate of the reciprocal condition number.
*
      IF( AINVNM.NE.ZERO )
     $   RCOND = ( ONE / AINVNM ) / ANORM
*
      RETURN
*
*     End of DSYCON
*
      END
