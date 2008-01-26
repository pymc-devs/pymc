      SUBROUTINE DTRCON( NORM, UPLO, DIAG, N, A, LDA, RCOND, WORK,
     $                   IWORK, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     Modified to call DLACN2 in place of DLACON, 5 Feb 03, SJH.
*
*     .. Scalar Arguments ..
      CHARACTER          DIAG, NORM, UPLO
      INTEGER            INFO, LDA, N
      DOUBLE PRECISION   RCOND
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   A( LDA, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DTRCON estimates the reciprocal of the condition number of a
*  triangular matrix A, in either the 1-norm or the infinity-norm.
*
*  The norm of A is computed and an estimate is obtained for
*  norm(inv(A)), then the reciprocal of the condition number is
*  computed as
*     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
*
*  Arguments
*  =========
*
*  NORM    (input) CHARACTER*1
*          Specifies whether the 1-norm condition number or the
*          infinity-norm condition number is required:
*          = '1' or 'O':  1-norm;
*          = 'I':         Infinity-norm.
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  A is upper triangular;
*          = 'L':  A is lower triangular.
*
*  DIAG    (input) CHARACTER*1
*          = 'N':  A is non-unit triangular;
*          = 'U':  A is unit triangular.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
*          The triangular matrix A.  If UPLO = 'U', the leading N-by-N
*          upper triangular part of the array A contains the upper
*          triangular matrix, and the strictly lower triangular part of
*          A is not referenced.  If UPLO = 'L', the leading N-by-N lower
*          triangular part of the array A contains the lower triangular
*          matrix, and the strictly upper triangular part of A is not
*          referenced.  If DIAG = 'U', the diagonal elements of A are
*          also not referenced and are assumed to be 1.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N).
*
*  RCOND   (output) DOUBLE PRECISION
*          The reciprocal of the condition number of the matrix A,
*          computed as RCOND = 1/(norm(A) * norm(inv(A))).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
*
*  IWORK   (workspace) INTEGER array, dimension (N)
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
      LOGICAL            NOUNIT, ONENRM, UPPER
      CHARACTER          NORMIN
      INTEGER            IX, KASE, KASE1
      DOUBLE PRECISION   AINVNM, ANORM, SCALE, SMLNUM, XNORM
*     ..
*     .. Local Arrays ..
      INTEGER            ISAVE( 3 )
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            IDAMAX
      DOUBLE PRECISION   DLAMCH, DLANTR
      EXTERNAL           LSAME, IDAMAX, DLAMCH, DLANTR
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLACN2, DLATRS, DRSCL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      UPPER = LSAME( UPLO, 'U' )
      ONENRM = NORM.EQ.'1' .OR. LSAME( NORM, 'O' )
      NOUNIT = LSAME( DIAG, 'N' )
*
      IF( .NOT.ONENRM .AND. .NOT.LSAME( NORM, 'I' ) ) THEN
         INFO = -1
      ELSE IF( .NOT.UPPER .AND. .NOT.LSAME( UPLO, 'L' ) ) THEN
         INFO = -2
      ELSE IF( .NOT.NOUNIT .AND. .NOT.LSAME( DIAG, 'U' ) ) THEN
         INFO = -3
      ELSE IF( N.LT.0 ) THEN
         INFO = -4
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -6
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DTRCON', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 ) THEN
         RCOND = ONE
         RETURN
      END IF
*
      RCOND = ZERO
      SMLNUM = DLAMCH( 'Safe minimum' )*DBLE( MAX( 1, N ) )
*
*     Compute the norm of the triangular matrix A.
*
      ANORM = DLANTR( NORM, UPLO, DIAG, N, N, A, LDA, WORK )
*
*     Continue only if ANORM > 0.
*
      IF( ANORM.GT.ZERO ) THEN
*
*        Estimate the norm of the inverse of A.
*
         AINVNM = ZERO
         NORMIN = 'N'
         IF( ONENRM ) THEN
            KASE1 = 1
         ELSE
            KASE1 = 2
         END IF
         KASE = 0
   10    CONTINUE
         CALL DLACN2( N, WORK( N+1 ), WORK, IWORK, AINVNM, KASE, ISAVE )
         IF( KASE.NE.0 ) THEN
            IF( KASE.EQ.KASE1 ) THEN
*
*              Multiply by inv(A).
*
               CALL DLATRS( UPLO, 'No transpose', DIAG, NORMIN, N, A,
     $                      LDA, WORK, SCALE, WORK( 2*N+1 ), INFO )
            ELSE
*
*              Multiply by inv(A').
*
               CALL DLATRS( UPLO, 'Transpose', DIAG, NORMIN, N, A, LDA,
     $                      WORK, SCALE, WORK( 2*N+1 ), INFO )
            END IF
            NORMIN = 'Y'
*
*           Multiply by 1/SCALE if doing so will not cause overflow.
*
            IF( SCALE.NE.ONE ) THEN
               IX = IDAMAX( N, WORK, 1 )
               XNORM = ABS( WORK( IX ) )
               IF( SCALE.LT.XNORM*SMLNUM .OR. SCALE.EQ.ZERO )
     $            GO TO 20
               CALL DRSCL( N, SCALE, WORK, 1 )
            END IF
            GO TO 10
         END IF
*
*        Compute the estimate of the reciprocal condition number.
*
         IF( AINVNM.NE.ZERO )
     $      RCOND = ( ONE / ANORM ) / AINVNM
      END IF
*
   20 CONTINUE
      RETURN
*
*     End of DTRCON
*
      END
