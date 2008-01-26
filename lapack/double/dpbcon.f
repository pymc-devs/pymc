      SUBROUTINE DPBCON( UPLO, N, KD, AB, LDAB, ANORM, RCOND, WORK,
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
      INTEGER            INFO, KD, LDAB, N
      DOUBLE PRECISION   ANORM, RCOND
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   AB( LDAB, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DPBCON estimates the reciprocal of the condition number (in the
*  1-norm) of a real symmetric positive definite band matrix using the
*  Cholesky factorization A = U**T*U or A = L*L**T computed by DPBTRF.
*
*  An estimate is obtained for norm(inv(A)), and the reciprocal of the
*  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangular factor stored in AB;
*          = 'L':  Lower triangular factor stored in AB.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  KD      (input) INTEGER
*          The number of superdiagonals of the matrix A if UPLO = 'U',
*          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
*
*  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
*          The triangular factor U or L from the Cholesky factorization
*          A = U**T*U or A = L*L**T of the band matrix A, stored in the
*          first KD+1 rows of the array.  The j-th column of U or L is
*          stored in the j-th column of the array AB as follows:
*          if UPLO ='U', AB(kd+1+i-j,j) = U(i,j) for max(1,j-kd)<=i<=j;
*          if UPLO ='L', AB(1+i-j,j)    = L(i,j) for j<=i<=min(n,j+kd).
*
*  LDAB    (input) INTEGER
*          The leading dimension of the array AB.  LDAB >= KD+1.
*
*  ANORM   (input) DOUBLE PRECISION
*          The 1-norm (or infinity-norm) of the symmetric band matrix A.
*
*  RCOND   (output) DOUBLE PRECISION
*          The reciprocal of the condition number of the matrix A,
*          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
*          estimate of the 1-norm of inv(A) computed in this routine.
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
      LOGICAL            UPPER
      CHARACTER          NORMIN
      INTEGER            IX, KASE
      DOUBLE PRECISION   AINVNM, SCALE, SCALEL, SCALEU, SMLNUM
*     ..
*     .. Local Arrays ..
      INTEGER            ISAVE( 3 )
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            IDAMAX
      DOUBLE PRECISION   DLAMCH
      EXTERNAL           LSAME, IDAMAX, DLAMCH
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLACN2, DLATBS, DRSCL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
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
      ELSE IF( KD.LT.0 ) THEN
         INFO = -3
      ELSE IF( LDAB.LT.KD+1 ) THEN
         INFO = -5
      ELSE IF( ANORM.LT.ZERO ) THEN
         INFO = -6
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DPBCON', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      RCOND = ZERO
      IF( N.EQ.0 ) THEN
         RCOND = ONE
         RETURN
      ELSE IF( ANORM.EQ.ZERO ) THEN
         RETURN
      END IF
*
      SMLNUM = DLAMCH( 'Safe minimum' )
*
*     Estimate the 1-norm of the inverse.
*
      KASE = 0
      NORMIN = 'N'
   10 CONTINUE
      CALL DLACN2( N, WORK( N+1 ), WORK, IWORK, AINVNM, KASE, ISAVE )
      IF( KASE.NE.0 ) THEN
         IF( UPPER ) THEN
*
*           Multiply by inv(U').
*
            CALL DLATBS( 'Upper', 'Transpose', 'Non-unit', NORMIN, N,
     $                   KD, AB, LDAB, WORK, SCALEL, WORK( 2*N+1 ),
     $                   INFO )
            NORMIN = 'Y'
*
*           Multiply by inv(U).
*
            CALL DLATBS( 'Upper', 'No transpose', 'Non-unit', NORMIN, N,
     $                   KD, AB, LDAB, WORK, SCALEU, WORK( 2*N+1 ),
     $                   INFO )
         ELSE
*
*           Multiply by inv(L).
*
            CALL DLATBS( 'Lower', 'No transpose', 'Non-unit', NORMIN, N,
     $                   KD, AB, LDAB, WORK, SCALEL, WORK( 2*N+1 ),
     $                   INFO )
            NORMIN = 'Y'
*
*           Multiply by inv(L').
*
            CALL DLATBS( 'Lower', 'Transpose', 'Non-unit', NORMIN, N,
     $                   KD, AB, LDAB, WORK, SCALEU, WORK( 2*N+1 ),
     $                   INFO )
         END IF
*
*        Multiply by 1/SCALE if doing so will not cause overflow.
*
         SCALE = SCALEL*SCALEU
         IF( SCALE.NE.ONE ) THEN
            IX = IDAMAX( N, WORK, 1 )
            IF( SCALE.LT.ABS( WORK( IX ) )*SMLNUM .OR. SCALE.EQ.ZERO )
     $         GO TO 20
            CALL DRSCL( N, SCALE, WORK, 1 )
         END IF
         GO TO 10
      END IF
*
*     Compute the estimate of the reciprocal condition number.
*
      IF( AINVNM.NE.ZERO )
     $   RCOND = ( ONE / AINVNM ) / ANORM
*
   20 CONTINUE
*
      RETURN
*
*     End of DPBCON
*
      END
