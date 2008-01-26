      SUBROUTINE DPBEQU( UPLO, N, KD, AB, LDAB, S, SCOND, AMAX, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, KD, LDAB, N
      DOUBLE PRECISION   AMAX, SCOND
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AB( LDAB, * ), S( * )
*     ..
*
*  Purpose
*  =======
*
*  DPBEQU computes row and column scalings intended to equilibrate a
*  symmetric positive definite band matrix A and reduce its condition
*  number (with respect to the two-norm).  S contains the scale factors,
*  S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
*  elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
*  choice of S puts the condition number of B within a factor N of the
*  smallest possible condition number over all possible diagonal
*  scalings.
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangular of A is stored;
*          = 'L':  Lower triangular of A is stored.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  KD      (input) INTEGER
*          The number of superdiagonals of the matrix A if UPLO = 'U',
*          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
*
*  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
*          The upper or lower triangle of the symmetric band matrix A,
*          stored in the first KD+1 rows of the array.  The j-th column
*          of A is stored in the j-th column of the array AB as follows:
*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
*
*  LDAB     (input) INTEGER
*          The leading dimension of the array A.  LDAB >= KD+1.
*
*  S       (output) DOUBLE PRECISION array, dimension (N)
*          If INFO = 0, S contains the scale factors for A.
*
*  SCOND   (output) DOUBLE PRECISION
*          If INFO = 0, S contains the ratio of the smallest S(i) to
*          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too
*          large nor too small, it is not worth scaling by S.
*
*  AMAX    (output) DOUBLE PRECISION
*          Absolute value of largest matrix element.  If AMAX is very
*          close to overflow or very close to underflow, the matrix
*          should be scaled.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value.
*          > 0:  if INFO = i, the i-th diagonal element is nonpositive.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            I, J
      DOUBLE PRECISION   SMIN
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN, SQRT
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
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DPBEQU', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 ) THEN
         SCOND = ONE
         AMAX = ZERO
         RETURN
      END IF
*
      IF( UPPER ) THEN
         J = KD + 1
      ELSE
         J = 1
      END IF
*
*     Initialize SMIN and AMAX.
*
      S( 1 ) = AB( J, 1 )
      SMIN = S( 1 )
      AMAX = S( 1 )
*
*     Find the minimum and maximum diagonal elements.
*
      DO 10 I = 2, N
         S( I ) = AB( J, I )
         SMIN = MIN( SMIN, S( I ) )
         AMAX = MAX( AMAX, S( I ) )
   10 CONTINUE
*
      IF( SMIN.LE.ZERO ) THEN
*
*        Find the first non-positive diagonal element and return.
*
         DO 20 I = 1, N
            IF( S( I ).LE.ZERO ) THEN
               INFO = I
               RETURN
            END IF
   20    CONTINUE
      ELSE
*
*        Set the scale factors to the reciprocals
*        of the diagonal elements.
*
         DO 30 I = 1, N
            S( I ) = ONE / SQRT( S( I ) )
   30    CONTINUE
*
*        Compute SCOND = min(S(I)) / max(S(I))
*
         SCOND = SQRT( SMIN ) / SQRT( AMAX )
      END IF
      RETURN
*
*     End of DPBEQU
*
      END
