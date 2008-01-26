      SUBROUTINE DOPGTR( UPLO, N, AP, TAU, Q, LDQ, WORK, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, LDQ, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AP( * ), Q( LDQ, * ), TAU( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DOPGTR generates a real orthogonal matrix Q which is defined as the
*  product of n-1 elementary reflectors H(i) of order n, as returned by
*  DSPTRD using packed storage:
*
*  if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
*
*  if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          = 'U': Upper triangular packed storage used in previous
*                 call to DSPTRD;
*          = 'L': Lower triangular packed storage used in previous
*                 call to DSPTRD.
*
*  N       (input) INTEGER
*          The order of the matrix Q. N >= 0.
*
*  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
*          The vectors which define the elementary reflectors, as
*          returned by DSPTRD.
*
*  TAU     (input) DOUBLE PRECISION array, dimension (N-1)
*          TAU(i) must contain the scalar factor of the elementary
*          reflector H(i), as returned by DSPTRD.
*
*  Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
*          The N-by-N orthogonal matrix Q.
*
*  LDQ     (input) INTEGER
*          The leading dimension of the array Q. LDQ >= max(1,N).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (N-1)
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            I, IINFO, IJ, J
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DORG2L, DORG2R, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      UPPER = LSAME( UPLO, 'U' )
      IF( .NOT.UPPER .AND. .NOT.LSAME( UPLO, 'L' ) ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDQ.LT.MAX( 1, N ) ) THEN
         INFO = -6
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DOPGTR', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
      IF( UPPER ) THEN
*
*        Q was determined by a call to DSPTRD with UPLO = 'U'
*
*        Unpack the vectors which define the elementary reflectors and
*        set the last row and column of Q equal to those of the unit
*        matrix
*
         IJ = 2
         DO 20 J = 1, N - 1
            DO 10 I = 1, J - 1
               Q( I, J ) = AP( IJ )
               IJ = IJ + 1
   10       CONTINUE
            IJ = IJ + 2
            Q( N, J ) = ZERO
   20    CONTINUE
         DO 30 I = 1, N - 1
            Q( I, N ) = ZERO
   30    CONTINUE
         Q( N, N ) = ONE
*
*        Generate Q(1:n-1,1:n-1)
*
         CALL DORG2L( N-1, N-1, N-1, Q, LDQ, TAU, WORK, IINFO )
*
      ELSE
*
*        Q was determined by a call to DSPTRD with UPLO = 'L'.
*
*        Unpack the vectors which define the elementary reflectors and
*        set the first row and column of Q equal to those of the unit
*        matrix
*
         Q( 1, 1 ) = ONE
         DO 40 I = 2, N
            Q( I, 1 ) = ZERO
   40    CONTINUE
         IJ = 3
         DO 60 J = 2, N
            Q( 1, J ) = ZERO
            DO 50 I = J + 1, N
               Q( I, J ) = AP( IJ )
               IJ = IJ + 1
   50       CONTINUE
            IJ = IJ + 2
   60    CONTINUE
         IF( N.GT.1 ) THEN
*
*           Generate Q(2:n,2:n)
*
            CALL DORG2R( N-1, N-1, N-1, Q( 2, 2 ), LDQ, TAU, WORK,
     $                   IINFO )
         END IF
      END IF
      RETURN
*
*     End of DOPGTR
*
      END
