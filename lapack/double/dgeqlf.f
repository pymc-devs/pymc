      SUBROUTINE DGEQLF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, LWORK, M, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DGEQLF computes a QL factorization of a real M-by-N matrix A:
*  A = Q * L.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M-by-N matrix A.
*          On exit,
*          if m >= n, the lower triangle of the subarray
*          A(m-n+1:m,1:n) contains the N-by-N lower triangular matrix L;
*          if m <= n, the elements on and below the (n-m)-th
*          superdiagonal contain the M-by-N lower trapezoidal matrix L;
*          the remaining elements, with the array TAU, represent the
*          orthogonal matrix Q as a product of elementary reflectors
*          (see Further Details).
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
*          The scalar factors of the elementary reflectors (see Further
*          Details).
*
*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*
*  LWORK   (input) INTEGER
*          The dimension of the array WORK.  LWORK >= max(1,N).
*          For optimum performance LWORK >= N*NB, where NB is the
*          optimal blocksize.
*
*          If LWORK = -1, then a workspace query is assumed; the routine
*          only calculates the optimal size of the WORK array, returns
*          this value as the first entry of the WORK array, and no error
*          message related to LWORK is issued by XERBLA.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  Further Details
*  ===============
*
*  The matrix Q is represented as a product of elementary reflectors
*
*     Q = H(k) . . . H(2) H(1), where k = min(m,n).
*
*  Each H(i) has the form
*
*     H(i) = I - tau * v * v'
*
*  where tau is a real scalar, and v is a real vector with
*  v(m-k+i+1:m) = 0 and v(m-k+i) = 1; v(1:m-k+i-1) is stored on exit in
*  A(1:m-k+i-1,n-k+i), and tau in TAU(i).
*
*  =====================================================================
*
*     .. Local Scalars ..
      LOGICAL            LQUERY
      INTEGER            I, IB, IINFO, IWS, K, KI, KK, LDWORK, LWKOPT,
     $                   MU, NB, NBMIN, NU, NX
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEQL2, DLARFB, DLARFT, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. External Functions ..
      INTEGER            ILAENV
      EXTERNAL           ILAENV
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      LQUERY = ( LWORK.EQ.-1 )
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -4
      END IF
*
      IF( INFO.EQ.0 ) THEN
         K = MIN( M, N )
         IF( K.EQ.0 ) THEN
            LWKOPT = 1
         ELSE
            NB = ILAENV( 1, 'DGEQLF', ' ', M, N, -1, -1 )
            LWKOPT = N*NB
         END IF
         WORK( 1 ) = LWKOPT
*
         IF( LWORK.LT.MAX( 1, N ) .AND. .NOT.LQUERY ) THEN
            INFO = -7
         END IF
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DGEQLF', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( K.EQ.0 ) THEN
         RETURN
      END IF
*
      NBMIN = 2
      NX = 1
      IWS = N
      IF( NB.GT.1 .AND. NB.LT.K ) THEN
*
*        Determine when to cross over from blocked to unblocked code.
*
         NX = MAX( 0, ILAENV( 3, 'DGEQLF', ' ', M, N, -1, -1 ) )
         IF( NX.LT.K ) THEN
*
*           Determine if workspace is large enough for blocked code.
*
            LDWORK = N
            IWS = LDWORK*NB
            IF( LWORK.LT.IWS ) THEN
*
*              Not enough workspace to use optimal NB:  reduce NB and
*              determine the minimum value of NB.
*
               NB = LWORK / LDWORK
               NBMIN = MAX( 2, ILAENV( 2, 'DGEQLF', ' ', M, N, -1,
     $                 -1 ) )
            END IF
         END IF
      END IF
*
      IF( NB.GE.NBMIN .AND. NB.LT.K .AND. NX.LT.K ) THEN
*
*        Use blocked code initially.
*        The last kk columns are handled by the block method.
*
         KI = ( ( K-NX-1 ) / NB )*NB
         KK = MIN( K, KI+NB )
*
         DO 10 I = K - KK + KI + 1, K - KK + 1, -NB
            IB = MIN( K-I+1, NB )
*
*           Compute the QL factorization of the current block
*           A(1:m-k+i+ib-1,n-k+i:n-k+i+ib-1)
*
            CALL DGEQL2( M-K+I+IB-1, IB, A( 1, N-K+I ), LDA, TAU( I ),
     $                   WORK, IINFO )
            IF( N-K+I.GT.1 ) THEN
*
*              Form the triangular factor of the block reflector
*              H = H(i+ib-1) . . . H(i+1) H(i)
*
               CALL DLARFT( 'Backward', 'Columnwise', M-K+I+IB-1, IB,
     $                      A( 1, N-K+I ), LDA, TAU( I ), WORK, LDWORK )
*
*              Apply H' to A(1:m-k+i+ib-1,1:n-k+i-1) from the left
*
               CALL DLARFB( 'Left', 'Transpose', 'Backward',
     $                      'Columnwise', M-K+I+IB-1, N-K+I-1, IB,
     $                      A( 1, N-K+I ), LDA, WORK, LDWORK, A, LDA,
     $                      WORK( IB+1 ), LDWORK )
            END IF
   10    CONTINUE
         MU = M - K + I + NB - 1
         NU = N - K + I + NB - 1
      ELSE
         MU = M
         NU = N
      END IF
*
*     Use unblocked code to factor the last or only block
*
      IF( MU.GT.0 .AND. NU.GT.0 )
     $   CALL DGEQL2( MU, NU, A, LDA, TAU, WORK, IINFO )
*
      WORK( 1 ) = IWS
      RETURN
*
*     End of DGEQLF
*
      END
