      SUBROUTINE DLAQPS( M, N, OFFSET, NB, KB, A, LDA, JPVT, TAU, VN1,
     $                   VN2, AUXV, F, LDF )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            KB, LDA, LDF, M, N, NB, OFFSET
*     ..
*     .. Array Arguments ..
      INTEGER            JPVT( * )
      DOUBLE PRECISION   A( LDA, * ), AUXV( * ), F( LDF, * ), TAU( * ),
     $                   VN1( * ), VN2( * )
*     ..
*
*  Purpose
*  =======
*
*  DLAQPS computes a step of QR factorization with column pivoting
*  of a real M-by-N matrix A by using Blas-3.  It tries to factorize
*  NB columns from A starting from the row OFFSET+1, and updates all
*  of the matrix with Blas-3 xGEMM.
*
*  In some cases, due to catastrophic cancellations, it cannot
*  factorize NB columns.  Hence, the actual number of factorized
*  columns is returned in KB.
*
*  Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A. M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A. N >= 0
*
*  OFFSET  (input) INTEGER
*          The number of rows of A that have been factorized in
*          previous steps.
*
*  NB      (input) INTEGER
*          The number of columns to factorize.
*
*  KB      (output) INTEGER
*          The number of columns actually factorized.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M-by-N matrix A.
*          On exit, block A(OFFSET+1:M,1:KB) is the triangular
*          factor obtained and block A(1:OFFSET,1:N) has been
*          accordingly pivoted, but no factorized.
*          The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
*          been updated.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A. LDA >= max(1,M).
*
*  JPVT    (input/output) INTEGER array, dimension (N)
*          JPVT(I) = K <==> Column K of the full matrix A has been
*          permuted into position I in AP.
*
*  TAU     (output) DOUBLE PRECISION array, dimension (KB)
*          The scalar factors of the elementary reflectors.
*
*  VN1     (input/output) DOUBLE PRECISION array, dimension (N)
*          The vector with the partial column norms.
*
*  VN2     (input/output) DOUBLE PRECISION array, dimension (N)
*          The vector with the exact column norms.
*
*  AUXV    (input/output) DOUBLE PRECISION array, dimension (NB)
*          Auxiliar vector.
*
*  F       (input/output) DOUBLE PRECISION array, dimension (LDF,NB)
*          Matrix F' = L*Y'*A.
*
*  LDF     (input) INTEGER
*          The leading dimension of the array F. LDF >= max(1,N).
*
*  Further Details
*  ===============
*
*  Based on contributions by
*    G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain
*    X. Sun, Computer Science Dept., Duke University, USA
*
*  Partial column norm updating strategy modified by
*    Z. Drmac and Z. Bujanovic, Dept. of Mathematics,
*    University of Zagreb, Croatia.
*    June 2006.
*  For more details see LAPACK Working Note 176.
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            ITEMP, J, K, LASTRK, LSTICC, PVT, RK
      DOUBLE PRECISION   AKK, TEMP, TEMP2, TOL3Z
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMM, DGEMV, DLARFG, DSWAP
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, MAX, MIN, NINT, SQRT
*     ..
*     .. External Functions ..
      INTEGER            IDAMAX
      DOUBLE PRECISION   DLAMCH, DNRM2
      EXTERNAL           IDAMAX, DLAMCH, DNRM2
*     ..
*     .. Executable Statements ..
*
      LASTRK = MIN( M, N+OFFSET )
      LSTICC = 0
      K = 0
      TOL3Z = SQRT(DLAMCH('Epsilon'))
*
*     Beginning of while loop.
*
   10 CONTINUE
      IF( ( K.LT.NB ) .AND. ( LSTICC.EQ.0 ) ) THEN
         K = K + 1
         RK = OFFSET + K
*
*        Determine ith pivot column and swap if necessary
*
         PVT = ( K-1 ) + IDAMAX( N-K+1, VN1( K ), 1 )
         IF( PVT.NE.K ) THEN
            CALL DSWAP( M, A( 1, PVT ), 1, A( 1, K ), 1 )
            CALL DSWAP( K-1, F( PVT, 1 ), LDF, F( K, 1 ), LDF )
            ITEMP = JPVT( PVT )
            JPVT( PVT ) = JPVT( K )
            JPVT( K ) = ITEMP
            VN1( PVT ) = VN1( K )
            VN2( PVT ) = VN2( K )
         END IF
*
*        Apply previous Householder reflectors to column K:
*        A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
*
         IF( K.GT.1 ) THEN
            CALL DGEMV( 'No transpose', M-RK+1, K-1, -ONE, A( RK, 1 ),
     $                  LDA, F( K, 1 ), LDF, ONE, A( RK, K ), 1 )
         END IF
*
*        Generate elementary reflector H(k).
*
         IF( RK.LT.M ) THEN
            CALL DLARFG( M-RK+1, A( RK, K ), A( RK+1, K ), 1, TAU( K ) )
         ELSE
            CALL DLARFG( 1, A( RK, K ), A( RK, K ), 1, TAU( K ) )
         END IF
*
         AKK = A( RK, K )
         A( RK, K ) = ONE
*
*        Compute Kth column of F:
*
*        Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K).
*
         IF( K.LT.N ) THEN
            CALL DGEMV( 'Transpose', M-RK+1, N-K, TAU( K ),
     $                  A( RK, K+1 ), LDA, A( RK, K ), 1, ZERO,
     $                  F( K+1, K ), 1 )
         END IF
*
*        Padding F(1:K,K) with zeros.
*
         DO 20 J = 1, K
            F( J, K ) = ZERO
   20    CONTINUE
*
*        Incremental updating of F:
*        F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'
*                    *A(RK:M,K).
*
         IF( K.GT.1 ) THEN
            CALL DGEMV( 'Transpose', M-RK+1, K-1, -TAU( K ), A( RK, 1 ),
     $                  LDA, A( RK, K ), 1, ZERO, AUXV( 1 ), 1 )
*
            CALL DGEMV( 'No transpose', N, K-1, ONE, F( 1, 1 ), LDF,
     $                  AUXV( 1 ), 1, ONE, F( 1, K ), 1 )
         END IF
*
*        Update the current row of A:
*        A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.
*
         IF( K.LT.N ) THEN
            CALL DGEMV( 'No transpose', N-K, K, -ONE, F( K+1, 1 ), LDF,
     $                  A( RK, 1 ), LDA, ONE, A( RK, K+1 ), LDA )
         END IF
*
*        Update partial column norms.
*
         IF( RK.LT.LASTRK ) THEN
            DO 30 J = K + 1, N
               IF( VN1( J ).NE.ZERO ) THEN
*
*                 NOTE: The following 4 lines follow from the analysis in
*                 Lapack Working Note 176.
*
                  TEMP = ABS( A( RK, J ) ) / VN1( J )
                  TEMP = MAX( ZERO, ( ONE+TEMP )*( ONE-TEMP ) )
                  TEMP2 = TEMP*( VN1( J ) / VN2( J ) )**2
                  IF( TEMP2 .LE. TOL3Z ) THEN
                     VN2( J ) = DBLE( LSTICC )
                     LSTICC = J
                  ELSE
                     VN1( J ) = VN1( J )*SQRT( TEMP )
                  END IF
               END IF
   30       CONTINUE
         END IF
*
         A( RK, K ) = AKK
*
*        End of while loop.
*
         GO TO 10
      END IF
      KB = K
      RK = OFFSET + KB
*
*     Apply the block reflector to the rest of the matrix:
*     A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
*                         A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'.
*
      IF( KB.LT.MIN( N, M-OFFSET ) ) THEN
         CALL DGEMM( 'No transpose', 'Transpose', M-RK, N-KB, KB, -ONE,
     $               A( RK+1, 1 ), LDA, F( KB+1, 1 ), LDF, ONE,
     $               A( RK+1, KB+1 ), LDA )
      END IF
*
*     Recomputation of difficult columns.
*
   40 CONTINUE
      IF( LSTICC.GT.0 ) THEN
         ITEMP = NINT( VN2( LSTICC ) )
         VN1( LSTICC ) = DNRM2( M-RK, A( RK+1, LSTICC ), 1 )
*
*        NOTE: The computation of VN1( LSTICC ) relies on the fact that 
*        SNRM2 does not fail on vectors with norm below the value of
*        SQRT(DLAMCH('S')) 
*
         VN2( LSTICC ) = VN1( LSTICC )
         LSTICC = ITEMP
         GO TO 40
      END IF
*
      RETURN
*
*     End of DLAQPS
*
      END
