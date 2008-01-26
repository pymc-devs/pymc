      SUBROUTINE DSPGST( ITYPE, UPLO, N, AP, BP, INFO )
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, ITYPE, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AP( * ), BP( * )
*     ..
*
*  Purpose
*  =======
*
*  DSPGST reduces a real symmetric-definite generalized eigenproblem
*  to standard form, using packed storage.
*
*  If ITYPE = 1, the problem is A*x = lambda*B*x,
*  and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
*
*  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
*  B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
*
*  B must have been previously factorized as U**T*U or L*L**T by DPPTRF.
*
*  Arguments
*  =========
*
*  ITYPE   (input) INTEGER
*          = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
*          = 2 or 3: compute U*A*U**T or L**T*A*L.
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangle of A is stored and B is factored as
*                  U**T*U;
*          = 'L':  Lower triangle of A is stored and B is factored as
*                  L*L**T.
*
*  N       (input) INTEGER
*          The order of the matrices A and B.  N >= 0.
*
*  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
*          On entry, the upper or lower triangle of the symmetric matrix
*          A, packed columnwise in a linear array.  The j-th column of A
*          is stored in the array AP as follows:
*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
*
*          On exit, if INFO = 0, the transformed matrix, stored in the
*          same format as A.
*
*  BP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
*          The triangular factor from the Cholesky factorization of B,
*          stored in the same format as A, as returned by DPPTRF.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, HALF
      PARAMETER          ( ONE = 1.0D0, HALF = 0.5D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            J, J1, J1J1, JJ, K, K1, K1K1, KK
      DOUBLE PRECISION   AJJ, AKK, BJJ, BKK, CT
*     ..
*     .. External Subroutines ..
      EXTERNAL           DAXPY, DSCAL, DSPMV, DSPR2, DTPMV, DTPSV,
     $                   XERBLA
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      DOUBLE PRECISION   DDOT
      EXTERNAL           LSAME, DDOT
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      UPPER = LSAME( UPLO, 'U' )
      IF( ITYPE.LT.1 .OR. ITYPE.GT.3 ) THEN
         INFO = -1
      ELSE IF( .NOT.UPPER .AND. .NOT.LSAME( UPLO, 'L' ) ) THEN
         INFO = -2
      ELSE IF( N.LT.0 ) THEN
         INFO = -3
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSPGST', -INFO )
         RETURN
      END IF
*
      IF( ITYPE.EQ.1 ) THEN
         IF( UPPER ) THEN
*
*           Compute inv(U')*A*inv(U)
*
*           J1 and JJ are the indices of A(1,j) and A(j,j)
*
            JJ = 0
            DO 10 J = 1, N
               J1 = JJ + 1
               JJ = JJ + J
*
*              Compute the j-th column of the upper triangle of A
*
               BJJ = BP( JJ )
               CALL DTPSV( UPLO, 'Transpose', 'Nonunit', J, BP,
     $                     AP( J1 ), 1 )
               CALL DSPMV( UPLO, J-1, -ONE, AP, BP( J1 ), 1, ONE,
     $                     AP( J1 ), 1 )
               CALL DSCAL( J-1, ONE / BJJ, AP( J1 ), 1 )
               AP( JJ ) = ( AP( JJ )-DDOT( J-1, AP( J1 ), 1, BP( J1 ),
     $                    1 ) ) / BJJ
   10       CONTINUE
         ELSE
*
*           Compute inv(L)*A*inv(L')
*
*           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1)
*
            KK = 1
            DO 20 K = 1, N
               K1K1 = KK + N - K + 1
*
*              Update the lower triangle of A(k:n,k:n)
*
               AKK = AP( KK )
               BKK = BP( KK )
               AKK = AKK / BKK**2
               AP( KK ) = AKK
               IF( K.LT.N ) THEN
                  CALL DSCAL( N-K, ONE / BKK, AP( KK+1 ), 1 )
                  CT = -HALF*AKK
                  CALL DAXPY( N-K, CT, BP( KK+1 ), 1, AP( KK+1 ), 1 )
                  CALL DSPR2( UPLO, N-K, -ONE, AP( KK+1 ), 1,
     $                        BP( KK+1 ), 1, AP( K1K1 ) )
                  CALL DAXPY( N-K, CT, BP( KK+1 ), 1, AP( KK+1 ), 1 )
                  CALL DTPSV( UPLO, 'No transpose', 'Non-unit', N-K,
     $                        BP( K1K1 ), AP( KK+1 ), 1 )
               END IF
               KK = K1K1
   20       CONTINUE
         END IF
      ELSE
         IF( UPPER ) THEN
*
*           Compute U*A*U'
*
*           K1 and KK are the indices of A(1,k) and A(k,k)
*
            KK = 0
            DO 30 K = 1, N
               K1 = KK + 1
               KK = KK + K
*
*              Update the upper triangle of A(1:k,1:k)
*
               AKK = AP( KK )
               BKK = BP( KK )
               CALL DTPMV( UPLO, 'No transpose', 'Non-unit', K-1, BP,
     $                     AP( K1 ), 1 )
               CT = HALF*AKK
               CALL DAXPY( K-1, CT, BP( K1 ), 1, AP( K1 ), 1 )
               CALL DSPR2( UPLO, K-1, ONE, AP( K1 ), 1, BP( K1 ), 1,
     $                     AP )
               CALL DAXPY( K-1, CT, BP( K1 ), 1, AP( K1 ), 1 )
               CALL DSCAL( K-1, BKK, AP( K1 ), 1 )
               AP( KK ) = AKK*BKK**2
   30       CONTINUE
         ELSE
*
*           Compute L'*A*L
*
*           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1)
*
            JJ = 1
            DO 40 J = 1, N
               J1J1 = JJ + N - J + 1
*
*              Compute the j-th column of the lower triangle of A
*
               AJJ = AP( JJ )
               BJJ = BP( JJ )
               AP( JJ ) = AJJ*BJJ + DDOT( N-J, AP( JJ+1 ), 1,
     $                    BP( JJ+1 ), 1 )
               CALL DSCAL( N-J, BJJ, AP( JJ+1 ), 1 )
               CALL DSPMV( UPLO, N-J, ONE, AP( J1J1 ), BP( JJ+1 ), 1,
     $                     ONE, AP( JJ+1 ), 1 )
               CALL DTPMV( UPLO, 'Transpose', 'Non-unit', N-J+1,
     $                     BP( JJ ), AP( JJ ), 1 )
               JJ = J1J1
   40       CONTINUE
         END IF
      END IF
      RETURN
*
*     End of DSPGST
*
      END
