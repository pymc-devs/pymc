

      SUBROUTINE dpotrf_wrap(A,n,info)

cf2py double precision dimension(n,n), intent(inplace) :: A
cf2py integer depend(A),intent(hide)::n=shape(A,0)
cf2py integer intent(out)::info

      DOUBLE PRECISION A(n,n)
      INTEGER n, info, i, j

      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      
!     C <- cholesky(C)      
      call DPOTRF( 'U', n, A, n, info )
      do i=2,n
        do j=1,i-1
          A(i,j)=0.0D0
        enddo
      enddo
      
      return
      END


      SUBROUTINE dpotrs_wrap(chol_fac, b, info, n, m, uplo)

cf2py double precision dimension(n,n), intent(inplace) :: chol_fac
cf2py integer depend(chol_fac), intent(hide)::n=shape(chol_fac,0)
cf2py integer depend(b), intent(hide)::m=shape(b,1)
cf2py optional character intent(in):: uplo='U'
cf2py integer intent(out)::info
cf2py double precision intent(inplace), dimension(n,m)::b

      DOUBLE PRECISION chol_fac(n,n), b(n,m)
      INTEGER n, info,m
      
      CHARACTER uplo

      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system

      call DPOTRS(uplo,n,m,chol_fac,n,b,s,info)
      
      return
      END

      
      SUBROUTINE robust_DPOTF2(N, A, LDA, good_rows, N_good_rows)
      
cf2py double precision intent(inplace), dimension(N,N) :: A
cf2py integer intent(hide), depend(A) :: N = shape(A,0)
cf2py integer intent(hide), depend(A) :: LDA = shape(A,0)
cf2py integer dimension(N), intent(out) :: good_rows
cf2py integer intent(out) :: N_good_rows

*     Modified from:
*     DPOTF2     
*
*  -- LAPACK routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
      INTEGER good_rows(N)
      INTEGER N_good_rows

*     .. Scalar Arguments ..
      INTEGER            LDA, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A(N,N)
*     ..

*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            J
      DOUBLE PRECISION   AJJ
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      DOUBLE PRECISION   DDOT
      EXTERNAL           LSAME, DDOT
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMV, DSCAL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, SQRT
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      UPPER = .TRUE.
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*

      N_good_rows = 0
      IF( .TRUE. ) THEN
*
*        Compute the Cholesky factorization A = U'*U.
*
         DO 10 J = 1, N
*
*           Compute U(J,J) and test for non-positive-definiteness.
*
            AJJ = A( J, J ) - DDOT( J-1, A( 1, J ), 1, A( 1, J ), 1 )
            IF( AJJ.LE.ZERO ) THEN
               A(J,J) = ZERO
            ELSE
               A(J,J) = SQRT( AJJ )
               N_good_rows = N_good_rows + 1
               good_rows(N_good_rows) = J-1
            END IF

*
*           Compute elements J+1:N of row J.
*
            IF( J.LT.N ) THEN
               IF (AJJ .GT. ZERO) THEN
                  CALL DGEMV( 'Transpose', J-1, N-J, -ONE, A( 1, J+1 ),
     $                        LDA, A( 1, J ), 1, ONE, A( J, J+1 ), LDA )
                  CALL DSCAL( N-J, ONE / A(J,J), A( J, J+1 ), LDA )
               ELSE
                  CALL DSCAL( N-J, ZERO, A( J, J+1 ), LDA )
               END IF
            END IF
   10    CONTINUE
      ENDIF
      
! Zero subdiagonals
      DO J=2,N
        CALL DSCAL( J-1, ZERO, A( J, 1 ), LDA )
      ENDDO
      RETURN

*
*     End of DPOTF2
*
      END


      SUBROUTINE DTRSM_wrap(M,N,A,B,UPLO)
cf2py double precision intent(inplace), dimension(m,n)::B
cf2py double precision intent(in), dimension(m,m)::A
cf2py optional character intent(in)::uplo='U'
cf2py integer intent(hide), depend(A)::M=shape(A,0)
cf2py integer intent(hide), depend(B)::N=shape(B,1)


      
*     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA
      INTEGER LDA,LDB,M,N
      CHARACTER DIAG,SIDE,TRANSA,UPLO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(M,M),B(M,N)
      
      EXTERNAL DTRSM
! DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
*     op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
      
      DIAG = 'N'
      ALPHA = 1.0D0
      LDA = M
      LDB = M
      SIDE = 'L'
      TRANSA = 'N'
      
      CALL DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
      RETURN
      END
