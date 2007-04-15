

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


      SUBROUTINE dgetrf_wrap(A,n,info,ipiv)

cf2py double precision dimension(n,n), intent(inplace) :: A
cf2py integer depend(A), intent(hide)::n=shape(A,0)
cf2py integer intent(out)::info
cf2py integer intent(out), dimension(n)::ipiv

      DOUBLE PRECISION A(n,n)
      INTEGER n, info, ipiv(n), i, j

      EXTERNAL DGETRF
! DGETRF( M, N, A, LDA, IPIV, INFO ) LU factorization, L gets unit diagonals, U gets nontrivial diagonals.
      
!     A <- LU(A)      
      call DGETRF( n, n, A, n, ipiv, info )
      
!     Multiply columns by either sqrt of diagonal or zero, interchange row and column,
!     zero lower triangle.

      do j=1,n
        if (A(j,j) .GE. 0.0D0) then
          A(j,j) = dsqrt(A(j,j))
        else
          A(j,j) = 0.0D0
        endif
      enddo
      
      do i=2,n
        do j=1,i-1
          A(j,i) = A(i,j) * A(j,j)
          A(i,j) = 0.0D0
        enddo
      enddo
      
      return
      END


      SUBROUTINE dpotrs_wrap(chol_fac, b, info, n, m)

cf2py double precision dimension(n,n), intent(inplace) :: chol_fac
cf2py integer depend(b), intent(hide)::n=shape(b,0)
cf2py integer depend(b), intent(hide)::m=shape(b,1)
cf2py integer depend(chol_fac), intent(hide)::n=shape(chol_fac,0)
cf2py integer intent(out)::info
cf2py double precision intent(inplace), dimension(n,m)::b

      DOUBLE PRECISION chol_fac(n,n), b(n,m)
      INTEGER n, info

      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system

      call DPOTRS('L',n,m,chol_fac,n,b,s,info)
      
      return
      END