      SUBROUTINE cov_mvnorm(x, mu, C, n, like, info)

cf2py double precision dimension(n), intent(copy) :: x
cf2py double precision dimension(n), intent(copy) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py double precision dimension(n,n), intent(copy) :: C
cf2py integer intent(hide), depend(C) :: n = shape(C,0)
cf2py double precision intent(out) :: like
cf2py integer intent(hide) :: info

      DOUBLE PRECISION C(n,n), x(n), mu(n), like
      INTEGER n, info
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION twopi_N, log_detC

      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system
      EXTERNAL DAXPY
! DAXPY(N,DA,DX,INCX,DY,INCY) Adding vectors
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y
      EXTERNAL DDOT
      
!     C <- cholesky(C)      
      call DPOTRF( 'L', n, C, n, info )
!       print *, C
      
!     Puke if C not positive definite
      if (info .GT. 0) then
        like=-infinity
        RETURN
      endif

!     x <- (x-mu)      
      call DAXPY(n, -1.0D0, mu, 1, x, 1)
      
!       mu <- x
      call DCOPY(n,x,1,mu,1)
      
!     x <- C ^-1 * x
      call DPOTRS('L',n,1,C,n,x,n,info)
      
!     like <- .5 dot(x,mu) (.5 (x-mu) C^{-1} (x-mu)^T)
      like = -0.5D0 * DDOT(n, x, 1, mu, 1)
!       print *, like
      
      twopi_N = 0.5D0 * N * dlog(2.0D0*PI)
!       print *, twopi_N
      
      log_detC = 0.0D0
      do i=1,n
        log_detC = log_detC + log(C(i,i))
      enddo
!       print *, log_detC
      
      like = like - twopi_N - log_detC
      
      return
      END

      SUBROUTINE blas_mvnorm(x, mu, tau, n, like)

cf2py double precision dimension(n), intent(copy) :: x
cf2py double precision dimension(n), intent(copy) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py double precision dimension(n,n), intent(in) :: tau
cf2py integer intent(hide), depend(tau) :: n = shape(tau,0)
cf2py double precision intent(out) :: like

      DOUBLE PRECISION tau(n,n), x(n), mu(n), like
      INTEGER n, info
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0) 
      DOUBLE PRECISION twopi_N, log_dettau

      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      EXTERNAL DSYMV
! Symmetric matrix-vector multiply
      EXTERNAL DAXPY
! DAXPY(N,DA,DX,INCX,DY,INCY) Adding vectors
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y
      EXTERNAL DDOT

      twopi_N = 0.5D0 * N * dlog(2.0D0*PI)

!     x <- (x-mu)      
      call DAXPY(n, -1.0D0, mu, 1, x, 1)

!       mu <- x
      call DCOPY(n,x,1,mu,1)

!     x <- tau * x
      call DSYMV('L',n,1.0D0,tau,n,x,1,0.0D0,mu,1)

!     like <- .5 dot(x,mu) (.5 (x-mu) C^{-1} (x-mu)^T)
      like = -0.5D0 * DDOT(n, x, 1, mu, 1)

!      Cholesky factorize tau for the determinant.      
       call DPOTRF( 'L', n, tau, n, info )
      
!      If cholesky failed, puke.
       if (info .GT. 0) then
         like = -infinity
         RETURN
       endif

!      Otherwise read off determinant.
       log_dettau = 0.0D0
       do i=1,n
         log_dettau = log_dettau + dlog(tau(i,i))
       enddo
            
       like = like - twopi_N + log_dettau
      
      return
      END
      
      SUBROUTINE blas_wishart(X,k,n,sigma,like)

c Wishart log-likelihood function.
c Doesn't vectorize the determinants, just the matrix multiplication.

cf2py double precision dimension(k,k),intent(in) :: X,sigma
cf2py double precision intent(in) :: n
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)

      INTEGER i,k
      DOUBLE PRECISION X(k,k),sigma(k,k),bx(k,k)
      DOUBLE PRECISION dx,n,db,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
c
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y      
      EXTERNAL DSYMM
! DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC) alpha*A*B + beta*C when side='l'
      print *, 'Warning, vectorized Wisharts are untested'
c determinants
      call dtrm(X,k,dx) 
      call dtrm(sigma,k,db)

c trace of sigma*X
!       bx <- X
      call DCOPY(n * n,X,1,bx,1)

!     bx <- sigma * bx
      call DSYMM('l','L',n,n,1.0D0,sigma,n,x,n,0.0D0,bx)
      call matmult(sigma,X,bx,k,k,k,k)
      call trace(bx,k,tbx)
      
      if ((dx .LE. 0.0) .OR. (db .LE. 0.0)) then
        like = -infinity
        RETURN
      endif
      
      if (k .GT. n) then
        like = -infinity
        RETURN
      endif
      
      like = (n - k - 1)/2.0 * dlog(dx)
      like = like + (n/2.0)*dlog(db)
      like = like - 0.5*tbx
      like = like - (n*k/2.0)*dlog(2.0d0)

      do i=1,k
        a = (n - i + 1)/2.0
        call gamfun(a, g)
        like = like - dlog(g)
      enddo

      return
      END

      SUBROUTINE gamfun(xx,gx)

c Return the logarithm of the gamma function
c Corresponds to scipy.special.gammaln

cf2py double precision intent(in) :: xx
cf2py double precision intent(out) :: gx

      INTEGER i
      DOUBLE PRECISION x,xx,ser,tmp,gx
      DIMENSION coeff(6)
      DATA coeff/76.18009173,-86.50532033,24.01409822,
     +-1.231739516,0.00120858003,-0.00000536382/

      x = xx
      tmp = x + 5.5
      tmp = tmp - (x+0.5) * dlog(tmp)
      ser = 1.000000000190015
      do i=1,6
        x = x+1
        ser = ser + coeff(i)/x
      enddo
      gx = -tmp + dlog(2.50662827465*ser/xx)
      return
      END

c

      SUBROUTINE blas_wishart_cov(X,k,n,V,like)

c Wishart log-likelihood function.
c Doesn't vectorize the determinants, just the matrix multiplication.

cf2py double precision dimension(k,k),intent(in) :: V,X
cf2py double precision intent(in) :: n
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)

      INTEGER i,k,info
      DOUBLE PRECISION X(k,k),V(k,k),bx(k,k)
      DOUBLE PRECISION dx,n,sqrtdb,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
c
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y      
      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system
      print *, 'Warning, vectorized Wisharts are untested'
c determinants
      call dtrm(X,k,dx)
      
c Cholesky factorize sigma      
!     V <- cholesky(V)      
      call DPOTRF( 'L', n, V, n, info )
! determinant of sigma
      sqrtdb=1.0D0
      do i=1,n
        sqrtdb = sqrtdb * V(i,i)
      enddo
c trace of sigma*X
!     bx <- X
      call DCOPY(n * n,X,1,bx,1)
!     bx <- sigma * bx
      call DPOTRS('L',n,n,sigma,n,bx,n,info)
      call trace(bx,k,tbx)
      
      if ((dx .LE. 0.0) .OR. (db .LE. 0.0)) then
        like = -infinity
        RETURN
      endif
      
      if (k .GT. n) then
        like = -infinity
        RETURN
      endif
      
      like = (n - k - 1)/2.0 * dlog(dx)
      like = like + (n)*dlog(sqrtdb)
      like = like - 0.5*tbx
      like = like - (n*k/2.0)*dlog(2.0d0)

      do i=1,k
        a = (n - i + 1)/2.0
        call gamfun(a, g)
        like = like - dlog(g)
      enddo

      return
      END

