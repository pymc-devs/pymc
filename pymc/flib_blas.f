c Author: Anand Patil, anand.prabhakar.patil@gmail.com

      SUBROUTINE checksymm(X,n,cs)
c Checks symmetry
cf2py intent(hide) n
cf2py intent(out) cs
      DOUBLE PRECISION X(n,n)
      INTEGER n,i,j
      LOGICAL cs

      cs = .FALSE.
      do j=1,n-1
          do i=j+1,n
              if (X(i,j).NE.X(j,i)) then
                  cs = .TRUE.
                  return
              end if
          end do
      end do

      return
      END


      SUBROUTINE chol_mvnorm(x, mu, sig, n, like, info)

cf2py double precision dimension(n), intent(copy) :: x
cf2py double precision dimension(n), intent(copy) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py double precision dimension(n,n), intent(in) :: sig
cf2py double precision intent(out) :: like
cf2py integer intent(hide) :: info
cf2py threadsafe

      DOUBLE PRECISION sig(n,n), x(n), mu(n), like
      INTEGER n, info, i
      DOUBLE PRECISION twopi_N, log_detC, gd
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)


      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system
      EXTERNAL DAXPY
! DAXPY(N,DA,DX,INCX,DY,INCY) Adding vectors
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y
! NB DDOT from ATLAS, compiled with gfortran 4.2 on Ubuntu Gutsy,
! was producing bad output- hence the manual dot product.

!     x <- (x-mu)
      call DAXPY(n, -1.0D0, mu, 1, x, 1)

!       mu <- x
      call DCOPY(n,x,1,mu,1)

!     x <- sig ^-1 * x
      call DPOTRS('L',n,1,sig,n,x,n,info)

      gd=0.0D0
      do i=1,n
          gd=gd+x(i)*mu(i)
      end do

!     like <- .5 dot(x,mu) (.5 (x-mu) C^{-1} (x-mu)^T)
      like = -0.5D0 * gd
!       print *, like

      twopi_N = 0.5D0 * N * dlog(2.0D0*PI)
!       print *, twopi_N

      log_detC = 0.0D0
      do i=1,n
        log_detC = log_detC + log(sig(i,i))
      enddo
!       print *, log_detC

      like = like - twopi_N - log_detC

      return
      END


      SUBROUTINE cov_mvnorm(x, mu, C, n, like, info)

cf2py double precision dimension(n), intent(copy) :: x
cf2py double precision dimension(n), intent(copy) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py double precision dimension(n,n), intent(copy) :: C
cf2py double precision intent(out) :: like
cf2py integer intent(hide) :: info
cf2py threadsafe

      DOUBLE PRECISION C(n,n), x(n), mu(n), like
      INTEGER n, info
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization


!     C <- cholesky(C)
      call DPOTRF( 'L', n, C, n, info )
!       print *, C

!      If cholesky failed, puke.
       if (info .GT. 0) then
         like = -infinity
         RETURN
       endif

!     Call to chol_mvnorm
      call chol_mvnorm(x,mu,C,n,like,info)

      return
      END

      SUBROUTINE prec_mvnorm(x, mu, tau, n, like)

cf2py double precision dimension(n), intent(copy) :: x
cf2py double precision dimension(n), intent(copy) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py double precision dimension(n,n), intent(copy) :: tau
cf2py double precision intent(out) :: like
cf2py threadsafe

      DOUBLE PRECISION tau(n,n), x(n), mu(n), like, gd
      INTEGER n, info, i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION twopi_N, log_dettau

      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      EXTERNAL DSYMV
! DSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY) Symmetric matrix-vector multiply
      EXTERNAL DAXPY
! DAXPY(N,DA,DX,INCX,DY,INCY) Adding vectors
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y

      twopi_N = 0.5D0 * N * dlog(2.0D0*PI)

!     x <- (x-mu)
      call DAXPY(n, -1.0D0, mu, 1, x, 1)

!       mu <- x
      call DCOPY(n,x,1,mu,1)

!     mu <- tau * x
      call DSYMV('L',n,1.0D0,tau,n,x,1,0.0D0,mu,1)

      gd = 0.0D0
      do i=1,n
          gd=gd+x(i)*mu(i)
      end do

!     like <- -.5 dot(x,mu) (.5 (x-mu) C^{-1} (x-mu)^T)
      like = -0.5D0 * gd

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

      SUBROUTINE blas_wishart(X,k,n,T,like)

c Wishart log-likelihood function.

cf2py double precision dimension(k,k),intent(copy) :: X,T
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)
cf2py threadsafe

      INTEGER i,k,n,info
      DOUBLE PRECISION X(k,k),T(k,k),bx(k,k)
      DOUBLE PRECISION dx,db,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION PI
      LOGICAL csx

      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y
      EXTERNAL DSYMM
! DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC) alpha*A*B + beta*C when side='l'
      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization

c Check X for symmetry
      CALL checksymm(X,k,csx)
      if (csx) then
          like = -infinity
          return
      end if

c trace of T*X
!     bx <- T * X
      call DSYMM('L','L',k,k,1.0D0,T,k,x,k,0.0D0,bx,k)

c Cholesky factor T, puke if not pos def.
      call DPOTRF( 'L', k, T, k, info )
C       if (info .GT. 0) then
C         like = -infinity
C         RETURN
C       endif

c Cholesky factor X, puke if not pos def.
      call DPOTRF( 'L', k, X, k, info )
C       if (info .GT. 0) then
C         like = -infinity
C         RETURN
C       endif

c Get the trace and log-sqrt-determinants
      tbx = 0.0D0
      dx = 0.0D0
      db = 0.0D0

      do i=1,k
        tbx = tbx + bx(i,i)
        dx = dx + dlog(X(i,i))
        db = db + dlog(T(i,i))
      enddo

      if (k .GT. n) then
        like = -infinity
        RETURN
      endif

      like = 0.0D0
      like = (n - k - 1) * dx
      like = like + n * db
      like = like - 0.5D0 * tbx
      like = like - (0.5D0*n*k)*dlog(2.0d0)

      do i=1,k
        a = (n - i + 1)/2.0D0
        call gamfun(a, g)
        like = like - g
      enddo

C       like = like - k * (k-1) * 0.25D0 * dlog(PI)
!
      return
      END

c

      SUBROUTINE blas_wishart_cov(X,k,n,V,like)

c Wishart log-likelihood function.
c Doesn't vectorize the determinants, just the matrix multiplication.

cf2py double precision dimension(k,k),intent(copy) :: V,X
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)
cf2py threadsafe

      INTEGER i,k,info, n
      DOUBLE PRECISION X(k,k),V(k,k),bx(k,k)
      DOUBLE PRECISION dx,db,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION PI
      LOGICAL csx
      PARAMETER (PI=3.141592653589793238462643d0)
c
      EXTERNAL DCOPY
! DCOPY(N,DX,INCX,DY,INCY) copies x to y
      EXTERNAL DPOTRF
! DPOTRF( UPLO, N, A, LDA, INFO ) Cholesky factorization
      EXTERNAL DPOTRS
! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system

c determinants
c Check X for symmetry
      CALL checksymm(X,k,csx)
      if (csx) then
          like = -infinity
          return
      end if

c Cholesky factorize sigma, puke if not pos def
!     V <- cholesky(V)
      call DPOTRF( 'L', k, V, k, info )
      if (info .GT. 0) then
        like = -infinity
        RETURN
      endif

c trace of V^{-1}*X
!     bx <- X
      call DCOPY(k * k,X,1,bx,1)
!     bx <- V^{-1} * bx
      call DPOTRS('L',k,k,V,k,bx,k,info)

!     X <- cholesky(X)
      call DPOTRF( 'L', k, X, k, info )

! sqrt-log-determinant of sigma and X, and trace
      db=0.0D0
      dx=0.0D0
      tbx = 0.0D0
      do i=1,k
        db = db + dlog(V(i,i))
        dx = dx + dlog(X(i,i))
        tbx = tbx + bx(i,i)
      enddo

      if (k .GT. n) then
        like = -infinity
        RETURN
      endif


      like = (n - k - 1) * dx
      like = like - n * db
      like = like - 0.5D0*tbx
      like = like - (n*k/2.0D0)*dlog(2.0d0)

      do i=1,k
        a = (n - i + 1)/2.0D0
        call gamfun(a, g)
        like = like - g
      enddo

C       like = like - k * (k-1) * 0.25D0 * dlog(PI)

      return
      END