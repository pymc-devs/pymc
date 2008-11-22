! Copyright (c) Anand Patil, 2007

! TODO: Multithread zeroing the lower triangle, do it straight from Covariance.

      SUBROUTINE remove_duplicates(x,N,Nd,Nr,rf,rt,Nu,xu,ui)
cf2py intent(out) rt
cf2py intent(out) rf
cf2py intent(out) Nr
cf2py intent(out) xu
cf2py intent(out) Nu
cf2py intent(out) ui
cf2py intent(hide) N
cf2py intent(hide) Nd
      INTEGER N, Nd, i, j, k, rf(N), rt(N), Nr, ui(N)
      INTEGER Nu
      DOUBLE PRECISION x(N,Nd), xu(N,Nd)
      LOGICAL match
      
      Nr = 0
      Nu = 1
      do k=1,Nd
        xu(1,k) = x(1,k)
      end do
      ui(1)=0
      match=.FALSE.
      do i=2,N
        do j=1,i-1
          match=.TRUE.
          do k=1,Nd
            if(x(i,k).NE.x(j,k)) then
              match=.FALSE.
              go to 10
            end if
          end do
   10   if (match) then
          Nr=Nr+1
          rt(Nr)=i-1
          rf(Nr)=j-1
          go to 20
        end if
        end do
   20   if (.NOT.match) then
          Nu=Nu+1
          ui(Nu)=i-1
          do k=1,Nd
            xu(Nu,k)=x(i,k)
          end do
        end if
      end do
          
      RETURN
      END


      SUBROUTINE check_repeats(x, x_sofar, f_sofar, N, N_dim, N_sofar, 
     +f, new_indices, N_new_indices)
cf2py double precision dimension(N,N_dim), intent(in) :: x
cf2py double precision dimension(N_sofar, N_dim), intent(in) :: x_sofar
cf2py double precision dimension(N_sofar), intent(in) :: f_sofar
cf2py integer intent(hide), depend(x):: N = shape(x,0)
cf2py integer intent(hide), depend(x):: N_dim = shape(x,1)
cf2py integer intent(hide), depend(x_sofar):: N_sofar = shape(x_sofar,0)
cf2py double precision intent(out), dimension(N) :: f
cf2py integer intent(out), dimension(N):: new_indices
cf2py integer intent(out):: N_new_indices
      INTEGER N, N_dim, N_sofar,
     +N_new_indices, new_indices(N), i, j, k
      DOUBLE PRECISION x(N,N_dim), x_sofar(N_sofar, N_dim),
     +f(N), f_sofar(N_sofar)
      LOGICAL match
      
      N_new_indices = 0
      match=.FALSE.
      do i=1,N
!         N_new_indices = N_new_indices + 1
!         new_indices(N_new_indices) = i-1
        do j=1,N_sofar

          match=.TRUE.

          do k=1,N_dim
            if (x(i,k) .NE. x_sofar(j,k)) then
              match=.FALSE.
              GO TO 10
            endif
          enddo

   10     continue  
        
          if (match) then
            GO TO 20
          endif        
        enddo

   20   continue
   
        if (match) then
          f(i) = f_sofar(j)
        else
          N_new_indices = N_new_indices+1
          new_indices(N_new_indices) = i-1
        endif

      enddo
          
      RETURN
      END

      subroutine dcopy_wrap(x,y,nx)
cf2py intent(inplace) y
cf2py intent(hide) nx
      double precision x(nx), y(nx)
      external dcopy
!       dcopy(n,dx,incx,dy,incy)

      CALL dcopy(nx,x,1,y,1)
      RETURN
      END

      subroutine diag_call(x,n,ndim,V,cov_fun)
cf2py intent(hide) n
cf2py intent(hide) ndim
cf2py intent(out) V
        integer i, n, ndim, j
        double precision x(n,ndim), xe(1,ndim), V(n)
        external cov_fun
cf2py double precision q
cf2py q = cov_fun(xe,ndim)
        
        do i=1,n
             do j=1,ndim
                 xe(1,j) = x(i,j)
             enddo

            V(i) = cov_fun(xe,ndim)
!             print *,xe,cov_fun(xe,ndim)
        enddo
        
        return
        END 

      SUBROUTINE basis_diag_call(basis_x, V, n, nbas)
cf2py intent(hide) n
cf2py intent(hide) nbas
cf2py intent(out) V
        integer n, i, j
        double precision V(n), basis_x(nbas,n)
        
        do i=1,n
            V(i) = 0
            do j=1,nbas
                V(i) = V(i) + basis_x(j,i) ** 2
            enddo
        enddo
        
        return
        END
        

      SUBROUTINE dtrmm_wrap(M,N,A,B,UPLO,TRANSA)
cf2py double precision intent(inplace), dimension(m,n)::B
cf2py double precision intent(in), dimension(m,m)::A
cf2py optional character intent(in)::uplo='U'
cf2py optional character intent(in)::transa='N'
cf2py integer intent(hide), depend(A)::M=shape(A,0)
cf2py integer intent(hide), depend(B)::N=shape(B,1)


      
*     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA
      INTEGER LDA,LDB,M,N
      CHARACTER DIAG,SIDE,TRANSA,UPLO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(M,M),B(M,N)
      
      EXTERNAL DTRMM
! DTRMM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
*     alpha*op( A )*X or alpha*X*op(A)
      
      DIAG = 'N'
      LDA = M
      LDB = M
      SIDE = 'L'
      ALPHA=1.0D0
      
      CALL DTRMM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
      RETURN
      END
      

      SUBROUTINE dtrsm_wrap(M,N,A,B,UPLO,TRANSA,ALPHA)
cf2py double precision intent(inplace), dimension(m,n)::B
cf2py double precision intent(in), dimension(m,m)::A
cf2py optional character intent(in)::uplo='U'
cf2py optional character intent(in)::transa='N'
cf2py optional double precision intent(in)::alpha=1.0
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
      LDA = M
      LDB = M
      SIDE = 'L'

      
      CALL DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
      RETURN
      END


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

      SUBROUTINE gp_array_logp(x, mu, sig, n, like, info)

cf2py intent(copy) x, mu
cf2py intent(in) sig
cf2py intent(out) like
cf2py intent(hide) info, n

      DOUBLE PRECISION sig(n,n), x(n), mu(n), like
      INTEGER n, info, i
      DOUBLE PRECISION twopi_N, log_detC, gd
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)      
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      EXTERNAL DTRSV
! DTRSV ( UPLO, TRANS, DIAG, N, A, LDA, X, INCX )
!       EXTERNAL DPOTRS
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
!       call DCOPY(n,x,1,mu,1)
      
!     x <- sig ^-1 * x
!       call DPOTRS('L',n,1,sig,n,x,n,info)
      call DTRSV('U','T','N',n,sig,n,x,1)

      gd=0.0D0
      do i=1,n
          gd=gd+x(i)*x(i)
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


c
! 
!       SUBROUTINE dpotrs_wrap(chol_fac, b, info, n, m, uplo)
! 
! cf2py double precision dimension(n,n), intent(in) :: chol_fac
! cf2py integer depend(chol_fac), intent(hide)::n=shape(chol_fac,0)
! cf2py integer depend(b), intent(hide)::m=shape(b,1)
! cf2py optional character intent(in):: uplo='U'
! cf2py integer intent(out)::info
! cf2py double precision intent(inplace), dimension(n,m)::b
! 
!       DOUBLE PRECISION chol_fac(n,n), b(n,m)
!       INTEGER n, info,m
!       
!       CHARACTER uplo
! 
!       EXTERNAL DPOTRS
! ! DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) Solves triangular system
! 
!       call DPOTRS(uplo,n,m,chol_fac,n,b,s,info)
!       
!       return
!       END
