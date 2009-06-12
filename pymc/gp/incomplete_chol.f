! Copyright (c) Anand Patil, 2007

      subroutine ichol_continue      
     *(n,nnew,sig,nnewpmold,m,diag,piv,reltol,x,ndim,rowfun)
cf2py intent(hide) nnew
cf2py intent(hide) n
cf2py intent(hide) nnewpmold
cf2py intent(hide) ndim
cf2py intent(copy) x
c
c m is the total rank of the matrix
cf2py intent(out) m
c
c sig will be updated in-place. 
c The first mold rows should be filled in on input.
cf2py intent(inplace) sig
c
c The pivot vector is are of size nnew
cf2py intent(in,out) piv
c
c The user has to tell it what the rank is so far.
cf2py intent(in) mold
        integer nnew, nold, nnewpmold, ptemp, ndim
        integer i, n, m, mold, piv(n), p(n), j
        double precision rowvec(n), diag(nnew), x(n,ndim)
        double precision sig(nnewpmold,n)
c rowfun is the 'get-row' function.
        external rowfun

        DOUBLE PRECISION maxdiag, tol, dtemp

        DOUBLE PRECISION ZERO, ONE, RELTOL, NEGONE
        PARAMETER (zero=0.0D0)
        PARAMETER (one=1.0D0)
        PARAMETER (negone = -1.0D0)
        
        EXTERNAL DGEMV
*       DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
*       Purpose
*       =======
*
*       DGEMV  performs one of the matrix-vector operations
*
*           y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
*
*       where alpha and beta are scalars, x and y are vectors and A is an
*       m by n matrix.
*
        EXTERNAL DSWAP
*       DSWAP(N,DX,INCX,DY,INCY)

        EXTERNAL IDAMAX
*       IDAMAX(N,DX,INCX)

        mold = nnewpmold - nnew
        nold = n - nnew

!       Make diagonal and index vectors
        do i=1,nnew
            itot = i + mold
            do j=1,mold
                diag(i)=diag(i)-sig(j,itot)*sig(j,itot)
            enddo
        enddo
        do i=1,n
            p(i) = piv(i) + 1
        enddo
        
        maxdiag = diag(idamax(nnew,diag,1))
      
        tol = maxdiag * reltol


!       Main loop
        do i=1,nnew
          
          
!         Find maximum remaining pivot
            l = idamax(nnew-i+1,diag(i),1)+i-1
            maxdiag = diag(l)
        
            itot = i+mold
            ltot = l+mold
         
!         Early return if there are no big pivots left
            if (maxdiag .LE. tol) then
                do j=1,n
                    piv(j) = p(j)-1
                enddo
                m = mold + i-1
                return
            endif

            if (i .NE. l) then
!                 Swap p and diag's elements i and l

                ptemp = p(itot)
                p(itot) = p(ltot)
                p(ltot) = ptemp
        
                dtemp = diag(i)
                diag(i) = diag(l)
                diag(l) = dtemp
        
!                 Swap the i and lth columns of sig

                CALL DSWAP(itot,sig(1,itot),1,sig(1,ltot),1)
!                 Also swap ith and lth rows of x  
                CALL DSWAP(ndim,x(itot,1),n,x(ltot,1),n)

!                 do j=1,ndim
!                     tmp = x(itot,j)
!                     x(itot,j) = x(ltot,j)
!                     x(ltot,j) = tmp
!                 enddo
                    
            endif
        
!             Write diagonal element
            sig(itot,itot) = dsqrt(diag(i))

!             Assemble the row vector
!               Try this: rowfun(itot, x, rowvec, n, ndim)
            if (itot.LT.n) then                
                call rowfun(itot,x,rowvec,n,ndim)
            endif


!               BLAS-less DGEMV might be useful if you ever do the sparse version.              
!               do j=itot+1,n
!                 do k=1,itot-1
!                   rowvec(j)=rowvec(j)-sig(k,j)*sig(k,itot)
!                 enddo
!               enddo              

!         Implement Cholesky algorithm.
        CALL DGEMV('T',itot-1,n-itot,negone,sig(1,itot+1),
     1                  nnew+mold,
     2                  sig(1,itot),
     3                  1,
     4                  one,rowvec(itot+1),1)
        if (itot.LT.n) then
          do j=itot+1,n
            sig(itot,j) = rowvec(j) / sig(itot,itot)
          enddo
          do j=i+1,nnew
            jtot = j+mold
            diag(j) = diag(j) - sig(itot,jtot)*sig(itot,jtot)
          enddo
          
        endif
        
        enddo
        
      
        do i=1,n
            piv(i)=p(i)-1
        enddo        

        
        m = mold + nnew
        return
        end

      subroutine ichol(n,sig,m,diag,piv,reltol,x,ndim,rowfun,rl)

cf2py intent(hide) n, ndim
cf2py intent(out) m,sig,piv
cf2py intent(copy) x
cf2py intent(in) rl

        integer i, n, p(n), m, piv(n), ptemp, ndim, rl
        double precision rowvec(n), diag(n), sig(rl,n), x(n,ndim)
c rowfun is the get-row function
        external rowfun
        
        DOUBLE PRECISION maxdiag, tol, dtemp

        DOUBLE PRECISION ZERO, ONE, RELTOL, NEGONE
        PARAMETER (zero=0.0D0)
        PARAMETER (one=1.0D0)
        PARAMETER (negone = -1.0D0)
        
        EXTERNAL DGEMV
*       DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
*       Purpose
*       =======
*
*       DGEMV  performs one of the matrix-vector operations
*
*           y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
*
*       where alpha and beta are scalars, x and y are vectors and A is an
*       m by n matrix.
*
        EXTERNAL DSWAP
*       DSWAP(N,DX,INCX,DY,INCY)

        EXTERNAL IDAMAX
*       IDAMAX(N,DX,INCX)
        
!       Make diagonal and index vectors
        do i=1,n
            p(i)=i
        enddo
      
        maxdiag = diag(idamax(n,diag,1))
      
        tol = maxdiag * reltol
        m = rl


!       Main loop
        do i=1,rl

!         Find maximum remaining pivot
            l = idamax(n-i+1,diag(i),1)+i-1
            maxdiag = diag(l)
        

!         Early return if there are no big pivots left
            if (maxdiag .LE. tol) then
                do j=1,n
                    piv(j) = p(j)-1
                enddo
                m = i-1
                return
            endif
            

            if (i .NE. l) then
!         Swap p and diag's elements i and l

            ptemp = p(i)
            p(i) = p(l)
            p(l) = ptemp
        
            dtemp = diag(i)
            diag(i) = diag(l)
            diag(l) = dtemp
        
!         Swap the i and lth columns of sig
            CALL DSWAP(i,sig(1,i),1,sig(1,l),1)
            
!         Swap the i and lth rows of x
*       DSWAP(N,DX,INCX,DY,INCY)
            CALL DSWAP(ndim,x(i,1),n,x(l,1),n)
            endif

!       Write diagonal element
        sig(i,i) = dsqrt(diag(i))

!       Assemble the row vector.
!         Try this: rowfun(i, x, rowvec, n, ndim)
        if (i.LT.n) then

            call rowfun(i,x,rowvec,n,ndim)

!             do j=i+1,n
!               rowvec(j) = c(p(i),p(j))
!             enddo   

        endif

        if (i.GT.1) then

!               BLAS-less DGEMV might be useful if you ever do the sparse version.              
!               do j=i+1,n
!                 do k=1,i-1
!                   rowvec(j)=rowvec(j)-sig(k,j)*sig(k,i)
!                 enddo
!               enddo              

!         Implement Cholesky algorithm.
        CALL DGEMV('T',i-1,n-i,negone,sig(1,i+1),
     1                  rl,
     2                  sig(1,i),
     3                  1,
     4                  one,rowvec(i+1),1)

        endif

        if (i.LT.n) then
          do j=i+1,n
            sig(i,j) = rowvec(j) / sig(i,i)
            diag(j) = diag(j) - sig(i,j)*sig(i,j)
          enddo
        endif

      enddo
      
      do i=1,n
        piv(i)=p(i)-1
      enddo


      return
      end
      
      subroutine ichol_basis(basis,nb,n_nug,n,sig,p,m,nug,reltol)

c Incomplete cholesky factorization
c Author: Anand Patil
c Date: May 6, 2007
c Port of mex function chol_incomplete.c by Matthias Seeger
c http://www.kyb.tuebingen.mpg.de/bs/people/seeger/
c
cf2py intent(out) sig
cf2py intent(out) p
cf2py intent(hide) rowvec
cf2py intent(hide) diag
cf2py intent(out) m
cf2py intent(hide) nb
cf2py intent(hide) n_nug
cf2py intent(hide) n

      DOUBLE PRECISION basis(nb,n), sig(n,n), diag(n), nug(n_nug)
      DOUBLE PRECISION rowvec(n)
      integer p(n), n, m, nb, i, j
      DOUBLE PRECISION maxdiag, tol, dtemp
      
      EXTERNAL DGEMV
      DOUBLE PRECISION ZERO, ONE, RELTOL, NEGONE
      PARAMETER (zero=0.0D0)
      PARAMETER (one=1.0D0)
      PARAMETER (negone = -1.0D0)
      
* DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
*  Purpose
*  =======
*
*  DGEMV  performs one of the matrix-vector operations
*
*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
*
*  where alpha and beta are scalars, x and y are vectors and A is an
*  m by n matrix.
*
      EXTERNAL DSWAP
* DSWAP(N,DX,INCX,DY,INCY)

      EXTERNAL IDAMAX
* IDAMAX(N,DX,INCX)

!       Make diagonal and index vectors
      do i=1,n
          if (n_nug.EQ.1) then
              diag(i) = nug(1)
          else
              diag(i) = nug(i)
          end if
          do j=1,nb
              diag(i) = diag(i) + basis(j,i) ** 2
          enddo
        p(i)=i
      enddo
      
      maxdiag = diag(idamax(n,diag,1))
      
      tol = maxdiag * reltol
      m = n
!       Main loop
      do i=1,n
          
!         Find maximum remaining pivot
        l = idamax(n-i+1,diag(i),1)+i-1
        maxdiag = diag(l)
        
        
!         Early return if there are no big pivots left
        if (maxdiag .LE. tol) then
          do j=1,n
            p(j) = p(j)-1
          enddo
          m = i-1
          return
        endif

        if (i .NE. l) then
!         Swap p and diag's elements i and l

          itemp = p(i)
          p(i) = p(l)
          p(l) = itemp
        
          dtemp = diag(i)
          diag(i) = diag(l)
          diag(l) = dtemp
        
!         Swap the i and lth columns of sig
          CALL DSWAP(i,sig(1,i),1,sig(1,l),1)
          
!           Swap ith and lth columns of the basis
          CALL DSWAP(nb,basis(1,i),1,basis(1,l),1)
        endif
        
!       Write diagonal element
        sig(i,i) = dsqrt(diag(i))

!       Assemble the row vector
        if (i.LT.n) then

* DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
            CALL DGEMV('T',nb,n-i,one,basis(1,i+1),
     1                  nb,
     2                  basis(1,i),
     3                  1,
     4                  zero,rowvec(i+1),1)

!         BLAS-less DGEMV
!         do j=(i+1),n
!             rowvec(j) = zero
!             do k=1,nb
!                 rowvec(j)=rowvec(j)+basis(k,i)*basis(k,j)
!             end do
!         end do
        endif

          if (i.GT.1) then

!               BLAS-less DGEMV might be useful if you ever do the sparse version.              
!               do j=i+1,n
!                 do k=1,i-1
!                   rowvec(j)=rowvec(j)-sig(k,j)*sig(k,i)
!                 enddo
!               enddo              

!         Implement Cholesky algorithm.
            CALL DGEMV('T',i-1,n-i,negone,sig(1,i+1),
     1                  n,
     2                  sig(1,i),
     3                  1,
     4                  one,rowvec(i+1),1)
          endif

        if (i.LT.n) then
          do j=i+1,n
            sig(i,j) = rowvec(j) / sig(i,i)
            diag(j) = diag(j) - sig(i,j)*sig(i,j)
          enddo
        endif

      enddo
      
      do i=1,n
        p(i)=p(i)-1
      enddo
      
      return
      end
      



      subroutine ichol_full(c,n,sig,m,p,rowvec,diag,reltol)
c
c Incomplete cholesky factorization
c Author: Anand Patil
c Date: May 6, 2007
c Port of mex function chol_incomplete.c by Matthias Seeger
c http://www.kyb.tuebingen.mpg.de/bs/people/seeger/
c
cf2py double precision dimension(n,n), intent(in)::c
cf2py double precision dimension(n,n), intent(out)::sig
cf2py integer dimension(n), intent(out)::p
cf2py double precision dimension(n), intent(hide)::rowvec
cf2py double precision dimension(n), intent(hide)::diag
cf2py integer intent(hide), depend(c):: n = shape(c,0)
cf2py integer intent(out)::m
cf2py double precision intent(in) :: reltol
cf2py threadsafe

      DOUBLE PRECISION c(n,n), sig(n,n), diag(n)
      DOUBLE PRECISION rowvec(n)
      integer p(n), n, m, i, j
      DOUBLE PRECISION maxdiag, tol, dtemp
      
      EXTERNAL DGEMV
      DOUBLE PRECISION ZERO, ONE, RELTOL, NEGONE
      PARAMETER (zero=0.0D0)
      PARAMETER (one=1.0D0)
      PARAMETER (negone = -1.0D0)
      
* DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
*  Purpose
*  =======
*
*  DGEMV  performs one of the matrix-vector operations
*
*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
*
*  where alpha and beta are scalars, x and y are vectors and A is an
*  m by n matrix.
*
      EXTERNAL DSWAP
* DSWAP(N,DX,INCX,DY,INCY)

      EXTERNAL IDAMAX
* IDAMAX(N,DX,INCX)

!       Make diagonal and index vectors
      do i=1,n
        diag(i) = c(i,i)
        p(i)=i
      enddo
      
      maxdiag = diag(idamax(n,diag,1))
      
      tol = maxdiag * reltol
      m = n
!       Main loop
      do i=1,n
          
!         Find maximum remaining pivot
        l = idamax(n-i+1,diag(i),1)+i-1
        maxdiag = diag(l)
        
        
!         Early return if there are no big pivots left
        if (maxdiag .LE. tol) then
          do j=1,n
            p(j) = p(j)-1
          enddo
          m = i-1
          return
        endif

        if (i .NE. l) then
!         Swap p and diag's elements i and l

          itemp = p(i)
          p(i) = p(l)
          p(l) = itemp
        
          dtemp = diag(i)
          diag(i) = diag(l)
          diag(l) = dtemp
        
!         Swap the i and lth columns of sig
          CALL DSWAP(i,sig(1,i),1,sig(1,l),1)
        endif
        
!       Write diagonal element
        sig(i,i) = dsqrt(diag(i))

!       Assemble the row vector
        if (i.LT.n) then
            do j=i+1,n
              rowvec(j) = c(p(i),p(j))
            enddo   
        endif

          if (i.GT.1) then

!               BLAS-less DGEMV might be useful if you ever do the sparse version.              
!               do j=i+1,n
!                 do k=1,i-1
!                   rowvec(j)=rowvec(j)-sig(k,j)*sig(k,i)
!                 enddo
!               enddo              

!         Implement Cholesky algorithm.
            CALL DGEMV('T',i-1,n-i,negone,sig(1,i+1),
     1                  n,
     2                  sig(1,i),
     3                  1,
     4                  one,rowvec(i+1),1)
          endif

        if (i.LT.n) then
          do j=i+1,n
            sig(i,j) = rowvec(j) / sig(i,i)
            diag(j) = diag(j) - sig(i,j)*sig(i,j)
          enddo
        endif

      enddo
      
      do i=1,n
        p(i)=p(i)-1
      enddo
      
      return
      end
      
