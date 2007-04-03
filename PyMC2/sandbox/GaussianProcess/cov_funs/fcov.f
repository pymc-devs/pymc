c A collection of covariance functions for Gaussian processes, implemented in Fortran.
c By convention, the first dimension of each input array iterates over points, and 
c subsequent dimensions iterate over spatial dimensions.

      SUBROUTINE axi_gauss(C,x,y,scale,amp,nx,ny,ndx,ndy,symm)

cf2py double precision dimension(nx,ny),intent(inplace)::C
cf2py double precision dimension(nx,ndx),intent(in)::x
cf2py double precision dimension(ny,ndy),intent(in)::y
cf2py double precision intent(in), optional::scale=1.
cf2py double precision intent(in), optional::amp=1.
cf2py integer intent(hide),depend(C)::nx=shape(C,0)
cf2py integer intent(hide),depend(C)::ny=shape(C,1)
cf2py integer intent(hide),depend(x)::ndx=shape(x,1)
cf2py integer intent(hide),depend(y)::ndy=shape(y,1)
cf2py logical intent(in), optional:: symm=0


      INTEGER nx,ny,i,j,ndx,ndy,k
      DOUBLE PRECISION C(nx,ny),x(nx,ndx),y(ny,ndy)
      DOUBLE PRECISION amp,scale,val_now,dev_now
      LOGICAL symm
      
      if(symm) then
        do i=1,nx
          do j=i,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (x(i,k)-y(j,k))/scale
              val_now = val_now + dev_now * dev_now
            enddo
            C(i,j) = amp * exp(-val_now/2.0) 
            C(j,i) = C(i,j)
          enddo
        enddo
      else
        do i=1,nx
          do j=1,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (x(i,k)-y(j,k))/scale
              val_now = val_now + dev_now * dev_now
            enddo
            C(i,j) = amp * exp(-val_now/2.0) 
          enddo
        enddo
      endif

      return
      END

      SUBROUTINE axi_exp(C,x,y,scale,amp,pow,nx,ny,ndx,ndy,symm)

cf2py double precision dimension(nx,ny),intent(inplace)::C
cf2py double precision dimension(nx,ndx),intent(in)::x
cf2py double precision dimension(ny,ndy),intent(in)::y
cf2py double precision intent(in), optional::scale = 1.
cf2py double precision intent(in), optional::amp = 1.
cf2py double precision intent(in)::pow
cf2py integer intent(hide),depend(C)::nx=shape(C,0)
cf2py integer intent(hide),depend(C)::ny=shape(C,1)
cf2py integer intent(hide),depend(x)::ndx=shape(x,1)
cf2py integer intent(hide),depend(y)::ndy=shape(y,1)
cf2py logical intent(in), optional:: symm=0


      INTEGER nx,ny,i,j,ndx,ndy,k
      DOUBLE PRECISION C(nx,ny),x(nx,ndx),y(ny,ndy)
      DOUBLE PRECISION amp,scale,val_now,dev_now,pow
      LOGICAL symm

      if(symm) then
        do i=1,nx
          do j=i,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (x(i,k)-y(j,k))/scale
              val_now = val_now + dev_now * dev_now
            enddo
            C(i,j) = amp * exp(-val_now**(pow/2.0)/2.0) 
            C(j,i) = C(i,j)
          enddo
        enddo
      else
        do i=1,nx
          do j=1,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (x(i,k)-y(j,k))/scale
              val_now = val_now + dev_now * dev_now
            enddo
            C(i,j) = amp * exp(-val_now**(pow/2.0)/2.0) 
          enddo
        enddo
      endif

      return
      END


