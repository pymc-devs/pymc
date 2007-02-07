c A collection of covariance functions for Gaussian processes, implemented in Fortran.
c By convention, the first dimension of each input array iterates over points, and 
c subsequent dimensions iterate over spatial dimensions.

      SUBROUTINE axi_gauss(a,pts_x,pts_y,symm,scale,amp,nx,ny,ndx,ndy)

cf2py real dimension(nx,ny),intent(inplace)::a
cf2py real dimension(nx,ndx),intent(in)::pts_x
cf2py real dimension(ny,ndy),intent(in)::pts_y
cf2py logical intent(in),optional::symm
cf2py real intent(in)::scale
cf2py real intent(in)::amp
cf2py integer intent(hide),depend(a)::nx=shape(a,0)
cf2py integer intent(hide),depend(a)::ny=shape(a,1)
cf2py integer intent(hide),depend(pts_x)::ndx=shape(pts_x,1)
cf2py integer intent(hide),depend(pts_y)::ndy=shape(pts_y,1)


      INTEGER nx,ny,i,j,ndx,ndy,k
      REAL a(nx,ny),pts_x(nx,ndx),pts_y(ny,ndy)
      REAL amp,scale,val_now,dev_now
      LOGICAL symm

      if(symm) then
        do i=1,nx
          do j=i,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (pts_x(i,k)-pts_y(j,k))*scale
              val_now = val_now + dev_now * dev_now
            enddo
            a(i,j) = amp * exp(-val_now/2.0) 
            a(j,i) = a(i,j)
          enddo
        enddo
      else
        do i=1,nx
          do j=1,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (pts_x(i,k)-pts_y(j,k))*scale
              val_now = val_now + dev_now * dev_now
            enddo
            a(i,j) = amp * exp(-val_now/2.0) 
          enddo
        enddo
      endif

      return
      END

      SUBROUTINE axi_exp(a,pts_x,pts_y,symm,scale,amp,pow,nx,ny,ndx,ndy)

cf2py real dimension(nx,ny),intent(inplace)::a
cf2py real dimension(nx,ndx),intent(in)::pts_x
cf2py real dimension(ny,ndy),intent(in)::pts_y
cf2py logical intent(in),optional::symm
cf2py real intent(in)::scale
cf2py real intent(in)::amp
cf2py real intent(in)::pow
cf2py integer intent(hide),depend(a)::nx=shape(a,0)
cf2py integer intent(hide),depend(a)::ny=shape(a,1)
cf2py integer intent(hide),depend(pts_x)::ndx=shape(pts_x,1)
cf2py integer intent(hide),depend(pts_y)::ndy=shape(pts_y,1)


      INTEGER nx,ny,i,j,ndx,ndy,k
      REAL a(nx,ny),pts_x(nx,ndx),pts_y(ny,ndy)
      REAL amp,scale,val_now,dev_now,pow
      LOGICAL symm

      if(symm) then
        do i=1,nx
          do j=i,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (pts_x(i,k)-pts_y(j,k))*scale
              val_now = val_now + dev_now * dev_now
            enddo
            a(i,j) = amp * exp(-val_now**(pow/2.0)/2.0) 
            a(j,i) = a(i,j)
          enddo
        enddo
      else
        do i=1,nx
          do j=1,ny
            val_now = 0.0
            do k=1,ndx
              dev_now = (pts_x(i,k)-pts_y(j,k))*scale
              val_now = val_now + dev_now * dev_now
            enddo
            a(i,j) = amp * exp(-val_now**(pow/2.0)/2.0) 
          enddo
        enddo
      endif

      return
      END


