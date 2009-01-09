! Copyright (c) Anand Patil, 2007

      SUBROUTINE euclidean(D,x,y,nx,ny,ndx,ndy,cmin,cmax,symm)

cf2py intent(inplace) D
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py logical intent(optional) :: symm=0
cf2py intent(hide) nx, ny,ndx,ndy
cf2py threadsafe

      DOUBLE PRECISION D(nx,ny), x(nx,ndx), y(ny,ndy)
      INTEGER nx,ny,i,j,k,cmin,cmax,ndx,ndy
      LOGICAL symm
      DOUBLE PRECISION dist, dev

      if (cmax.EQ.-1) then
          cmax = ny
      end if
!       print *,nx,ny,ndx,ndy,cmin,cmax,symm      

      if(symm) then

        do j=cmin+1,cmax
          D(j,j) = 0.0D0
          do i=1,j-1
            dist = 0.0D0
            do k=1,ndx
              dev=(x(i,k) - y(j,k))
              dist = dist + dev*dev
            enddo
            D(i,j) = dsqrt(dist)
          enddo
        enddo
      else
        do j=cmin+1,cmax
          do i=1,nx
            dist = 0.0D0
            do k=1,ndx
              dev=(x(i,k) - y(j,k))
              dist = dist + dev*dev
            enddo
            D(i,j) = dsqrt(dist)
          enddo    
        enddo  
      endif
      RETURN
      END



      SUBROUTINE geographic(D,x,y,nx,ny,cmin,cmax,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py logical intent(optional) :: symm=0
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py intent(hide) nx, ny
cf2py intent(inplace) D
cf2py threadsafe

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,j,i,i_hi,cmin,cmax
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      do j=cmin+1,cmax
        clat2 = dcos(y(j,2))
        if(symm) then
            D(j,j)=0.0D0            
            i_hi = j-1
        else 
            i_hi = nx
        endif
        
        do i=1,i_hi
            clat1 = dcos(x(i,2))
            dlat = (x(i,2)-y(j,2))*0.5D0
            dlon = (x(i,1)-y(j,1))*0.5D0
            a=dsin(dlat)**2 + clat1*clat2*dsin(dlon)**2
            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)    
!             if(symm) then                  
!                 D(j,i) = D(i,j)
!             end if
        enddo          
      enddo
      RETURN
      END

      
      SUBROUTINE paniso_geo_rad(D,x,y,nx,ny,cmin,cmax
     *,ctrs,scals,na,symm)

cf2py intent(inplace) D
cf2py logical intent(optional) :: symm=0
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py intent(hide) na, nx, ny
cf2py threadsafe
      
      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)   
      DOUBLE PRECISION ctrs(na), scals(na), w       
      integer nx,ny,i,j,na,i_hi,cmin,cmax                 
      LOGICAL symm                                  
      DOUBLE PRECISION a,pi,da,dlon,dlat
      PARAMETER (pi=3.141592653589793238462643d0)   
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      CALL geographic(D,x,y,nx,ny,cmin,cmax,symm)      
      w = 0.5D0/real(na)
      do k=1,na
          ctrs(k) = ctrs(k)/pi
      end do

      do j=cmin+1,cmax
        if(symm) then
            D(j,j)=0.0D0            
            i_hi = j-1
        else 
            i_hi = nx
        endif          
        do i = 1,i_hi
            if (D(i,j).GT.0.0D0) then
                dlat = (x(i,2)-y(j,2))
                dlon = (x(i,1)-y(j,1))
                a=dsqrt(dlon*dlon+dlat*dlat)

                theta = DATAN2(dlat/a,dlon/a)/pi

                do k=1,na
                    da=theta-ctrs(k)
                    do while (da.LT.0.0D0)
                        da = da + 2.0D0
                    end do
                    if ((da.LE.w).OR.
     *              (da.GT.2.0D0-w).OR.
     *              ((da.GT.1.0D0).AND.(da.LE.1.0D0+w)).OR.
     *              ((da.GT.1.0D0-w).AND.(da.LE.1.0D0))) then
                        D(i,j)=D(i,j)/scals(k)
                        go to 1
                    end if
                enddo

  1         continue        
            end if
!             if(symm) then                  
!                 D(j,i) = D(i,j)
!             end if  
        enddo
      enddo  
           
      RETURN
      END
      

c
      SUBROUTINE aniso_geo_rad(D,x,y,nx,ny,cmin,cmax,inc,ecc,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py intent(inplace) D
cf2py logical intent(optional) :: symm = 0
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py intent(hide) nx
cf2py intent(hide) ny
cf2py threadsafe

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,i,j,cmin,cmax
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      DOUBLE PRECISION slat1, slat2, inc, ecc, theta, dtheta
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      if (symm) then
          
      do j=cmin+1,cmax
        clat2 = dcos(y(j,2))
        slat2 = dsin(y(j,2))
        D(j,j)=0.0D0            

        do i=1,j-1
            clat1 = dcos(x(i,2))
            slat1 = dsin(x(i,2))
            dlat = (x(i,2)-y(j,2))
            dlon = (x(i,1)-y(j,1))
            a=dsin(dlat*0.5D0)**2 + clat1*clat2*dsin(dlon*0.5D0)**2
            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)


            if (D(i,j).GT.0.0D0) then

                a=dsqrt(dlon*dlon+dlat*dlat)
                theta = DATAN2(dlat/a,dlon/a)
                
                dtheta = theta-inc                
                dtheta = dcos(dtheta)              
                dtheta=ecc*ecc*dtheta*dtheta
                D(i,j)=D(i,j)*dsqrt(1.0D0 - dtheta)

            end if
            
!             D(j,i) = D(i,j)
        enddo          
      enddo
      
      else
      
      do j=cmin+1,cmax
         
        clat2 = dcos(y(j,2))
        slat2 = dsin(y(j,2))

        do i=1,nx

            clat1 = dcos(x(i,2))
            slat1 = dsin(x(i,2))
            dlat = (x(i,2)-y(j,2))
            dlon = (x(i,1)-y(j,1))
            a=dsin(dlat*0.5D0)**2 + clat1*clat2*dsin(dlon*0.5D0)**2

            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)

            if (D(i,j).GT.0.0D0) then

                a=dsqrt(dlon*dlon+dlat*dlat)
                theta = DATAN2(dlat/a,dlon/a)
                
                dtheta = theta-inc                
                dtheta = dcos(dtheta)              
                dtheta=ecc*ecc*dtheta*dtheta
                D(i,j)=D(i,j)*dsqrt(1.0D0 - dtheta)

            end if

        enddo          
      enddo

      endif

      RETURN
      END
