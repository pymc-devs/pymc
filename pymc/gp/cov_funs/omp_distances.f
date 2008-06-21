! Copyright (c) Anand Patil, 2007

      SUBROUTINE set_b(nx,nt,i,b)
      integer nx, nt, i, b(2)
      
      
        b(1)=int(sqrt(real(i)/real(nt))*real(nx))+1      
        b(2)=int(sqrt(real(i+1)/real(nt))*real(nx))
      
      return
      END


      SUBROUTINE symmetrize(D,n)
     
cf2py intent(hide) n
cf2py intent(inplace) D
      
      DOUBLE PRECISION D(n,n)
      INTEGER i,j,n
      
      do j=1,n
          do i=1,j-1
              D(j,i) = D(i,j)
          end do
      end do
      
      return
      END


      SUBROUTINE euclidean(D,x,y,nx,ny,ndx,ndy,symm)

cf2py threadsafe
cf2py double precision dimension(nx,ny), intent(out)::D
cf2py double precision dimension(nx,ndx), intent(in)::x
cf2py double precision dimension(ny,ndy), intent(in)::y 
cf2py logical intent(in), optional:: symm=0
cf2py integer intent(hide), depend(x)::nx=shape(x,0)
cf2py integer intent(hide), depend(y)::ny=shape(y,0)
cf2py integer intent(hide), depend(x)::ndx=shape(x,1)
cf2py integer intent(hide), depend(y)::ndy=shape(y,1)

      DOUBLE PRECISION D(nx,ny), x(nx,ndx), y(ny,ndy)
      integer nx,ny,ndx,ndy,i,j,k
      LOGICAL symm
      DOUBLE PRECISION dist, dev
      INTEGER nt,oppt,nmt,b(2)
      INTEGER OMP_GET_MAX_THREADS
      INTEGER OMP_GET_THREAD_NUM
      EXTERNAL OMP_SET_NUM_THREADS


      oppt=50000
      ntm=OMP_GET_MAX_THREADS()
      if (symm) then
        ntm=min(int(real(nx)*real(ny)/real(oppt)),ntm)
      else
        ntm=min(int(real(nx)*real(ny)/real(oppt))/2,ntm)
      end if
      nt=max(ntm,1)
      
      CALL OMP_SET_NUM_THREADS(nt)
      

!$OMP  PARALLEL
!$OMP& DEFAULT(SHARED) PRIVATE(i,j,k,dist,dev,b)

      if (symm) then
          
      j=OMP_GET_THREAD_NUM()
      CALL set_b(nx,nt,j,b)
          
      do j=b(1),b(2)

        D(j,j)=0.0D0            

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

!$OMP  DO SCHEDULE(STATIC)
        do j=1,ny
          do i=1,nx
            dist = 0.0D0
            do k=1,ndx
              dev=(x(i,k) - y(j,k))
              dist = dist + dev*dev
            enddo
            D(i,j) = dsqrt(dist)
          enddo    
        enddo  
!$OMP  END DO NOWAIT

      endif

!$OMP  END PARALLEL

      if (symm) then
        CALL symmetrize(D,nx)
      end if

      CALL OMP_SET_NUM_THREADS(ntm)
           
      RETURN
      END



      SUBROUTINE geographic(D,x,y,nx,ny,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py threadsafe
cf2py double precision intent(out), dimension(nx,ny) :: D
cf2py double precision intent(in), dimension(nx,2) :: x
cf2py double precision intent(in), dimension(ny,2) :: y
cf2py logical intent(in), optional :: symm = 0
cf2py integer intent(hide), depend(x)::nx=shape(x,0)
cf2py integer intent(hide), depend(y)::ny=shape(y,0)

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,j,i,i_hi
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      INTEGER nt,oppt,nmt,b(2)
      INTEGER OMP_GET_MAX_THREADS
      INTEGER OMP_GET_THREAD_NUM
      EXTERNAL OMP_SET_NUM_THREADS

      oppt=50000
      ntm=OMP_GET_MAX_THREADS()
      if (symm) then
        ntm=min(int(real(nx)*real(ny)/real(oppt)),ntm)
      else
        ntm=min(int(real(nx)*real(ny)/real(oppt))/2,ntm)
      end if
      nt=max(ntm,1)
      
      CALL OMP_SET_NUM_THREADS(nt)
      

!$OMP  PARALLEL
!$OMP& DEFAULT(SHARED) 
!$OMP& PRIVATE(i,j,k,clat1,clat2,dlat,dlon,a,sterm,cterm,b)

      if (symm) then
          
      j=OMP_GET_THREAD_NUM()
      CALL set_b(nx,nt,j,b)
          
      do j=b(1),b(2)
        clat2 = dcos(y(j,2))
        D(j,j)=0.0D0            

        do i=1,j-1            
            clat1 = dcos(x(i,2))
            dlat = (x(i,2)-y(j,2))*0.5D0
            dlon = (x(i,1)-y(j,1))*0.5D0
            a=dsin(dlat)**2 + clat1*clat2*dsin(dlon)**2
            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)    
        enddo          
      enddo
   
      else
      
!$OMP DO SCHEDULE(STATIC)
      do j=1,ny
        clat2 = dcos(y(j,2))
        D(j,j)=0.0D0            

        do i=1,j-1            
            clat1 = dcos(x(i,2))
            dlat = (x(i,2)-y(j,2))*0.5D0
            dlon = (x(i,1)-y(j,1))*0.5D0
            a=dsin(dlat)**2 + clat1*clat2*dsin(dlon)**2
            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)    

        enddo          
      enddo
!$OMP  END DO NOWAIT

      end if
            
!$OMP  END PARALLEL

      if (symm) then
        CALL symmetrize(D,nx)
      end if

      CALL OMP_SET_NUM_THREADS(ntm)
      RETURN
      END

      
      SUBROUTINE paniso_geo_rad(D,x,y,nx,ny,ctrs,scals,na,symm)

cf2py threadsafe      
cf2py intent(out) D
cf2py logical intent(optional) :: symm=0
cf2py intent(hide) na
cf2py intent(hide) nx
cf2py intent(hide) ny
      
      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)   
      DOUBLE PRECISION ctrs(na), scals(na), w       
      integer nx,ny,i,j,na,i_hi                 
      LOGICAL symm                                  
      DOUBLE PRECISION a,pi,da,dlon,dlat
      INTEGER nt,oppt,nmt,b(2)
      INTEGER OMP_GET_MAX_THREADS
      INTEGER OMP_GET_THREAD_NUM
      EXTERNAL OMP_SET_NUM_THREADS
      PARAMETER (pi=3.141592653589793238462643d0)   

      CALL geographic(D,x,y,nx,ny,symm)      
      w = 0.5D0/real(na)
      do k=1,na
          ctrs(k) = ctrs(k)/pi
      end do


      oppt=50000
      ntm=OMP_GET_MAX_THREADS()
      if (symm) then
        ntm=min(int(real(nx)*real(ny)/real(oppt)),ntm)
      else
        ntm=min(int(real(nx)*real(ny)/real(oppt))/2,ntm)
      end if
      nt=max(ntm,1)
      
      CALL OMP_SET_NUM_THREADS(nt)
      
!$OMP  PARALLEL
!$OMP& DEFAULT(SHARED) 
!$OMP& PRIVATE(i,j,k,i_hi,a,da,dlon,dlat,b)

      if (symm) then
          
      j=OMP_GET_THREAD_NUM()
      CALL set_b(nx,nt,j,b)
          
      do j=b(1),b(2)
        D(j,j)=0.0D0            

        do i=1,j-1      
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
        enddo
      enddo  


      else
!$OMP DO SCHEDULE(STATIC)
      do j=1,ny
      clat2 = dcos(y(j,2))
      slat2 = dsin(y(j,2))

        do i=1,nx
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
                      go to 2
                  end if
              enddo
2         continue        
          D(j,i) = D(i,j)
          end if
      enddo
      enddo  
!$OMP END DO
      end if
     
!$OMP END PARALLEL    

      if (symm) then
        CALL symmetrize(D,nx)
      end if

      CALL OMP_SET_NUM_THREADS(ntm)
      
      RETURN
      END
      

      SUBROUTINE aniso_geo_rad(D,x,y,nx,ny,inc,ecc,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py threadsafe
cf2py intent(out) D
cf2py logical intent(optional) :: symm = 0
cf2py intent(hide) nx
cf2py intent(hide) ny

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,i,j,i_hi
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      DOUBLE PRECISION slat1, slat2, inc, ecc, theta, dtheta
      INTEGER nt,oppt,nmt,b(2)
      INTEGER OMP_GET_MAX_THREADS
      INTEGER OMP_GET_THREAD_NUM
      EXTERNAL OMP_SET_NUM_THREADS

      oppt=50000
      ntm=OMP_GET_MAX_THREADS()
      if (symm) then
        ntm=min(int(real(nx)*real(ny)/real(oppt)),ntm)
      else
        ntm=min(int(real(nx)*real(ny)/real(oppt))/2,ntm)
      end if
      nt=max(ntm,1)
      
      CALL OMP_SET_NUM_THREADS(nt)

!$OMP  PARALLEL
!$OMP& DEFAULT(SHARED) 
!$OMP& PRIVATE(i,j,k,clat1,clat2,slat1,slat2,dlat,
!$OMP& dlon,a,sterm,cterm,i_hi,theta,dtheta)

      if (symm) then
          
      j=OMP_GET_THREAD_NUM()
      CALL set_b(nx,nt,j,b)
          
      do j=b(1),b(2)
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
            
        enddo          
      enddo
      
      
      else
      
!$OMP DO SCHEDULE(STATIC)
      do j=1,ny
        clat2 = dcos(y(j,2))
        slat2 = dsin(y(j,2))
        D(j,j)=0.0D0            

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
!$OMP END DO NOWAIT

      endif

!$OMP  END PARALLEL      

      if (symm) then
        CALL symmetrize(D,nx)
      end if

      CALL OMP_SET_NUM_THREADS(ntm)

      RETURN
      END
