c A collection of covariance functions for Gaussian processes, implemented in Fortran.
c By convention, the first dimension of each input array iterates over points, and 
c subsequent dimensions iterate over spatial dimensions.

      SUBROUTINE matern(C,x,y,diff_degree,scale,amp,nx,ny,ndx,ndy,
     +symm,BI,DI,BK,DK,N)

cf2py double precision dimension(nx,ny),intent(out)::C
cf2py double precision dimension(nx,ndx),intent(in)::x
cf2py double precision dimension(ny,ndy),intent(in)::y
cf2py double precision intent(in)::diff_degree
cf2py double precision intent(in), optional::scale=1.0
cf2py double precision intent(in), optional::amp=1.0
cf2py integer intent(hide),depend(x)::nx=shape(x,0)
cf2py integer intent(hide),depend(y)::ny=shape(y,0)
cf2py integer intent(hide),depend(x)::ndx=shape(x,1)
cf2py integer intent(hide),depend(y)::ndy=shape(y,1)
cf2py logical intent(in), optional:: symm=0
cf2py double precision dimension(N), intent(hide) :: BI
cf2py double precision dimension(N), intent(hide) :: BK
cf2py double precision dimension(N), intent(hide) :: DI
cf2py double precision dimension(N), intent(hide) :: DK
cf2py integer intent(hide), depend(diff_degree) :: N = diff_degree+2

      DOUBLE PRECISION C(nx,ny), x(nx,ndx), y(ny,ndy)
      DOUBLE PRECISION scale, amp, diff_degree
      DOUBLE PRECISION GA, prefac, dev, VM, t
      INTEGER nx, ny, ndx, ndy, i, j, k, N
      DOUBLE PRECISION BI(N), BK(N), DI(N), DK(N)
      LOGICAL symm
      
      CALL GAMMA2(diff_degree,GA)
      prefac = 0.5D0 ** (diff_degree-1.0D0) / GA
      
      do i=1,nx

        if(symm) then            
          C(i,i)=amp
          do j=i+1,ny
            t = 0.0D0
            do k=1,ndx
              dev = x(i,k) - y(j,k)
              t = t + dev * dev
            enddo
            t = dsqrt(t) / scale
            if (t .EQ. 0.0) then
              C(i,j)=amp
            else
              CALL IKV(diff_degree,t,VM,BI,DI,BK,DK,N)
              C(i,j)=amp * prefac * (t ** diff_degree) * BK(N+1)
            endif
            C(j,i) = C(i,j)
          enddo

        else
          do j=1,ny
            t = 0.0D0
            do k=1,ndx
              dev = x(i,k) - y(j,k)
              t = t + dev * dev
            enddo
            t = dsqrt(t) / scale
            if (t .EQ. 0.0) then
              C(i,j)=amp
            else
              CALL IKV(diff_degree,t,VM,BI,DI,BK,DK,N)         
              C(i,j)=amp * prefac * (t ** diff_degree) * BK(N+1)
            endif
          enddo    
        endif
          
      enddo
      END

      SUBROUTINE axi_gauss(C,x,y,scale,amp,nx,ny,ndx,ndy,symm)

cf2py double precision dimension(nx,ny),intent(out)::C
cf2py double precision dimension(nx,ndx),intent(in)::x
cf2py double precision dimension(ny,ndy),intent(in)::y
cf2py double precision intent(in), optional::scale=1.
cf2py double precision intent(in), optional::amp=1.
cf2py integer intent(hide),depend(x)::nx=shape(x,0)
cf2py integer intent(hide),depend(y)::ny=shape(y,0)
cf2py integer intent(hide),depend(x)::ndx=shape(x,1)
cf2py integer intent(hide),depend(y)::ndy=shape(y,1)
cf2py logical intent(in), optional:: symm=0


      INTEGER nx,ny,i,j,ndx,ndy,k
      DOUBLE PRECISION C(nx,ny),x(nx,ndx),y(ny,ndy)
      DOUBLE PRECISION amp,scale,val_now,dev_now
      LOGICAL symm
      
      if(symm) then
        C(i,i)=amp
        do i=1,nx
          do j=i+1,ny
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

cf2py double precision dimension(nx,ny),intent(out)::C
cf2py double precision dimension(nx,ndx),intent(in)::x
cf2py double precision dimension(ny,ndy),intent(in)::y
cf2py double precision intent(in), optional::scale = 1.
cf2py double precision intent(in), optional::amp = 1.
cf2py double precision intent(in)::pow
cf2py integer intent(hide),depend(x)::nx=shape(x,0)
cf2py integer intent(hide),depend(y)::ny=shape(y,0)
cf2py integer intent(hide),depend(x)::ndx=shape(x,1)
cf2py integer intent(hide),depend(y)::ndy=shape(y,1)
cf2py logical intent(in), optional:: symm=0


      INTEGER nx,ny,i,j,ndx,ndy,k
      DOUBLE PRECISION C(nx,ny),x(nx,ndx),y(ny,ndy)
      DOUBLE PRECISION amp,scale,val_now,dev_now,pow
      LOGICAL symm

      if(symm) then
        do i=1,nx
          C(i,i)=amp            
          do j=i+1,ny
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
      

C       *********************************
C       Following routines are from:
C
C       COMPUTATION OF SPECIAL FUNCTIONS
C 
C          Shanjie Zhang and Jianming Jin
C
C       Copyrighted but permission granted to use code in programs. 
C       Buy their book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
C
C       Code taken from specfun.f in scipy.special
C       *********************************

C       **********************************
        REAL*8 FUNCTION ENVJ(N,X)
        DOUBLE PRECISION X
        ENVJ=0.5D0*DLOG10(6.28D0*N)-N*DLOG10(1.36D0*X/N)
        RETURN
        END


C       **********************************

        SUBROUTINE GAMMA2(X,GA)
C
C       ==================================================
C       Purpose: Compute gamma function â(x)
C       Input :  x  --- Argument of â(x)
C                       ( x is not equal to 0,-1,-2,úúú)
C       Output:  GA --- â(x)
C       ==================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION G(26)
        PI=3.141592653589793D0
        IF (X.EQ.INT(X)) THEN
           IF (X.GT.0.0D0) THEN
              GA=1.0D0
              M1=X-1
              DO 10 K=2,M1
10               GA=GA*K
           ELSE
              GA=1.0D+300
           ENDIF
        ELSE
           R=1.0D0
           IF (DABS(X).GT.1.0D0) THEN
              Z=DABS(X)
              M=INT(Z)
              DO 15 K=1,M
15               R=R*(Z-K)
              Z=Z-M
           ELSE
              Z=X
           ENDIF
           DATA G/1.0D0,0.5772156649015329D0,
     &          -0.6558780715202538D0, -0.420026350340952D-1,
     &          0.1665386113822915D0,-.421977345555443D-1,
     &          -.96219715278770D-2, .72189432466630D-2,
     &          -.11651675918591D-2, -.2152416741149D-3,
     &          .1280502823882D-3, -.201348547807D-4,
     &          -.12504934821D-5, .11330272320D-5,
     &          -.2056338417D-6, .61160950D-8,
     &          .50020075D-8, -.11812746D-8,
     &          .1043427D-9, .77823D-11,
     &          -.36968D-11, .51D-12,
     &          -.206D-13, -.54D-14, .14D-14, .1D-15/
           GR=G(26)
           DO 20 K=25,1,-1
20            GR=GR*Z+G(K)
           GA=1.0D0/(GR*Z)
           IF (DABS(X).GT.1.0D0) THEN
              GA=GA*R
              IF (X.LT.0.0D0) GA=-PI/(X*GA*DSIN(PI*X))
           ENDIF
        ENDIF
        RETURN
        END

        INTEGER FUNCTION MSTA1(X,MP)
C
C       ===================================================
C       Purpose: Determine the starting point for backward  
C                recurrence such that the magnitude of    
C                Jn(x) at that point is about 10^(-MP)
C       Input :  x     --- Argument of Jn(x)
C                MP    --- Value of magnitude
C       Output:  MSTA1 --- Starting point   
C       ===================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        N0=INT(1.1D0*A0)+1
        F0=ENVJ(N0,A0)-MP
        N1=N0+5
        F1=ENVJ(N1,A0)-MP
        DO 10 IT=1,20             
           NN=N1-(N1-N0)/(1.0D0-F0/F1)                  
           F=ENVJ(NN,A0)-MP
           IF(ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
 10        F1=F
 20     MSTA1=NN
        RETURN
        END


        INTEGER FUNCTION MSTA2(X,N,MP)
C
C       ===================================================
C       Purpose: Determine the starting point for backward
C                recurrence such that all Jn(x) has MP
C                significant digits
C       Input :  x  --- Argument of Jn(x)
C                n  --- Order of Jn(x)
C                MP --- Significant digit
C       Output:  MSTA2 --- Starting point
C       ===================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        HMP=0.5D0*MP
        EJN=ENVJ(N,A0)
        IF (EJN.LE.HMP) THEN
           OBJ=MP
           N0=INT(1.1*A0)+1
        ELSE
           OBJ=HMP+EJN
           N0=N
        ENDIF
        F0=ENVJ(N0,A0)-OBJ
        N1=N0+5
        F1=ENVJ(N1,A0)-OBJ
        DO 10 IT=1,20
           NN=N1-(N1-N0)/(1.0D0-F0/F1)
           F=ENVJ(NN,A0)-OBJ
           IF (ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
10         F1=F
20      MSTA2=NN+10
        RETURN
        END


C       **********************************

        SUBROUTINE IKV(V,X,VM,BI,DI,BK,DK,N)

cf2py double precision intent(in):: V
cf2py double precision intent(in):: x
cf2py double precision intent(out):: VM
cf2py double precision dimension(N), intent(out):: BI
cf2py double precision dimension(N), intent(out):: DI
cf2py double precision dimension(N), intent(out):: BK
cf2py double precision dimension(N), intent(out):: DK
cf2py integer intent(in) :: N
        
C
C       =======================================================
C       Purpose: Compute modified Bessel functions Iv(x) and
C                Kv(x), and their derivatives
C       Input :  x --- Argument ( x ò 0 )
C                v --- Order of Iv(x) and Kv(x)
C                      ( v = n+v0, n = 0,1,2,..., 0 ó v0 < 1 )
C       Output:  BI(n) --- In+v0(x)
C                DI(n) --- In+v0'(x)
C                BK(n) --- Kn+v0(x)
C                DK(n) --- Kn+v0'(x)
C                VM --- Highest order computed
C       Routines called:
C            (1) GAMMA2 for computing the gamma function
C            (2) MSTA1 and MSTA2 to compute the starting
C                point for backward recurrence
C       =======================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION BI(0:*),DI(0:*),BK(0:*),DK(0:*)
        PI=3.141592653589793D0
        X2=X*X
        N=INT(V)
        V0=V-N
        IF (N.EQ.0) N=1
        IF (X.LT.1.0D-100) THEN
           DO 10 K=0,N
              BI(K)=0.0D0
              DI(K)=0.0D0
              BK(K)=-1.0D+300
10            DK(K)=1.0D+300
           IF (V.EQ.0.0) THEN
              BI(0)=1.0D0
              DI(1)=0.5D0
           ENDIF
           VM=V
           RETURN
        ENDIF
        PIV=PI*V0
        VT=4.0D0*V0*V0
        IF (V0.EQ.0.0D0) THEN
           A1=1.0D0
        ELSE
           V0P=1.0D0+V0
           CALL GAMMA2(V0P,GAP)
           A1=(0.5D0*X)**V0/GAP
        ENDIF
        K0=14
        IF (X.GE.35.0) K0=10
        IF (X.GE.50.0) K0=8
        IF (X.LE.18.0) THEN
           BI0=1.0D0
           R=1.0D0
           DO 15 K=1,30
              R=0.25D0*R*X2/(K*(K+V0))
              BI0=BI0+R
              IF (DABS(R/BI0).LT.1.0D-15) GO TO 20
15         CONTINUE
20         BI0=BI0*A1
        ELSE
           CA=DEXP(X)/DSQRT(2.0D0*PI*X)
           SUM=1.0D0
           R=1.0D0
           DO 25 K=1,K0
              R=-0.125D0*R*(VT-(2.0D0*K-1.0D0)**2.0)/(K*X)
25            SUM=SUM+R
           BI0=CA*SUM
        ENDIF
        M=MSTA1(X,200)
        IF (M.LT.N) THEN
           N=M
        ELSE
           M=MSTA2(X,N,15)
        ENDIF
        F=0.0D0
        F2=0.0D0
        F1=1.0D-100
        WW=0.0D0
        DO 30 K=M,0,-1
           F=2.0D0*(V0+K+1.0D0)/X*F1+F2
           IF (K.LE.N) BI(K)=F
           F2=F1
30         F1=F
        CS=BI0/F
        DO 35 K=0,N
35         BI(K)=CS*BI(K)
        DI(0)=V0/X*BI(0)+BI(1)
        DO 40 K=1,N
40         DI(K)=-(K+V0)/X*BI(K)+BI(K-1)
        IF (X.LE.9.0D0) THEN
           IF (V0.EQ.0.0D0) THEN
              CT=-DLOG(0.5D0*X)-0.5772156649015329D0
              CS=0.0D0
              W0=0.0D0
              R=1.0D0
              DO 45 K=1,50
                 W0=W0+1.0D0/K
                 R=0.25D0*R/(K*K)*X2
                 CS=CS+R*(W0+CT)
                 WA=DABS(CS)
                 IF (DABS((WA-WW)/WA).LT.1.0D-15) GO TO 50
45               WW=WA
50            BK0=CT+CS
           ELSE
              V0N=1.0D0-V0
              CALL GAMMA2(V0N,GAN)
              A2=1.0D0/(GAN*(0.5D0*X)**V0)
              A1=(0.5D0*X)**V0/GAP
              SUM=A2-A1
              R1=1.0D0
              R2=1.0D0
              DO 55 K=1,120
                 R1=0.25D0*R1*X2/(K*(K-V0))
                 R2=0.25D0*R2*X2/(K*(K+V0))
                 SUM=SUM+A2*R1-A1*R2
                 WA=DABS(SUM)
                 IF (DABS((WA-WW)/WA).LT.1.0D-15) GO TO 60
55               WW=WA
60            BK0=0.5D0*PI*SUM/DSIN(PIV)
           ENDIF
        ELSE
           CB=DEXP(-X)*DSQRT(0.5D0*PI/X)
           SUM=1.0D0
           R=1.0D0
           DO 65 K=1,K0
              R=0.125D0*R*(VT-(2.0*K-1.0)**2.0)/(K*X)
65            SUM=SUM+R
           BK0=CB*SUM
        ENDIF
        BK1=(1.0D0/X-BI(1)*BK0)/BI(0)
        BK(0)=BK0
        BK(1)=BK1
        DO 70 K=2,N
           BK2=2.0D0*(V0+K-1.0D0)/X*BK1+BK0
           BK(K)=BK2
           BK0=BK1
70         BK1=BK2
        DK(0)=V0/X*BK(0)-BK(1)
        DO 80 K=1,N
80         DK(K)=-(K+V0)/X*BK(K)-BK(K-1)
        VM=N+V0
        RETURN
        END
