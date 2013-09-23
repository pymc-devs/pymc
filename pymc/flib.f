      SUBROUTINE symmetrize(C,nx,cmin,cmax)

cf2py intent(inplace) C
cf2py intent(hide) nx
cf2py integer intent(in), optional :: cmin=0
cf2py integer intent(in), optional :: cmax=-1
cf2py threadsafe


      DOUBLE PRECISION C(nx,nx)
      INTEGER nx, i, j, cmin, cmax

      if (cmax.EQ.-1) then
          cmax = nx
      end if

      do j=cmin,cmax
          do i=1,j-1
              C(j,i) = C(i,j)
          end do
      end do

      RETURN
      END

      SUBROUTINE logsum(x, nx, s)
cf2py intent(hide) nx
cf2py intent(out) s
cf2py threadsafe
      DOUBLE PRECISION x(nx), s, diff, li, infinity
      INTEGER nx, i
      PARAMETER (li=709.78271289338397)
      PARAMETER (infinity = 1.7976931348623157d308)

      s = x(1)

      do i=2,nx
          diff = x(i)-s
!          If sum so far is zero, start from here.
          if (s.LE.-infinity) then
              s = x(i)
!           If x(i) swamps the sum so far, ditch the sum so far.
          else if (diff.GE.li) then
              s = x(i)
          else
              s = s + dlog(1.0D0+dexp(x(i)-s))
          end if
      end do

      RETURN
      END

      SUBROUTINE logsum_cpx(x, nx, s)
cf2py intent(hide) nx
cf2py intent(out) s
cf2py threadsafe
      COMPLEX*16 x(nx), s
      DOUBLE PRECISION li, diff
      INTEGER nx, i
      PARAMETER (li=709.78271289338397)

      s = x(1)

      do i=2,nx
!           If x(i) swamps the sum so far, ditch the sum so far.
          diff = DBLE(x(i)-s)
          if (diff.GE.li) then
              s = x(i)
          else
              s = s + CDLOG((1.0D0,0.0D0)+CDEXP(x(i)-s))
          end if
      end do

      RETURN
      END



      DOUBLE PRECISION FUNCTION combinationln(n,k)

c Ln of the number of different combinations of n different things, taken k at a time.
c DH, 5.02.2007

      IMPLICIT NONE
      INTEGER n, k
      DOUBLE PRECISION factln

      combinationln= factln(n) - factln(k) - factln(n-k)

      END FUNCTION combinationln

      SUBROUTINE expand_triangular(d, f, nf, t, n)
!       d is diagonal,
!       f is flattened lower triangle in column-major format,
!       t is unflattened triangular output matrix.
!       nf must be n * (n-1) / 2
cf2py intent(hide) nf
cf2py intent(hide) n
cf2py intent(out) t
cf2py threadsafe
       DOUBLE PRECISION f(nf), t(n,n), d(n)
       INTEGER i_f,i_t,j_t
       i_f = 0
       do j_t = 1,n
           t(j_t,j_t) = d(j_t)
           do i_t = j_t+1, n
               i_f = i_f + 1
               t(i_t,j_t) = f(i_f)
            end do
       end do

       RETURN
       END

      SUBROUTINE mod_to_circle(x, nx, u, nu, l, nl, mx)

cf2py intent(hide) nx, nu, nl
cf2py intent(out) mx

      DOUBLE PRECISION x(nx), u(nu), l(nl), mx(nx)
      DOUBLE PRECISION hi, lo, xi
      INTEGER nx, nu, nl, i

      lo = l(1)
      hi = u(1)
      do i=1,nx
        if (nl .NE. 1) lo = l(i)
        if (nu .NE. 1) hi = u(i)
        xi = x(i)
        if (xi < lo) then
            xi = hi-dmod(lo-xi, hi - lo)
        end if
        if (xi >= hi) then
            xi = lo+dmod(xi-hi, hi - lo)
        end if
        mx(i) = xi
      enddo

      RETURN
      END

      SUBROUTINE standardize(x, loc, scale, n, nloc, nscale, z)

c Compute z = (x-mu)/scale

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(n), intent(out) :: z
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py threadsafe


      DOUBLE PRECISION x(n), loc(nloc), scale(nscale), z(n)
      DOUBLE PRECISION mu, sigma
      INTEGER n, nloc, nscale, i
      LOGICAL not_scalar_loc, not_scalar_scale

      mu = loc(1)
      sigma = scale(1)
      not_scalar_loc = (nloc .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

      do i=1,n
        if (not_scalar_loc) mu = loc(i)
        if (not_scalar_scale) sigma = scale(i)
        z(i) = (x(i) - mu)/sigma
      enddo
      END


      DOUBLE PRECISION FUNCTION gammln(xx)
C Returns the value ln[gamma(xx)] for xx > 0.

      DOUBLE PRECISION xx
      INTEGER j
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6)

C Internal arithmetic will be done in double precision,
C a nicety that you can omit if five-figure accuracy is good enough.

      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     +24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     +-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*dlog(tmp)-tmp
      ser=1.000000000190015d0
      do j=1,6
         y=y+1.d0
         ser=ser+cof(j)/y
      enddo
      gammln=tmp+dlog(stp*ser/x)
      return
      END


      DOUBLE PRECISION FUNCTION mvgammln(x, k)
C Returns the logarithm of the multivariate gamma function for x > 0
      IMPLICIT NONE
      DOUBLE PRECISION PI, x
      DOUBLE PRECISION gammln
      PARAMETER (PI=3.141592653589793238462643d0)
      INTEGER j,k

      mvgammln = k * (k-1) / 4 * log(PI)

      do j=1,k
        mvgammln = mvgammln + gammln(x + (1-j)/2)
      enddo

      return
      END


      DOUBLE PRECISION FUNCTION factrl(n)
C Returns the value n! as a floating-point number.

      INTEGER n
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      INTEGER j,ntop
C Table to be filled in only as required.
      DOUBLE PRECISION a(33),gammln
      SAVE ntop,a
C Table initialized with 0! only.
      DATA ntop,a(1)/0,1./

      if (n.lt.0) then
c        write (*,*) 'negative factorial in factrl'
        factrl=-infinity
        return
      else if (n.le.ntop) then
C Already in table.
        factrl=a(n+1)
      else if (n.le.32) then
C Fill in table up to desired value.
        do j=ntop+1,n
          a(j+1)=j*a(j)
        enddo
        ntop=n
        factrl=a(n+1)
      else
C Larger value than size of table is required. Actually,
C this big a value is going to overflow on many computers,
C but no harm in trying.
        factrl=dexp(gammln(n+1.d0))
      endif
      return
      END

      DOUBLE PRECISION FUNCTION factln(n)
C USES gammln Returns ln(n!).

      INTEGER n
      DOUBLE PRECISION a(100),gammln, pass_val
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      SAVE a
C Initialize the table to negative values.
      DATA a/100*-1./
      pass_val = n + 1
      if (n.lt.0) then
c        write (*,*) 'negative factorial in factln'
        factln=-infinity
        return
      endif
C In range of the table.
      if (n.le.99) then
C If not already in the table, put it in.
        if (a(n+1).lt.0.) a(n+1)=gammln(pass_val)
        factln=a(n+1)
      else
C Out of range of the table.
        factln=gammln(pass_val)
      endif
      return
      END

!       def normcdf(x):
!           """Normal cumulative density function."""
!           x = np.atleast_1d(x)
!           return np.array([.5*(1+flib.derf(y/sqrt(2))) for y in x])

      SUBROUTINE normcdf(x, nx)
cf2py intent(hide) nx
cf2py intent(inplace) x
      INTEGER i, nx
      DOUBLE PRECISION x(nx), sqrttwo

      sqrttwo = dsqrt(2.0D0)
      do i=1,nx
          x(i) = x(i) / sqrttwo
          x(i) = 0.5D0*(1.0D0 + derf(x(i)))
      end do
      RETURN
      end

!       mu = np.asarray(mu)
!       tau = np.asarray(tau)
!       return  np.sum(np.log(2.) + np.log(pymc.utils.normcdf((x-mu)*np.sqrt(tau)*alpha))) + normal_like(x,mu,tau)

      SUBROUTINE sn_like(x,nx,mu,tau,alph,nmu,ntau,nalph,like)
cf2py intent(hide) nmu, ntau, nalph, nx
cf2py intent(out) like
cf2py threadsafe

      INTEGER i, nx, nalph, nmu, ntau, tnx
      DOUBLE PRECISION x(nx), mu(nmu), tau(ntau), alph(nalph)
      DOUBLE PRECISION mu_now, tau_now, alph_now, d_now, like
      DOUBLE PRECISION scratch
      LOGICAL vec_mu, vec_tau, vec_alph
      DOUBLE PRECISION PI, sqrttwo, infinity
      PARAMETER (PI=3.141592653589793238462643d0)
      PARAMETER (infinity = 1.7976931348623157d308)

      sqrttwo = dsqrt(2.0D0)
      like = dlog(2.0D0) * nx

      vec_mu = (nmu.GT.1)
      vec_tau = (ntau.GT.1)
      vec_alph = (nalph.GT.1)

      alph_now = alph(1)
      tau_now = tau(1)
      mu_now = mu(1)

      do i=1,nx
         if (vec_mu) then
             mu_now = mu(i)
         end if
         if (vec_alph) then
             alph_now = alph(i)
         end if
         if (vec_tau) then
             tau_now = tau(i)
         end if
         if ((tau_now .LE. 0.0).OR.(dabs(tau_now).GE.infinity)) then
           like = -infinity
           RETURN
         endif

         like = like - 0.5 * tau_now * (x(i) - mu_now) ** 2
         like = like + 0.5 * dlog(0.5 * tau_now / PI)

         scratch = (x(i)-mu_now)*dsqrt(tau_now)*alph_now
         like = like + dlog(0.5D0*(1.0D0+derf(scratch / sqrttwo)))
!          print *, scratch, x(i), mu_now, tau_now, alpha_now
!          print *,

      end do

      RETURN
      END

      SUBROUTINE RSKEWNORM(x,nx,mu,tau,alph,nmu,ntau,nalph,rn,tnx)
cf2py intent(hide) nmu, ntau, nalph, tnx
cf2py intent(out) x
cf2py threadsafe


      INTEGER i, nx, nalph, nmu, ntau, tnx
      DOUBLE PRECISION x(nx), mu(nmu), tau(ntau), alph(nalph)
      DOUBLE PRECISION U1,U2, mu_now, tau_now, alph_now, d_now
      DOUBLE PRECISION rn(tnx)
      LOGICAL vec_mu, vec_tau, vec_alph

      vec_mu = (nmu.GT.1)
      vec_tau = (ntau.GT.1)
      vec_alph = (nalph.GT.1)

      alph_now = alph(1)
      tau_now = tau(1)
      mu_now = mu(1)

      do i=1,nx
         if (vec_mu) then
             mu_now = mu(i)
         end if
         if (vec_alph) then
             alph_now = alph(i)
         end if
         if (vec_tau) then
             tau_now = tau(i)
         end if

         U1 = rn(2*i-1)
         U2 = rn(2*i)
         d_now = alph_now / dsqrt(1.0D0 + alph_now * alph_now)

         x(i)=(d_now*dabs(U1)+dsqrt(1.0D0-d_now**2)*U2)
     *   /dsqrt(tau_now)+mu_now

      end do

      RETURN
      END

      subroutine uniform_like(x,lower,upper,n,nlower,nupper,like)

c Return the uniform likelihood of x.
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(nlower), intent(in) :: lower
cf2py double precision dimension(nupper), intent(in) :: upper
cf2py integer intent(hide), depend(x) :: n=len(x)
cf2py integer intent(hide), depend(lower) :: nlower=len(lower)
cf2py integer intent(hide), depend(upper) :: nupper=len(upper)
cf2py double precision intent(out) :: like
cf2py threadsafe

        IMPLICIT NONE

        INTEGER n, nlower, nupper, i
        DOUBLE PRECISION x(n), lower(nlower), upper(nupper)
        DOUBLE PRECISION like, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)

        low = lower(1)
        high = upper(1)
        like = 0.0
        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            like = -infinity
            RETURN
          else
            like = like - dlog(high-low)
          endif
        enddo
      END subroutine uniform_like

      subroutine uniform_grad_x(x,lower,upper,n,nlower,nupper,gradxlike)

c Return the uniform likelihood gradient wrt x.
c CREATED 01/10

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(nlower), intent(in) :: lower
cf2py double precision dimension(nupper), intent(in) :: upper
cf2py integer intent(hide), depend(x) :: n=len(x)
cf2py integer intent(hide), depend(lower) :: nlower=len(lower)
cf2py integer intent(hide), depend(upper) :: nupper=len(upper)
cf2py double precision dimension(n), intent(out) :: gradxlike
cf2py threadsafe

        IMPLICIT NONE

        INTEGER n, nlower, nupper, i
        DOUBLE PRECISION x(n), lower(nlower), upper(nupper)
        double precision gradxlike(n)
        DOUBLE PRECISION like, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)

      END subroutine uniform_grad_x


      subroutine uniform_grad_l(x,lower,upper,n,nlower,nupper,gradllike)

c Return the uniform likelihood gradient wrt lower.
c CREATED 1/10 JS

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(nlower), intent(in) :: lower
cf2py double precision dimension(nupper), intent(in) :: upper
cf2py integer intent(hide), depend(x) :: n=len(x)
cf2py integer intent(hide), depend(lower) :: nlower=len(lower)
cf2py integer intent(hide), depend(upper) :: nupper=len(upper)
cf2py double precision dimension(nlower), intent(out) :: gradllike
cf2py threadsafe

        IMPLICIT NONE

        INTEGER n, nlower, nupper, i
        DOUBLE PRECISION x(n), lower(nlower), upper(nupper)
        double precision gradllike(nlower)
        DOUBLE PRECISION gradlower, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)

        low = lower(1)
        high = upper(1)


        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            RETURN
          endif
        enddo

        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)

          gradlower = 1.0/(high - low)

	        if (nlower .NE. 1) then
	        	gradllike(i) = gradlower
	        else
	        	gradllike(1) = gradlower + gradllike(1)
	        endif
        enddo
      END subroutine uniform_grad_l

      subroutine uniform_grad_u(x,lower,upper,n,nlower,nupper,gradulike)

c Return the uniform likelihood gradient wrt lower.
c CREATED 01/10 JS

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(nlower), intent(in) :: lower
cf2py double precision dimension(nupper), intent(in) :: upper
cf2py integer intent(hide), depend(x) :: n=len(x)
cf2py integer intent(hide), depend(lower) :: nlower=len(lower)
cf2py integer intent(hide), depend(upper) :: nupper=len(upper)
cf2py double precision dimension(nupper), intent(out) :: gradulike
cf2py threadsafe

        IMPLICIT NONE

        INTEGER n, nlower, nupper, i
        DOUBLE PRECISION x(n), lower(nlower), upper(nupper)
        double precision gradulike(nupper)
        DOUBLE PRECISION gradupper, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)

        low = lower(1)
        high = upper(1)

        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            RETURN
          endif
        enddo

        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)

          gradupper = 1.0/(low - high)

	        if (nlower .NE. 1) then
	        	gradulike(i) = gradupper
	        else
	        	gradulike(1) = gradupper + gradulike(1)
	        endif
        enddo
      END

      subroutine duniform_like(x,lower,upper,n,nlower,nupper,like)

c Return the discrete uniform likelihood of x.
c CREATED 12/06 DH

cf2py intent(hide) n,nlower,nupper
cf2py intent(out) like
cf2py threadsafe
        IMPLICIT NONE

        INTEGER n, nlower, nupper, i
        INTEGER x(n), lower(nlower), upper(nupper)
        DOUBLE PRECISION like, low, high
        DOUBLE PRECISION infinity
        PARAMETER (infinity = 1.7976931348623157d308)

        low = lower(1)
        high = upper(1)
        like = 0.0
        do i=1,n
          if (nlower .NE. 1) low = lower(i)
          if (nupper .NE. 1) high = upper(i)
          if ((x(i) < low) .OR. (x(i) > high)) then
            like = -infinity
            RETURN
          else
            like = like - dlog(high-low+1.0D0)
          endif
        enddo
      END subroutine duniform_like


      SUBROUTINE exponweib(x,a,c,loc,scale,n,na,nc,nloc,nscale,like)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nc), intent(in) :: c
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(a) :: na=len(a)
cf2py integer intent(hide),depend(c) :: nc=len(c)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision intent(out) :: like
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), a(na)
      DOUBLE PRECISION c(nc), loc(nloc), scale(nscale)
      INTEGER i, n, na, nc, nloc, nscale
      DOUBLE PRECISION like, t1
      LOGICAL not_scalar_a, not_scalar_c, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = a(1)
      cc = c(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_c = (nc .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      like = 0.0
      do i=1,n
        if (not_scalar_a) aa = a(i)
        if (not_scalar_c) cc = c(i)
        if (not_scalar_scale) sigma = scale(i)

! Check c(i) > 0
        if (cc .LE. 0.0) then
          like = -infinity
          RETURN
        endif

! Check a(i) > 0
        if (aa .LE. 0.0) then
          like = -infinity
          RETURN
        endif

! Check z(i) > 0
        if (z(i) .LE. 0.0) then
          like = -infinity
          RETURN
        endif

        t1 = dexp(-z(i)**cc)
        pdf = aa*cc*(1.0-t1)**(aa-1.0)*t1*z(i)**(cc-1.0)
        like = like + dlog(pdf/sigma)
      enddo
      END SUBROUTINE EXPONWEIB

      SUBROUTINE exponweib_gx(x,alpha,k,loc,scale,n,na,nk
     &,nloc,nscale,gradlike)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: alpha
cf2py double precision dimension(nk), intent(in) :: k
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: na=len(alpha)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision dimension(n), intent(out) :: gradlike
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), alpha(na), t1, t2
      DOUBLE PRECISION k(nk), loc(nloc), scale(nscale)
      INTEGER i, n, na, nk, nloc, nscale
      DOUBLE PRECISION gradlike(n)
      LOGICAL not_scalar_a, not_scalar_k, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = alpha(1)
      cc = k(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_k = (nk .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      do i = 1,na
     	if (alpha(i) .LE. 0.0) return
      enddo

      do i = 1,nk
     	if (k(i) .LE. 0.0) return
      enddo

      do i = 1,n
     	if (z(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_a) aa = alpha(i)
        if (not_scalar_k) cc = k(i)
        if (not_scalar_scale) sigma = scale(i)

		t2 = -z(i)**cc
        t1 = dexp(t2)

        gradlike(i) = (aa - 1d0)/(1d0 - t1) * t1 *
     & z(i) **(cc -1d0) * cc / sigma
        gradlike(i) = gradlike(i) + (-  z(i) **(cc -1d0)
     & * cc/sigma) - (cc -1d0)/(z(i) *sigma)
      enddo
      END SUBROUTINE


      SUBROUTINE exponweib_gl(x,alpha,k,loc,scale,n,na,
     &nk,nloc,nscale,gradlike)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: alpha
cf2py double precision dimension(nk), intent(in) :: k
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: na=len(alpha)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision dimension(nloc), intent(out) :: gradlike
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), alpha(na), t1, t2
      DOUBLE PRECISION k(nk), loc(nloc), scale(nscale)
      INTEGER i, n, na, nk, nloc, nscale
      DOUBLE PRECISION gradlike(nloc), grad
      LOGICAL not_scalar_a, not_scalar_k, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = alpha(1)
      cc = k(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_k = (nk .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      do i = 1,na
     	if (alpha(i) .LE. 0.0) return
      enddo

      do i = 1,nc
     	if (k(i) .LE. 0.0) return
      enddo

      do i = 1,n
     	if (z(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_a) aa = alpha(i)
        if (not_scalar_k) cc = k(i)
        if (not_scalar_scale) sigma = scale(i)

		t2 = -z(i)**cc
        t1 = dexp(t2)


        grad = (aa - 1d0)/(1d0 - t1) * t1 * z(i) **(cc -1d0)
     & * cc / sigma
        grad = grad + (-  z(i) **(cc -1d0) * cc/sigma) -
     & (cc -1d0)/(z(i) *sigma)
        grad = -grad
        if (nloc .NE. 1) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif
      enddo
      END SUBROUTINE

      SUBROUTINE exponweib_gk(x,alpha,k,loc,scale,n,na,nk,
     &nloc,nscale,gradlike)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: alpha
cf2py double precision dimension(nk), intent(in) :: k
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: na=len(alpha)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision dimension(nk), intent(out) :: gradlike
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), alpha(na), t1, t2
      DOUBLE PRECISION k(nk), loc(nloc), scale(nscale)
      INTEGER i, n, na, nk, nloc, nscale
      DOUBLE PRECISION gradlike(nk), grad
      LOGICAL not_scalar_a, not_scalar_k, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = alpha(1)
      cc = k(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_k = (nk .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      do i = 1,na
     	if (alpha(i) .LE. 0.0) return
      enddo

      do i = 1,nk
     	if (k(i) .LE. 0.0) return
      enddo

      do i = 1,n
     	if (z(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_a) aa = alpha(i)
        if (not_scalar_k) cc = k(i)
        if (not_scalar_scale) sigma = scale(i)

		t2 = -z(i)**cc
        t1 = dexp(t2)

        grad = 1d0/cc  + (aa - 1d0)/(1d0 - t1) * (-t1) * (-t2)
        grad = grad + t2 +1d0
        grad = grad * dlog(z(i))

        if (not_scalar_k) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif
      enddo
      END SUBROUTINE

      SUBROUTINE exponweib_ga(x,alpha,k,loc,scale,n,na,
     &nk,nloc,nscale,gradlike)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: alpha
cf2py double precision dimension(nk), intent(in) :: k
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: na=len(alpha)
cf2py integer intent(hide),depend(k) :: nc=len(k)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision dimension(na), intent(out) :: gradlike
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), alpha(na), t1, t2
      DOUBLE PRECISION k(nk), loc(nloc), scale(nscale)
      INTEGER i, n, na, nk, nloc, nscale
      DOUBLE PRECISION gradlike(na), grad
      LOGICAL not_scalar_a, not_scalar_k, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = alpha(1)
      cc = k(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_k = (nk .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      do i = 1,na
     	if (alpha(i) .LE. 0.0) return
      enddo

      do i = 1,nk
     	if (k(i) .LE. 0.0) return
      enddo

      do i = 1,n
     	if (z(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_a) aa = alpha(i)
        if (not_scalar_k) cc = k(i)
        if (not_scalar_scale) sigma = scale(i)

		t2 = -z(i)**cc
        t1 = dexp(t2)

        grad = 1d0/aa + dlog(1d0 - t1)

        if (not_scalar_a) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif
      enddo
      END SUBROUTINE

      SUBROUTINE exponweib_gs(x,alpha,k,loc,scale,n,na,
     &nk,nloc,nscale,gradlike)

c Exponentiated log-likelihood function
c pdf(z) = a*c*(1-exp(-z**c))**(a-1)*exp(-z**c)*z**(c-1)
c Where z is standardized, ie z = (x-mu)/scale
c CREATED 12/06 DH

cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: alpha
cf2py double precision dimension(nk), intent(in) :: k
cf2py double precision dimension(nloc), intent(in) :: loc
cf2py double precision dimension(nscale), intent(in) :: scale
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: na=len(alpha)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py integer intent(hide),depend(loc) :: nloc=len(loc)
cf2py integer intent(hide),depend(scale) :: nscale=len(scale)
cf2py double precision dimension(nscale), intent(out) :: gradlike
cf2py threadsafe

      DOUBLE PRECISION x(n), z(n), alpha(na), t1, t2
      DOUBLE PRECISION k(nk), loc(nloc), scale(nscale)
      INTEGER i, n, na, nk, nloc, nscale
      DOUBLE PRECISION gradlike(nscale), grad
      LOGICAL not_scalar_a, not_scalar_k, not_scalar_scale
      DOUBLE PRECISION aa, cc, sigma, pdf
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      aa = alpha(1)
      cc = k(1)
      sigma = scale(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_k = (nk .NE. 1)
      not_scalar_scale = (nscale .NE. 1)

c Compute z
      CALL standardize(x, loc, scale, n, nloc, nscale, z)

      do i = 1,na
     	if (alpha(i) .LE. 0.0) return
      enddo

      do i = 1,nc
     	if (k(i) .LE. 0.0) return
      enddo

      do i = 1,n
     	if (z(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_a) aa = alpha(i)
        if (not_scalar_k) cc = k(i)
        if (not_scalar_scale) sigma = scale(i)

		t2 = -z(i)**cc
        t1 = dexp(t2)

        grad = -1d0/sigma + (aa -1d0)/(1-t1) *
     & t1 * z(i) **(cc - 1d0) *cc
        grad = grad + z(i) **(cc - 1d0) *cc
     & + (cc - 1d0)/z(i)
        grad = grad * (-z(i)/sigma)

        if (not_scalar_a) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif
      enddo
      END SUBROUTINE


      SUBROUTINE exponweib_ppf(q,a,c,n,na,nc,ppf)

c     Compute the percentile point function for the
c     Exponentiated Weibull distribution.
c     Accept parameters a,c of length 1 or n.

c CREATED 12/06 DH.

cf2py double precision dimension(n), intent(in) :: q
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nc), intent(in) :: c
cf2py integer intent(hide),depend(q) :: n=len(q)
cf2py integer intent(hide),depend(a) :: na=len(a)
cf2py integer intent(hide),depend(c) :: nc=len(c)
cf2py double precision dimension(n), intent(out) :: ppf
cf2py threadsafe


      IMPLICIT NONE
      INTEGER n,na,nc,i
      DOUBLE PRECISION q(n), a(na), c(nc), ppf(n),ta,tc
      LOGICAL not_scalar_a, not_scalar_c
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

c     Check length of input arrays.
      not_scalar_a = (na .NE. 1)
      not_scalar_c = (nc .NE. 1)
      if ((not_scalar_a) .AND. (na .NE. n)) return
      if ((not_scalar_c) .AND. (nc .NE. n)) return

      ta = a(1)
      tc = c(1)

      DO i=1,n
        if (not_scalar_a) ta = a(i)
        if (not_scalar_c) tc = c(i)
        ppf(i) = (-dlog(1.0 - q(i)**(1.0/ta)))**(1.0/tc)
      ENDDO

      END SUBROUTINE exponweib_ppf


      SUBROUTINE constrain(pass, x, a, b, allow_equal, n, na, nb)

c Check that x is in [a, b] if allow_equal, or
c that x is in ]a, b[ if not.

cf2py integer, intent(out) :: pass
cf2py double precision dimension(n), intent(in) :: x
cf2py double precision dimension(na), intent(in) :: a
cf2py double precision dimension(nb), intent(in) :: b
cf2py integer intent(hide), depend(x) :: n = len(x)
cf2py integer intent(hide), depend(a) :: na = len(a)
cf2py integer intent(hide), depend(b) :: nb = len(b)
cf2py logical intent(in) :: allow_equal
cf2py threadsafe


      IMPLICIT NONE
      INTEGER n, na, nb, i, pass
      DOUBLE PRECISION x(n), a(na), b(nb), ta, tb
      LOGICAL allow_equal, not_scalar_a, not_scalar_b
      pass = 1

      ta = a(1)
      tb = b(1)
      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      if (allow_equal) then
        do i=1,n
          if (not_scalar_a) ta = a(i)
          if (not_scalar_b) tb = b(i)
          if ((x(i) .LT. ta) .OR. (x(i) .GT. tb)) then
            pass = 0
            RETURN
          endif
        enddo
      else
        do i=1,n
          if (not_scalar_a) ta = a(i)
          if (not_scalar_b) tb = b(i)
          if ((x(i) <= ta) .OR. (x(i) >= tb)) then
            pass = 0
            RETURN
          endif
        enddo
      endif
      END


      SUBROUTINE poisson(x,mu,n,nmu,like)

c Poisson log-likelihood function
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nmu
      INTEGER x(n)
      DOUBLE PRECISION mu(nmu), like
      DOUBLE PRECISION sumx, mut, infinity, sumfact
      DOUBLE PRECISION factln
      PARAMETER (infinity = 1.7976931348623157d308)


      mut = mu(1)

c      CALL constrain(x,0,INFINITY,allow_equal=1)
c      CALL constrain(mu,0,INFINITY,allow_equal=0)

      sumx = 0.0
      sumfact = 0.0
      do i=1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif

        if (mut .LT. 0.0) then
          like = -infinity
          RETURN
        endif

        if (x(i) .LT. 0.0) then
          like = -infinity
          RETURN
        endif

        if (.NOT.((x(i) .EQ. 0.0) .AND. (mut .EQ. 0.0))) then
          sumx = sumx + x(i)*dlog(mut) - mut
          sumfact = sumfact + factln(x(i))
        endif
      enddo
      like = sumx - sumfact
      return
      END

      SUBROUTINE poisson_gmu(x,mu,n,nmu,gradlike)

c Poisson log-likelihood function
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nmu
      INTEGER x(n)
      DOUBLE PRECISION mu(nmu), gradlike(nmu), grad
      DOUBLE PRECISION sumx, mut, infinity, sumfact
      DOUBLE PRECISION factln
      PARAMETER (infinity = 1.7976931348623157d308)


      mut = mu(1)

c      CALL constrain(x,0,INFINITY,allow_equal=1)
c      CALL constrain(mu,0,INFINITY,allow_equal=0)

		do i = 1,nmu
	    	if (mu(i) .LT. 0.0) return
	    enddo

        do i = 1,n
	    	if (x(i) .LT. 0.0) return
	    enddo

      do i=1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif

        grad = x(i) / mut -1d0
        if (nmu .NE. 1) then
          gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif

      enddo
      return
      END

      SUBROUTINE trpoisson(x,mu,k,n,nmu,nk,like)

c Truncated poisson log-likelihood function
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer dimension(nk),intent(in) :: k
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nmu, nk, kt
      INTEGER x(n), k(nk)
      DOUBLE PRECISION mu(nmu), like, cdf
      DOUBLE PRECISION sumx, mut, infinity, sumfact, sumcdf
      DOUBLE PRECISION factln, gammq
      PARAMETER (infinity = 1.7976931348623157d308)


      mut = mu(1)
      kt = k(1)

      sumx = 0.0
      sumfact = 0.0
      sumcdf = 0.0
      do i=1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif

        if (nk .NE. 1) then
          kt = k(i)
        endif

C         if (mut .LT. kt) then
C           like = -infinity
C           RETURN
C         endif

        if (kt .LT. 0.0) then
          like = -infinity
          RETURN
        endif

        if (x(i) .LT. kt) then
          like = -infinity
          RETURN
        endif

        if (.NOT.((x(i) .EQ. kt) .AND. (mut .EQ. kt))) then
          sumx = sumx + x(i)*dlog(mut) - mut
          sumfact = sumfact + factln(x(i))
          cdf = gammq(dble(kt), mut)
          sumcdf = sumcdf + dlog(1.-cdf)
        endif
      enddo
      like = sumx - sumfact - sumcdf
      return
      END

      SUBROUTINE trpoisson_gmu(x,mu,k,n,nmu,nk,gradlike)

c Truncated poisson log-likelihood function
c UPDATED 1/16/07 AP

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer dimension(nk),intent(in) :: k
cf2py double precision dimension(nmu),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)
cf2py integer intent(hide),depend(k) :: nk=len(k)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nmu, nk, kt
      INTEGER x(n), k(nk)
      DOUBLE PRECISION mu(nmu), gradlike(nmu),grad, cdf
      DOUBLE PRECISION sumx, mut, infinity, sumfact, sumcdf
      DOUBLE PRECISION factln, gammq
      PARAMETER (infinity = 1.7976931348623157d308)


      mut = mu(1)
      kt = k(1)

      do i = 1, nk
        if (kt .LT. 0.0) return
      enddo

      do i = 1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif
        if (nk .NE. 1) then
          kt = k(i)
        endif
        if (x(i) .LT. kt) return
        if (mut .LT. kt) return
      enddo

      mut = mu(1)
      kt = k(1)

      do i=1,n
        if (nmu .NE. 1) then
          mut = mu(i)
        endif

        if (nk .NE. 1) then
          kt = k(i)
        endif

        grad = x(i) / mut -1d0
        if (nmu .NE. 1) then
          gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif
      enddo
      return
      END


      SUBROUTINE t(x,nu,n,nnu,like)

c Student's t log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nnu),intent(in) :: nu
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(nu) :: nnu=len(nu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nnu
      DOUBLE PRECISION x(n)
      DOUBLE PRECISION nu(nnu), like, infinity, nut
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION gammln
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      nut = nu(1)

      like = 0.0
      do i=1,n
        if (nnu .GT. 1) then
          nut = nu(i)
        endif

        if (nut .LE. 0.0) then
          like = -infinity
          RETURN
        endif

        like = like + gammln((nut+1.0)/2.0)
        like = like - 0.5*dlog(nut * PI) - gammln(nut/2.0)
        like = like - (nut+1)/2 * dlog(1 + (x(i)**2)/nut)
      enddo
      return
      END

      SUBROUTINE t_grad_x(x,nu,n,nnu,gradlikex)

c Student's t log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nnu),intent(in) :: nu
cf2py double precision dimension(n),intent(out) :: gradlikex
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(nu) :: nnu=len(nu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nnu
      DOUBLE PRECISION x(n), nu(nnu), gradlikex(n),gradx, infinity, nut
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION gammln
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      nut = nu(1)

		do i =1,nnu
		  if (nu(i) .LE. 0.0) then
          	RETURN
          endif
		enddo

      gradx = 0.0
      do i=1,n
        if (nnu .GT. 1) then
          nut = nu(i)
        endif

    	gradx = - (nut + 1) * x(i) / ( nut + x(i)**2)
    	if (nnu .GT. 1) then
    		gradlikex(i) = gradx
    	else
    		gradlikex(1) = gradlikex(1) + gradx
    	endif
      enddo
      return
      END

            SUBROUTINE t_grad_nu(x,nu,n,nnu,gradlikenu)

c Student's t log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nnu),intent(in) :: nu
cf2py double precision dimension(nnu),intent(out) :: gradlikenu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(nu) :: nnu=len(nu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nnu
      double precision x(n)
      DOUBLE PRECISION nu(nnu), gradlikenu(nnu),gradnu, infinity, nut
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION psi
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      nut = nu(1)

		do i =1,nnu

		  if (nu(i) .LE. 0.0) then
          	RETURN
          endif
		enddo

      gradnu = 0.0
      do i=1,n
        if (nnu .GT. 1) then
          nut = nu(i)
        endif

    	gradnu = psi((nut + 1d0)/2d0) * .5 - .5/nut - psi(nut/2d0) *.5
    	gradnu = gradnu - .5 * dlog(1 + x(i)**2/nut)
    	gradnu = gradnu + ((nut + 1)/2) * x(i)**2/(nut**2+x(i)**2*nut)

    	if (nnu .GT. 1) then
    		gradlikenu(i) = gradnu
    	else
    		gradlikenu(1) = gradlikenu(1) + gradnu
    	endif
      enddo
      return
      END

	  SUBROUTINE chi2_grad_nu(x,nu,n,nnu,gradlikenu)

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nnu),intent(in) :: nu
cf2py double precision dimension(nnu),intent(out) :: gradlikenu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(nu) :: nnu=len(nu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nnu
      double precision x(n)
      DOUBLE PRECISION nu(nnu), gradlikenu(nnu),gradnu, infinity, nut
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION psi
      DOUBLE PRECISION PI, C
      PARAMETER (PI=3.141592653589793238462643d0)
      PARAMETER (C = -0.34657359027997264d0)

      nut = nu(1)

		do i =1,nnu

		  if (nu(i) .LE. 0.0) then
          	RETURN
          endif
		enddo

      gradnu = 0.0
      do i=1,n
        if (nnu .GT. 1) then
          nut = nu(i)
        endif

    	gradnu = C - psi(nut /2d0) + dlog(x(i))/2d0
    	if (nnu .GT. 1) then
    		gradlikenu(i) = gradnu
    	else
    		gradlikenu(1) = gradlikenu(1) + gradnu
    	endif
      enddo
      return
      END

      SUBROUTINE nct(x,mu,lam,nu,n,nmu,nlam,nnu,like)

c Non-central Student's t log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(nlam),intent(in) :: lam
cf2py double precision dimension(nnu),intent(in) :: nu
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu) :: nmu=len(mu)
cf2py integer intent(hide),depend(lam) :: nlam=len(lam)
cf2py integer intent(hide),depend(nu) :: nnu=len(nu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, nnu, nmu, nlam
      DOUBLE PRECISION x(n)
      DOUBLE PRECISION nu(nnu), mu(nmu), lam(nlam), like, infinity
      DOUBLE PRECISION mut, lamt, nut
      PARAMETER (infinity = 1.7976931348623157d308)
      DOUBLE PRECISION gammln
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)

      nut = nu(1)
      mut = mu(1)
      lamt = lam(1)

      like = 0.0
      do i=1,n
        if (nmu .GT. 1) then
          mut = mu(i)
        endif
        if (nlam .GT. 1) then
          lamt = lam(i)
        endif
        if (nnu .GT. 1) then
          nut = nu(i)
        endif

        if (nut .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        if (lamt .LE. 0.0) then
          like = -infinity
          RETURN
        endif

        like = like + gammln((nut+1.0)/2.0)
        like = like - gammln(nut/2.0)
        like = like + 0.5*dlog(lamt) - 0.5*dlog(nut * PI)
        like = like - (nut+1)/2 * dlog(1 + (lamt*(x(i) - mut)**2)/nut)
      enddo
      return
      END


      SUBROUTINE multinomial(x,n,p,nx,nn,np,k,like)

c Multinomial log-likelihood function
c Updated 12/02/2007 DH. N-D still buggy.
c Fixed 22/11/2007 CF

cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
cf2py integer intent(hide),depend(n) :: nn=shape(n,0)
cf2py integer intent(hide),depend(p) :: np=shape(p,0)
cf2py integer intent(hide),depend(x,p),check(k==shape(p,1)) :: k=shape(x,1)
cf2py intent(out) like
cf2py threadsafe

      DOUBLE PRECISION like, factln, infinity, sump
      DOUBLE PRECISION p(np,k), p_tmp(k)
      INTEGER i,j,n(nn),n_tmp,sumx
      INTEGER x(nx,k)
      PARAMETER (infinity = 1.7976931348623157d308)


      like = 0.0
      n_tmp = n(1)
      do i=1,k
            p_tmp(i) = p(1,i)
      enddo
      do j=1,nx
        if (np .NE. 1) then
              do i=1,k
                    p_tmp(i) = p(j,i)
              enddo
        endif
        if (nn .NE. 1) n_tmp = n(j)

!       protect against negative n
        if (n_tmp .LT. 0) then
          like=-infinity
          RETURN
        endif

        sump = 0.0
        sumx = 0
        do i=1,k

!         protect against negative x or negative p
          if ((x(j,i) .LT. 0) .OR. (p_tmp(i).LT.0.0D0)) then
            like = -infinity
            RETURN
          endif

!         protect against zero p AND nonzero x
          if (p_tmp(i) .EQ. 0.0D0) then
              if (x(j,i) .GT. 0.0D0) then
                  like=-infinity
                  return
              end if
          else
             like = like + x(j,i)*dlog(p_tmp(i))
          end if

          like = like - factln(x(j,i))

          sump = sump + p_tmp(i)
          sumx = sumx + x(j,i)

        enddo
c This is to account for the kth term that is not passed!
c The kth term does get passed... we can check for consistency.
c But roundoff error ofter triggers a false alarm.

          if (sumx .NE. n_tmp) then
              like=-infinity
              return
          endif
         if ((sump .GT. 1.000001) .OR. (sump .LT. 0.999999)) then
             like=-infinity
             return
         endif
!         xk = n_tmp - sumx
!         pk = 1.0 - sump
!         print *,sump, pk, sumx, n_tmp, xk
!         if ((xk .LT. 0) .OR. (pk .LT. 0)) then
!           like = -infinity
!           RETURN
!         endif
!         like = like + (xk) * dlog(pk)
!         like = like - factln(xk)
        like = like + factln(n_tmp)
      enddo
      RETURN
      END

      SUBROUTINE weibull(x,alpha,beta,n,nalpha,nbeta,like)

c Weibull log-likelihood function
c UPDATED 1/16/07 AP

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nbeta),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(beta) :: nbeta=len(beta)
cf2py threadsafe

      DOUBLE PRECISION x(n),alpha(nalpha),beta(nbeta)
      DOUBLE PRECISION like, alphat, betat
      INTEGER n,nalpha,nbeta,i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alphat = alpha(1)
      betat = beta(1)

      like = 0.0
      do i=1,n
        if (nalpha .NE. 1) alphat = alpha(i)
        if (nbeta .NE. 1) betat = beta(i)
        if ((alphat .LE. 0.0) .OR. (betat .LE. 0.0)) then
          like=-infinity
          RETURN
        endif
        if (x(i) .LE. 0.0) then
          like=-infinity
          RETURN
        endif

c normalizing constant
        like = like + (dlog(alphat) - alphat*dlog(betat))
c kernel of distribution
        like = like + (alphat-1) * dlog(x(i))
        like = like - (x(i)/betat)**alphat
      enddo
      return
      END

      SUBROUTINE weibull_gx(x,alpha,beta,n,nalpha,nbeta,gradlike)

c Weibull log-likelihood function
c UPDATED 1/16/07 AP

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nbeta),intent(in) :: beta
cf2py double precision dimension(n), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(beta) :: nbeta=len(beta)
cf2py threadsafe

      DOUBLE PRECISION x(n),alpha(nalpha),beta(nbeta)
      DOUBLE PRECISION gradlike(n), alphat, betat
      INTEGER n,nalpha,nbeta,i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alphat = alpha(1)
      betat = beta(1)

	  do i=1,nalpha
        if (alpha(i) .LE. 0.0) return
      enddo
	  do i=1,nbeta
        if (beta(i) .LE. 0.0) return
      enddo
	  do i=1,n
        if (x(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (nalpha .NE. 1) alphat = alpha(i)
        if (nbeta .NE. 1) betat = beta(i)


		gradlike(i) =(alphat-1d0)/x(i)
     +-alphat*betat**(-alphat)*x(i)**(alphat-1d0)
      enddo
      return
      END
            SUBROUTINE weibull_ga(x,alpha,beta,n,nalpha,nbeta,gradlike)

c Weibull log-likelihood function
c UPDATED 1/16/07 AP

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nbeta),intent(in) :: beta
cf2py double precision dimension(nalpha), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(beta) :: nbeta=len(beta)
cf2py threadsafe

      DOUBLE PRECISION x(n),alpha(nalpha),beta(nbeta)
      DOUBLE PRECISION gradlike(nalpha), alphat, betat, glike
      INTEGER n,nalpha,nbeta,i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alphat = alpha(1)
      betat = beta(1)

	  do i=1,nalpha
        if (alpha(i) .LE. 0.0) return
      enddo
	  do i=1,nbeta
        if (beta(i) .LE. 0.0) return
      enddo
	  do i=1,n
        if (x(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (nalpha .NE. 1) alphat = alpha(i)
        if (nbeta .NE. 1) betat = beta(i)


		glike = 1d0/alphat+dlog(x(i))-dlog(betat)
     +-(x(i)/betat)**alphat*dlog(x(i)/betat)
		if (nalpha .NE. 1) then
        	gradlike(i) = glike
        else
        	gradlike(1) = glike + gradlike(1)
        endif
      enddo
      return
      END

      SUBROUTINE weibull_gb(x,alpha,beta,n,nalpha,nbeta,gradlike)

c Weibull log-likelihood function
c UPDATED 1/16/07 AP

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nbeta),intent(in) :: beta
cf2py double precision dimension(nbeta), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(beta) :: nbeta=len(beta)
cf2py threadsafe

      DOUBLE PRECISION x(n),alpha(nalpha),beta(nbeta)
      DOUBLE PRECISION gradlike(nbeta), alphat, betat, glike
      INTEGER n,nalpha,nbeta,i
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alphat = alpha(1)
      betat = beta(1)

	  do i=1,nalpha
        if (alpha(i) .LE. 0.0) return
      enddo
	  do i=1,nbeta
        if (beta(i) .LE. 0.0) return
      enddo
	  do i=1,n
        if (x(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (nalpha .NE. 1) alphat = alpha(i)
        if (nbeta .NE. 1) betat = beta(i)


		glike = -1d0/betat-(alphat-1d0)/betat+x(i)**alphat
     +*alphat*betat**(-alphat-1d0)
		if (nalpha .NE. 1) then
        	gradlike(i) = glike
        else
        	gradlike(1) = glike + gradlike(1)
        endif
      enddo
      return
      END
      SUBROUTINE logistic(x, mu, tau, n, nmu, ntau, like)

c Logistic log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if (tau_tmp .LE. 0.0) then
          like = -infinity
          RETURN
        endif
        like = like + dlog(tau_tmp) - tau_tmp * (x(i)-mu_tmp)
        like = like - 2.0*dlog(1.0 + dexp(-tau_tmp * (x(i)-mu_tmp)))
      enddo
      return
      END


      SUBROUTINE normal(x,mu,tau,n,nmu, ntau, like)

c Normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if ((tau_tmp .LE. 0.0).OR.(dabs(tau_tmp).GE.infinity)) then
          like = -infinity
          RETURN
        endif
        like = like - 0.5 * tau_tmp * (x(i)-mu_tmp)**2
        like = like + 0.5*dlog(0.5*tau_tmp/PI)
      enddo
      return
      END

      SUBROUTINE normal_grad_tau(x,mu,tau,n,nmu, ntau, grad_tau_like)

c Normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(ntau), intent(out) :: grad_tau_like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION grad_tau_like(ntau)
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp,grad_tau
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)


	  do i = 1, ntau
	  	  if (tau(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        grad_tau = 1.0 / (2 * tau_tmp) - .5 * (x(i) - mu_tmp)**2
        if (not_scalar_tau) then
        	grad_tau_like(i) = grad_tau
        else
        	grad_tau_like(1) = grad_tau + grad_tau_like(1)
        endif
      enddo
      return
      END

      SUBROUTINE normal_grad_x(x,mu,tau,n,nmu, ntau, grad_x_like)

c Normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py double precision dimension(n), intent(out) :: grad_x_like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION grad_x_like(n)
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)


	  do i = 1, ntau
	  	  if (tau(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        grad_x_like(i) = -(x(i) - mu_tmp) * tau_tmp

      enddo
      return
      END

            SUBROUTINE normal_grad_mu(x,mu,tau,n,nmu,ntau,gradmulike)

c Normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py double precision dimension(nmu), intent(out) :: gradmulike
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION gradmulike(nmu)
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp,grad_mu
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)


	  do i = 1, ntau
	  	  if (tau(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        grad_mu = (x(i) - mu_tmp) * tau_tmp

        if (not_scalar_mu) then
        	gradmulike(i) = grad_mu
        else
        	gradmulike(1) = grad_mu + gradmulike(1)
        endif
      enddo
      return
      END


      SUBROUTINE hnormal(x,tau,n,ntau,like)

c Half-normal log-likelihood function

c Updated 24/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(tau,n),check(ntau==1 || ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),tau(ntau),tau_tmp
      LOGICAL not_scalar_tau

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_tau = (ntau .NE. 1)

      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_tau) tau_tmp = tau(i)
        if ((tau_tmp .LE. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        like = like + 0.5 * (dlog(2. * tau_tmp / PI))
        like = like - (0.5 * x(i)**2 * tau_tmp)
      enddo
      return
      END

      SUBROUTINE hnormal_gradx(x,tau,n,ntau,gradlike)

c Half-normal log-likelihood function

c Updated 24/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(n),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(tau,n),check(ntau==1 || ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau
      DOUBLE PRECISION gradlike(n), grad
      DOUBLE PRECISION x(n),tau(ntau),tau_tmp
      LOGICAL not_scalar_tau

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_tau = (ntau .NE. 1)

      tau_tmp = tau(1)

      do i = 1, ntau
	  	  if (tau(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i = 1, n
	  	  if (x(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i=1,n
        if (not_scalar_tau) tau_tmp=tau(i)

        gradlike(i) = -x(i) * tau_tmp

      enddo
      return
      END


      SUBROUTINE hnormal_gradtau(x,tau,n,ntau,gradlike)

c Half-normal log-likelihood function

c Updated 24/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(ntau),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(tau,n),check(ntau==1 || ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau
      DOUBLE PRECISION gradlike(ntau), grad
      DOUBLE PRECISION x(n),tau(ntau),tau_tmp
      LOGICAL not_scalar_tau

      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_tau = (ntau .NE. 1)

      tau_tmp = tau(1)

      do i = 1, ntau
	  	  if (tau(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i = 1, n
	  	  if (x(i) .LE. 0.0) then
          	RETURN
          endif
      enddo

      do i=1,n
        if (not_scalar_tau) tau_tmp=tau(i)

        grad = 1.0 / (2 * tau_tmp) - .5 * x(i)**2
        if (not_scalar_tau) then
        	gradlike(i) = grad
        else
        	gradlike(1) = grad + gradlike(1)
        endif
      enddo
      return
      END


      SUBROUTINE lognormal(x,mu,tau,n,nmu,ntau,like)

c Log-normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)
        if ((tau_tmp .LE. 0.0).OR.(x(i) .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        like = like + 0.5 * (dlog(tau_tmp) - dlog(2.0*PI))
        like = like - 0.5*tau_tmp*(dlog(x(i))-mu_tmp)**2 - dlog(x(i))
      enddo
      return
      END

      SUBROUTINE lognormal_gradx(x,mu,tau,n,nmu,ntau,gradlike)

c Log-normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(n), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION gradlike(n)
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)

      do i=1,n
        if (x(i) .LE. 0.0) return
      enddo
      do i=1,nmu
        if (mu(i) .LE. 0.0) return
      enddo
      do i=1,ntau
        if (tau(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        gradlike(i) = - (1 + (dlog(x(i)) - mu_tmp) * tau_tmp)/x(i)
      enddo
      return
      END

	  SUBROUTINE lognormal_gradmu(x,mu,tau,n,nmu,ntau,gradlike)

c Log-normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(nmu), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION gradlike(nmu), glike
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)

      do i=1,n
        if (x(i) .LE. 0.0) return
      enddo
      do i=1,nmu
        if (mu(i) .LE. 0.0) return
      enddo
      do i=1,ntau
        if (tau(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        glike = (dlog(x(i)) - mu_tmp) * tau_tmp
        if (not_scalar_mu) then
        	gradlike(i) = glike
        else
        	gradlike(1) = glike + gradlike(1)
        end if
      enddo
      return
      END

	  SUBROUTINE lognormal_gradtau(x,mu,tau,n,nmu,ntau,gradlike)

c Log-normal log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(ntau),intent(in) :: tau
cf2py double precision dimension(ntau), intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(tau,n),check(ntau==1||ntau==n) :: ntau=len(tau)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,ntau,nmu
      DOUBLE PRECISION gradlike(ntau), glike
      DOUBLE PRECISION x(n),mu(nmu),tau(ntau)
      DOUBLE PRECISION mu_tmp, tau_tmp
      LOGICAL not_scalar_mu, not_scalar_tau
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_tau = (ntau .NE. 1)

      mu_tmp = mu(1)
      tau_tmp = tau(1)

      do i=1,n
        if (x(i) .LE. 0.0) return
      enddo
      do i=1,nmu
        if (mu(i) .LE. 0.0) return
      enddo
      do i=1,ntau
        if (tau(i) .LE. 0.0) return
      enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_tau) tau_tmp=tau(i)

        glike = 1D0 /(2D0 * tau_tmp) - (dlog(x(i)) - mu_tmp)**2/2D0
        if (not_scalar_tau) then
        	gradlike(i) = glike
        else
        	gradlike(1) = glike + gradlike(1)
        end if
      enddo
      return
      END
      SUBROUTINE arlognormal(x, mu, sigma, rho, beta, n, nmu, like)

C Autocorrelated lognormal loglikelihood.
C David Huard, June 2007

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision intent(in) :: sigma, rho, beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),mu(nmu),sigma, rho, beta
      DOUBLE PRECISION mu_tmp,logx(n),r(n),t1,t2,t3,t4,quad
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      if ((abs(rho) > 1).OR.(sigma <= 0.0)) then
        like = -infinity
        RETURN
      endif

      t1 = n/2. *dlog(2*pi)
      t2 = .5 * (dlog(beta) + 2*n*dlog(sigma) - dlog(1.-rho**2))

      t3 = 0.0
      mu_tmp = mu(1)
      do i=1,n
        if (x(i) <= 0.0) then
          like = -infinity
          RETURN
        endif
        logx(i) = dlog(x(i))
        t3 = t3+logx(i)
        if (nmu .NE. 1) mu_tmp=mu(i)
        r(i) = logx(i) - mu_tmp
      enddo

      quad = 1.0/beta * (1.-rho**2)*r(1)**2
      do i=1,n-1
        quad = quad + (r(i+1) - rho*r(i))**2
      enddo

      t4 = .5 * quad/sigma**2
      like = -t1-t2-t3-t4
      END SUBROUTINE



      SUBROUTINE gev(x,xi,mu,sigma,n,nxi,nmu,nsigma,like)
C
C     COMPUTE THE LIKELIHOOD OF THE GENERALIZED EXTREME VALUE DISTRIBUTION.
C
Cf2py double precision dimension(n), intent(in):: x
Cf2py double precision dimension(nxi), intent(in):: xi
Cf2py double precision dimension(nmu), intent(in):: mu
Cf2py double precision dimension(nsigma), intent(in):: sigma
Cf2py integer intent(hide), depend(x) :: n=len(x)
Cf2py integer intent(hide), depend(xi,n),check(nxi==1||nxi==n) :: nxi=len(xi)
Cf2py integer intent(hide), depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
Cf2py integer intent(hide), depend(sigma,n),check(nsigma==1||nsigma==n) :: nsigma=len(sigma)
Cf2py double precision intent(out):: like

      INTEGER n, nmu, nxi, nsigma, i
      DOUBLE PRECISION x(n), xi(nxi), mu(nmu), sigma(nsigma), like
      DOUBLE PRECISION Z(N), EX(N), PEX(N)
      DOUBLE PRECISION XI_tmp, SIGMA_tmp
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      CALL standardize(x,mu,sigma,n,nmu,nsigma,z)

      xi_tmp = xi(1)
      sigma_tmp = sigma(1)
      LIKE = 0.0
      DO I=1,N
        if (nxi .NE. 1) xi_tmp = xi(i)
        if (nsigma .NE. 1) sigma_tmp = sigma(i)
        IF (ABS(xi_tmp) .LT. 10.**(-5.)) THEN
          LIKE = LIKE - Z(I) - dexp(-Z(I)) - dlog(sigma_tmp)
        ELSE
          EX(I) = 1. + xi_tmp*z(i)
          IF (EX(I) .LT. 0.) THEN
            LIKE = -infinity
            RETURN
          ENDIF
          PEX(I) = EX(I)**(-1./xi_tmp)
          LIKE = LIKE - dlog(sigma_tmp) - PEX(I)
          LIKE = LIKE - (1./xi_tmp +1.)* dlog(EX(I))
        ENDIF
      ENDDO

      end subroutine gev


      SUBROUTINE gev_ppf(q,xi,n,nxi,ppf)
C
C     COMPUTE THE Percentile Point function (PPF) OF THE
C     GENERALIZED EXTREME VALUE DISTRIBUTION.
C
C Created 29/01/2007 DH.
C
Cf2py double precision dimension(n), intent(in):: q
Cf2py double precision dimension(nxi), intent(in):: xi
Cf2py integer intent(hide), depend(q)::n=len(q)
Cf2py integer intent(hide), depend(xi,n),check(nxi==1 || nxi==n) :: nxi=len(xi)
Cf2py double precision dimension(n), intent(out):: ppf

      IMPLICIT NONE
      INTEGER n,nxi,i
      DOUBLE PRECISION q(n), xi(nxi), ppf(n)
      DOUBLE PRECISION xi_tmp

      xi_tmp = xi(1)
      do i=1,n
        if (nxi .NE. 1) xi_tmp= xi(i)
        IF (ABS(xi_tmp) .LT. 10.**(-5.)) THEN
          ppf(i) = -dlog(-dlog(q(i)))
        ELSE
          ppf(i) = 1./xi_tmp * ( (-dlog(q(i)))**(-xi_tmp) -1. )
        ENDIF
      enddo
      return
      END SUBROUTINE gev_ppf




      SUBROUTINE gamma(x,alpha,beta,n,na,nb,like)

c Gamma log-likelihood function

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py threadsafe


      INTEGER i,n,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)
      like = 0.0
      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)
        if ((x(i) .LT. 0.0) .OR. (alpha_tmp .LE. 0.0) .OR.
     +(beta_tmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if (x(i).EQ.0.0) then

            if (alpha_tmp.EQ.1.0) then
                like = like + dlog(beta_tmp)
            else if (alpha_tmp.LT.1.0) then
                like = infinity
                RETURN
            else
                like = -infinity
                RETURN
            end if
        else
            like = like - gammln(alpha_tmp) + alpha_tmp*dlog(beta_tmp)
            like = like + (alpha_tmp - 1.0)*dlog(x(i)) - beta_tmp*x(i)
        end if
      enddo

      return
      END

      SUBROUTINE gamma_grad_x(x,alpha,beta,n,na,nb,gradxlike)

c Gamma log-likelihood gradient function wrt x

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(n),intent(out) :: gradxlike
cf2py threadsafe


      INTEGER i,n,na,nb

      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradxlike(n)
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)

      do i = 1, n
      	if (x(i) .LT. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)

        if (x(i).EQ.0.0) then
			if (alpha_tmp .EQ. 1.0) then
				gradxlike(i) = -beta_tmp
			else
c might be a better way to handle this, but this will do for now
				gradxlike(i) = 0
			end if
        else
            gradxlike(i) = (alpha_tmp - 1.0)/x(i) - beta_tmp
        end if
      enddo

      return
      END

      SUBROUTINE gamma_grad_alpha(x,alpha,beta,n,na,nb,gradalphalike)

c Gamma log-likelihood gradient function wrt alpha

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(na),intent(out) :: gradalphalike
cf2py threadsafe


      INTEGER i,n,na,nb
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradalphalike(na)
      double precision gradalpha
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln, psi
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)


      do i = 1, n
      	if (x(i) .LT. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)

        if (x(i) .EQ. 0.0) then
				gradalpha = -infinity
        else
            gradalpha = dlog(x(i)) - psi(alpha_tmp) + dlog(beta_tmp)
        end if

        if (not_scalar_a) then
        	gradalphalike(i) = gradalpha
        else
        	gradalphalike(1) = gradalpha + gradalphalike(1)
        end if
      enddo

      return
      END

      SUBROUTINE gamma_grad_beta(x,alpha,beta,n,na,nb,gradbetalike)

c Gamma log-likelihood gradient function wrt beta

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(nb),intent(out) :: gradbetalike
cf2py threadsafe


      INTEGER i,n,na,nb
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradbetalike(nb)
      double precision gradbeta
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)

      do i = 1, n
      	if (x(i) .LT. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)

        if (beta_tmp .EQ. 0.0) then
				gradbeta = infinity
        else
            gradbeta = -x(i) + alpha_tmp/beta_tmp
        end if

        if (not_scalar_b) then
        	gradbetalike(i) = gradbeta
        else
        	gradbetalike(1) = gradbeta + gradbetalike(1)
        end if
      enddo

      return
      END

      SUBROUTINE igamma(x,alpha,beta,n,na,nb,like)

c Inverse gamma log-likelihood function

c Updated 26/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha,n),check(na==1||na==n) :: na=len(alpha)
cf2py integer intent(hide),depend(beta,n),check(nb==1||nb==n) :: nb=len(beta)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER i,n,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      DOUBLE PRECISION alpha_tmp, beta_tmp
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      alpha_tmp=alpha(1)
      beta_tmp=beta(1)
      like = 0.0D0
      do i=1,n
        if (na .NE. 1) alpha_tmp=alpha(i)
        if (nb .NE. 1) beta_tmp=beta(i)
        if ((alpha_tmp .LT. 0.0) .OR. (beta_tmp .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        if ((x(i) .LE. 0.0).OR.(alpha_tmp.LE.0.0).OR.
     +       (beta_tmp.LE.0.0)) then
          like = -infinity
          RETURN
        endif
        like = like - gammln(alpha_tmp) + alpha_tmp*dlog(beta_tmp)
        like = like - (alpha_tmp+1.0D0)*dlog(x(i)) - 1.0D0*beta_tmp/x(i)
      enddo

      return
      END

	  SUBROUTINE igamma_grad_x(x,alpha,beta,n,na,nb,gradxlike)

c Gamma log-likelihood gradient function wrt x

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(n),intent(out) :: gradxlike
cf2py threadsafe


      INTEGER i,n,na,nb

      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradxlike(n)
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)

      do i = 1, n
      	if (x(i) .LE. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)


        gradxlike(i) = -(alpha_tmp + 1.0)/x(i) + beta_tmp/x(i)**2

      enddo

      return
      END

      SUBROUTINE igamma_grad_alpha(x,alpha,beta,n,na,nb,gradalphalike)

c Gamma log-likelihood gradient function wrt alpha

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(na),intent(out) :: gradalphalike
cf2py threadsafe


      INTEGER i,n,na,nb
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradalphalike(na)
      double precision gradalpha
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln, psi
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)


      do i = 1, n
      	if (x(i) .LE. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)


        gradalpha = -dlog(x(i)) - psi(alpha_tmp) + dlog(beta_tmp)

        if (not_scalar_a) then
        	gradalphalike(i) = gradalpha
        else
        	gradalphalike(1) = gradalpha + gradalphalike(1)
        end if
      enddo

      return
      END

      SUBROUTINE igamma_grad_beta(x,alpha,beta,n,na,nb,gradbetalike)

c Gamma log-likelihood gradient function wrt beta

c Updated 19/01/2007 DH.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(nb),intent(out) :: gradbetalike
cf2py threadsafe


      INTEGER i,n,na,nb
      DOUBLE PRECISION x(n),alpha(na),beta(nb)
      double precision gradbetalike(nb)
      double precision gradbeta
      DOUBLE PRECISION beta_tmp, alpha_tmp
      LOGICAL not_scalar_a, not_scalar_b
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_a = (na .NE. 1)
      not_scalar_b = (nb .NE. 1)

      alpha_tmp = alpha(1)
      beta_tmp = beta(1)

      do i = 1, n
      	if (x(i) .LE. 0.0) return
      enddo

      do i=1,na
        if  (alpha(i) .LE. 0.0) RETURN
      enddo

      do i=1,nb
        if  (beta(i) .LE. 0.0) RETURN
      enddo

      do i=1,n
        if (not_scalar_a) alpha_tmp = alpha(i)
        if (not_scalar_b) beta_tmp = beta(i)

        gradbeta = alpha_tmp/beta_tmp - 1d0/x(i)

        if (not_scalar_a) then
        	gradbetalike(i) = gradbeta
        else
        	gradbetalike(1) = gradbeta + gradbetalike(1)
        end if
      enddo

      return
      END

      SUBROUTINE hyperg(x,draws,success,total,n,nd,ns,nt,like)

c Hypergeometric log-likelihood function
c Updated 5/02/07, DH. Changed variable names.

c Distribution models the probability of drawing x successes in a
c given number of draws, knowing the population composition (success, failures).
c where failures = total - success

cf2py integer dimension(n),intent(in) :: x
cf2py integer dimension(nd),intent(in) :: draws
cf2py integer dimension(ns),intent(in) :: success
cf2py integer dimension(nt),intent(in) :: total
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(draws,n),check(nd==1||nd==n) :: nd=len(draws)
cf2py integer intent(hide),depend(success,n),check(ns==1||ns==n) :: ns=len(success)
cf2py integer intent(hide),depend(total,n),check(nt==1||nt==n) :: nt=len(total)
cf2py double precision intent(out) :: like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER i,n,nd,ns,nt
      INTEGER x(n),draws(nd), success(ns),total(nt)
      INTEGER draws_tmp, s_tmp, t_tmp
      DOUBLE PRECISION like, combinationln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

c      CALL constrain(d,x,total,allow_equal=1)
c      CALL constrain(red,x,total,allow_equal=1)
c      CALL constrain(x, 0, d, allow_equal=1)

      draws_tmp = draws(1)
      s_tmp = success(1)
      t_tmp = total(1)

!       print *,draws,success,total

      like = 0.0
      do i=1,n
c Combinations of x red balls
        if (nd .NE. 1) draws_tmp = draws(i)
        if (ns .NE. 1) s_tmp = success(i)
        if (nt .NE. 1) t_tmp = total(i)
        if ((draws_tmp .LE. 0) .OR. (s_tmp .LT. 0)) then
          like = -infinity
          RETURN
        endif
        if (t_tmp .LE. 0) then
          like = -infinity
          RETURN
        endif
        if (x(i) .LT. MAX(0, draws_tmp - t_tmp + s_tmp)) then
          like = -infinity
          RETURN
        else if (x(i) .GT. MIN(draws_tmp, s_tmp)) then
          like = -infinity
          RETURN
        endif
c        like = like + combinationln(t_tmp-s_tmp, x(i))
c        like = like + combinationln(s_tmp,draws_tmp-x(i))
        like = like + combinationln(t_tmp-s_tmp, draws_tmp-x(i))
        like = like + combinationln(s_tmp, x(i))
        like = like - combinationln(t_tmp, draws_tmp)
      enddo
      return
      END


      SUBROUTINE geometric(x,p,n,np,like)

c Geometric log-likelihood

c Created 29/01/2007 DH.

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(p,n),check(np==1 || np==n) :: np=len(p)
cf2py double precision intent(out) :: like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,np,i
      INTEGER x(n)
      DOUBLE PRECISION p(np), p_tmp
      DOUBLE PRECISION like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)


      p_tmp = p(1)
      like = 0.0
      do i=1, n
        if (np .NE. 1) p_tmp = p(i)
        if ((p_tmp .LE. 0.0) .OR. (p_tmp .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
        if (x(i) .LT. 1) then
          like = -infinity
          RETURN
        endif
        like = like + dlog(p_tmp) + (x(i)-1)* dlog(1.0D0-p_tmp)
      enddo
      return
      END SUBROUTINE geometric

      SUBROUTINE geometric_gp(x,p,n,np,gradlike)

c Geometric log-likelihood

c Created 29/01/2007 DH.

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(p,n),check(np==1 || np==n) :: np=len(p)
cf2py double precision dimension(np), intent(out) :: gradlike
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,np,i
      INTEGER x(n)
      DOUBLE PRECISION p(np), p_tmp
      DOUBLE PRECISION gradlike(np), grad
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)


      p_tmp = p(1)
      do i=1, np
		  if ((p(i) .LE. 0.0) .OR. (p(i) .GE. 1.0)) return
	  enddo

      do i=1, n
      	if (x(i) .LT. 1) return
      enddo

      do i=1, n
        if (np .NE. 1) p_tmp = p(i)

        grad = - (x(i) - 1)/(1 - p_tmp) + 1d0 / p_tmp

        if (np .NE. 1) then
        	gradlike(i) = grad
        else
        	gradlike(1) = grad + gradlike(1)
        end if
      enddo
      return
      END SUBROUTINE


!       SUBROUTINE dirichlet(x,theta,nx,nt,k,like)
!
! c Dirichlet log-likelihood function
!
! cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
! cf2py integer intent(hide),depend(theta) :: nt=shape(theta,0)
! cf2py integer intent(hide),depend(x,theta),check(k==shape(theta,1)) :: k=shape(x,1)
! cf2py intent(out) like
!
!       IMPLICIT NONE
!       INTEGER i,j,k,nx,nt
!       DOUBLE PRECISION like,sumt,sumx
!       DOUBLE PRECISION x(nx,k),theta(nt,k)
!       DOUBLE PRECISION t_tmp(k)
!       DOUBLE PRECISION gammln
!       DOUBLE PRECISION infinity
!       PARAMETER (infinity = 1.7976931348623157d308)
!
!
!       like = 0.0D0
!       do i=1,k
!             t_tmp(i) = theta(1,i)
!       enddo
!       do j=1,nx
!         if (nt .NE. 1) then
!               do i=1,k
!                     t_tmp(i) = theta(j,i)
!               enddo
!         endif
!
!         sumt = 0.0D0
!         sumx = 0.0D0
!
!         do i=1,k
! !         protect against non-positive x or theta
!           if ((x(j,i) .LE. 0.0D0) .OR. (t_tmp(i) .LE. 0.0D0)) then
!             like = -infinity
!             RETURN
!           endif
!
!           like = like + (t_tmp(i)-1.0D0)*dlog(x(j,i))
!           like = like - gammln(t_tmp(i))
!
!           sumt = sumt + t_tmp(i)
!           sumx = sumx + x(j,i)
!
!         enddo
! !       make sure x sums approximately to unity
!         if ((sumx .GT. 1.000001) .OR. (sumx .LT. 0.999999)) then
!           like=-infinity
!           return
!         endif
!         like = like + gammln(sumt)
!       enddo
!       RETURN
!       END SUBROUTINE dirichlet


      SUBROUTINE dirichlet(x,theta,nx,nt,k,like)

c Dirichlet log-likelihood function with k-1 elements

cf2py integer intent(hide),depend(x) :: nx=shape(x,0)
cf2py integer intent(hide),depend(theta) :: nt=shape(theta,0)
cf2py integer intent(hide),depend(x,theta) :: k=shape(theta,1)
cf2py intent(out) like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER i,j,k,nx,nt
      DOUBLE PRECISION like,sumt,sumx
      DOUBLE PRECISION x(nx,k-1),theta(nt,k)
      DOUBLE PRECISION t_tmp(k)
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)


      like = 0.0D0
      do i=1,k
            t_tmp(i) = theta(1,i)
      enddo
      do j=1,nx
        if (nt .NE. 1) then
              do i=1,k
                    t_tmp(i) = theta(j,i)
              enddo
        endif

        sumt = 0.0D0
        sumx = 0.0D0

        do i=1,k-1
!         protect against non-positive x or theta
          if ((x(j,i) .LE. 0.0D0) .OR. (t_tmp(i).LE.0.0D0)) then
            like = -infinity
            RETURN
          endif

          like = like + (t_tmp(i)-1.0D0)*dlog(x(j,i))
          like = like - gammln(t_tmp(i))

          sumt = sumt + t_tmp(i)
          sumx = sumx + x(j,i)

        enddo
!       implicit kth term
        like = like + (t_tmp(k)-1.0D0)*dlog(1.0D0-sumx)
        like = like - gammln(t_tmp(k))
        sumt = sumt + t_tmp(k)
        if (sumx .GT. 1.0D0) then
            like = -infinity
            RETURN
          endif
        like = like + gammln(sumt)
      enddo
      RETURN
      END SUBROUTINE dirichlet


      SUBROUTINE cauchy(x,alpha,beta,nx, na, nb,like)

c Cauchy log-likelihood function

c UPDATED 17/01/2007 DH.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER nx,na,nb,i
      DOUBLE PRECISION x(nx),alpha(na),beta(nb)
      DOUBLE PRECISION like, atmp, btmp, PI
      LOGICAL not_scalar_alpha, not_scalar_beta
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)
      like = -nx*dlog(PI)
      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)

        if (btmp .LE. 0.0) then
          like = -infinity
          RETURN
        endif

        like = like - dlog(btmp)
        like = like -  dlog( 1. + ((x(i)-atmp) / btmp) ** 2 )
      enddo
      return
      END

      SUBROUTINE cauchy_grad_x(x,alpha,beta,nx, na, nb,gradlike)

c Cauchy log-likelihood function

c UPDATED 17/01/2007 DH.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision dimension(nx),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER nx,na,nb,i
      DOUBLE PRECISION x(nx),alpha(na),beta(nb)
      DOUBLE PRECISION gradlike(nx), atmp, btmp, PI, glike
      LOGICAL not_scalar_alpha, not_scalar_beta
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)

      do i=1,nb
        if (beta(i) .LE. 0.0) return
      enddo

      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)

        gradlike(i) = - 2 * (x(i) - atmp)/
     &(btmp**2 + (x(i) - atmp)**2)

      enddo
      return
      END

      SUBROUTINE cauchy_grad_a(x,alpha,beta,nx, na, nb,gradlike)

c Cauchy log-likelihood function

c UPDATED 17/01/2007 DH.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision dimension(nx),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER nx,na,nb,i
      DOUBLE PRECISION x(nx),alpha(na),beta(nb)
      DOUBLE PRECISION gradlike(na), atmp, btmp, PI, glike
      LOGICAL not_scalar_alpha, not_scalar_beta
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)

      do i=1,nb
        if (beta(i) .LE. 0.0) return
      enddo

      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)

        glike = 2 * (x(i) - atmp)/(btmp**2 + (x(i)-atmp)**2)
        if (not_scalar_alpha) then
        	gradlike(i) = glike

        else
        	gradlike(1) = gradlike(1) + glike
        endif

      enddo
      return
      END

      SUBROUTINE cauchy_grad_b(x,alpha,beta,nx, na, nb,gradlike)

c Cauchy log-likelihood function

c UPDATED 17/01/2007 DH.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py double precision dimension(nx),intent(out) :: gradlike
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER nx,na,nb,i
      DOUBLE PRECISION x(nx),alpha(na),beta(nb)
      DOUBLE PRECISION gradlike(nb), atmp, btmp, PI, glike
      LOGICAL not_scalar_alpha, not_scalar_beta
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_alpha = (na .NE. 1)
      not_scalar_beta = (nb .NE. 1)

      atmp = alpha(1)
      btmp = beta(1)

      do i=1,nb
        if (beta(i) .LE. 0.0) return
      enddo

      do i=1,nx
        if (not_scalar_alpha) atmp = alpha(i)
        if (not_scalar_beta) btmp = beta(i)

        glike = -1D0/btmp
        glike = glike + 2d0 * (x(i)-atmp)**2/
     &    (btmp**3*(1D0+(x(i)-atmp)**2/btmp**2))

        if (not_scalar_beta) then
        	gradlike(i) = glike

        else
        	gradlike(1) = gradlike(1) + glike
        endif

      enddo
      return
      END

      SUBROUTINE negbin(x,r,p,n,nr,np,like)

c Negative binomial log-likelihood function

c Updated 24/01/2007.

cf2py integer dimension(n),intent(in) :: x
cf2py integer dimension(nr),intent(in) :: r
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(r,n),check(nr==1 || nr==n) :: nr=len(r)
cf2py integer intent(hide),depend(p,n),check(np==1 || np==n) :: np=len(p)
cf2py double precision intent(out) :: like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,nr,np,i
      DOUBLE PRECISION like
      DOUBLE PRECISION p(np),p_tmp
      INTEGER x(n),r(nr),r_tmp
      LOGICAL not_scalar_r, not_scalar_p
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_r = (nr .NE. 1)
      not_scalar_p = (np .NE. 1)

      r_tmp = r(1)
      p_tmp = p(1)
      like = 0.0
      do i=1,n
        if (not_scalar_r) r_tmp = r(i)
        if (not_scalar_p) p_tmp = p(i)

        if ((r_tmp .LE. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif

        if ((p_tmp .LE. 0.0) .OR. (p_tmp .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
        like = like + r_tmp*dlog(p_tmp) + x(i)*dlog(1.-p_tmp)
        like = like+factln(x(i)+r_tmp-1)-factln(x(i))-factln(r_tmp-1)
      enddo
      return
      END


      SUBROUTINE negbin2(x,mu,a,n,nmu,na,like)

c Negative binomial log-likelihood function
c (alternative parameterization)
c Updated 1/4/08 CF

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: a
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,x),check(nmu==1 || nmu==len(x)) :: nmu=len(mu)
cf2py integer intent(hide),depend(a,x),check(na==1 || na==len(x)) :: na=len(a)
cf2py double precision intent(out) :: like
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nmu,na
      DOUBLE PRECISION like
      DOUBLE PRECISION a(na),mu(nmu), a_tmp, mu_tmp
      INTEGER x(n)
      LOGICAL not_scalar_a, not_scalar_mu
      DOUBLE PRECISION gammln, factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_a = (na .NE. 1)

      mu_tmp = mu(1)
      a_tmp = a(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp = mu(i)
        if (not_scalar_a) a_tmp = a(i)
        if ((x(i) .LT. 0) .OR. (mu_tmp .LE. 0.0) .OR.
     +(a_tmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_a) a_tmp=a(i)
        like=like + gammln(x(i)+a_tmp) - factln(x(i)) - gammln(a_tmp)
        like=like + x(i)*(dlog(mu_tmp/a_tmp) - dlog(1.0+mu_tmp/a_tmp))
        like=like - a_tmp*dlog(1.0 + mu_tmp/a_tmp)
      enddo
      return
      END

      SUBROUTINE negbin2_gmu(x,mu,alpha,n,nmu,na,gradlike)

c Negative binomial log-likelihood function
c (alternative parameterization)
c Updated 1/4/08 CF

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,x),check(nmu==1 || nmu==len(x)) :: nmu=len(mu)
cf2py integer intent(hide),depend(alpha,x),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py double precision dimension (nmu), intent(out) :: gradlike
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nmu,na
      DOUBLE PRECISION gradlike(nmu), grad
      DOUBLE PRECISION alpha(na),mu(nmu), a_tmp, mu_tmp
      INTEGER x(n)
      LOGICAL not_scalar_a, not_scalar_mu
      DOUBLE PRECISION gammln, factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_a = (na .NE. 1)

      mu_tmp = mu(1)
      a_tmp = alpha(1)

	  do i =1,n
	  	if (x(i) .LT. 0) return
	  enddo

	  do i =1,nmu
	  	if (mu(i) .LE. 0.0) return
	  enddo

	  do i=1,na
	  	if (alpha(i) .LE. 0.0) return
	  enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp = mu(i)
        if (not_scalar_a) a_tmp = alpha(i)

        grad= x(i)/mu_tmp - (x(i) + a_tmp)/(mu_tmp + a_tmp)

        if (not_scalar_mu) then
        	gradlike(i) = grad

        else
        	gradlike(1) = gradlike(1) + grad
        endif

      enddo
      return
      END

      SUBROUTINE negbin2_ga(x,mu,alpha,n,nmu,na,gradlike)

c Negative binomial log-likelihood function
c (alternative parameterization)
c Updated 1/4/08 CF

cf2py integer dimension(n),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,x),check(nmu==1 || nmu==len(x)) :: nmu=len(mu)
cf2py integer intent(hide),depend(alpha,x),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py double precision dimension (na), intent(out) :: gradlike
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nmu,na
      DOUBLE PRECISION gradlike(na), grad
      DOUBLE PRECISION alpha(na),mu(nmu), a_tmp, mu_tmp
      INTEGER x(n)
      LOGICAL not_scalar_a, not_scalar_mu
      DOUBLE PRECISION gammln, factln, psi
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_a = (na .NE. 1)

      mu_tmp = mu(1)
      a_tmp = alpha(1)

	  do i =1,n
	  	if (x(i) .LT. 0) return
	  enddo

	  do i =1,nmu
	  	if (mu(i) .LE. 0.0) return
	  enddo

	  do i=1,na
	  	if (alpha(i) .LE. 0.0) return
	  enddo

      do i=1,n
        if (not_scalar_mu) mu_tmp = mu(i)
        if (not_scalar_a) a_tmp = alpha(i)

		grad=psi(x(i)+a_tmp)-psi(a_tmp)+dlog(a_tmp)+1.0
     +  - dlog(a_tmp+mu_tmp)-a_tmp/(a_tmp+mu_tmp)
     +  - x(i)/(mu_tmp+a_tmp)

        if (not_scalar_a) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif

      enddo
      return
      END


      SUBROUTINE binomial(x,n,p,nx,nn,np,like)

c Binomial log-likelihood function

c  Updated 17/01/2007. DH.

cf2py integer dimension(nx),intent(in) :: x
cf2py integer dimension(nn),intent(in) :: n
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py integer intent(hide),depend(p),check(np==1 || np==len(x)) :: np=len(p)
cf2py double precision intent(out) :: like
cf2py threadsafe
      IMPLICIT NONE
      INTEGER nx,nn,np,i
      DOUBLE PRECISION like, p(np)
      INTEGER x(nx),n(nn)
      LOGICAL not_scalar_n,not_scalar_p
      INTEGER ntmp
      DOUBLE PRECISION ptmp
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_n = (nn .NE. 1)
      not_scalar_p = (np .NE. 1)

      ntmp = n(1)
      ptmp = p(1)

      like = 0.0
      do i=1,nx
        if (not_scalar_n) ntmp = n(i)
        if (not_scalar_p) ptmp = p(i)

        if ((x(i) .LT. 0) .OR. (ntmp .LT. 0) .OR. (x(i) .GT. ntmp)) then
          like = -infinity
          RETURN
        endif

        if ((ptmp .LE. 0.0D0) .OR. (ptmp .GE. 1.0D0)) then
!         if p = 0, number of successes must be 0
          if (ptmp .EQ. 0.0D0) then
            if (x(i) .GT. 0.0D0) then
                like = -infinity
                RETURN
!                 else like = like + 0
            end if
          else if (ptmp .EQ. 1.0D0) then
!           if p = 1, number of successes must be n
            if (x(i) .LT. ntmp) then
                like = -infinity
                RETURN
!                 else like = like + 0
            end if
          else
            like = -infinity
            RETURN
          endif
        else
            like = like + x(i)*dlog(ptmp) + (ntmp-x(i))*dlog(1.-ptmp)
            like = like + factln(ntmp)-factln(x(i))-factln(ntmp-x(i))
        end if
      enddo
      return
      END

      SUBROUTINE binomial_gp(x,n,p,nx,nn,np,gradlike)

c Binomial log-likelihood function

c  Updated 17/01/2007. DH.

cf2py integer dimension(nx),intent(in) :: x
cf2py integer dimension(nn),intent(in) :: n
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py integer intent(hide),depend(p),check(np==1 || np==len(x)) :: np=len(p)
cf2py double precision dimension(np), intent(out) :: gradlike
cf2py threadsafe
      IMPLICIT NONE
      INTEGER nx,nn,np,i
      DOUBLE PRECISION gradlike(np), p(np)
      INTEGER x(nx),n(nn)
      LOGICAL not_scalar_n,not_scalar_p
      INTEGER ntmp
      DOUBLE PRECISION ptmp
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      not_scalar_n = (nn .NE. 1)
      not_scalar_p = (np .NE. 1)

      ntmp = n(1)
      ptmp = p(1)

	  do i =1,nx
	    if (not_scalar_n) ntmp = n(i)
        if ((x(i).LT.0).OR.(ntmp.LT.0) .OR.(x(i) .GT. ntmp)) return
	  enddo

	  do i=1,nx
        if (not_scalar_n) ntmp = n(i)
        if (not_scalar_p) ptmp = p(i)

        if ((ptmp .LE. 0.0D0) .OR. (ptmp .GE. 1.0D0)) then
!         if p = 0, number of successes must be 0
          if (ptmp .EQ. 0.0D0) then
            if (x(i) .GT. 0.0D0) return

          else if (ptmp .EQ. 1.0D0) then
!           if p = 1, number of successes must be n
            if (x(i) .LT. ntmp) RETURN
          else
            RETURN
          endif
        endif
      enddo

      do i=1,nx
        if (not_scalar_n) ntmp = n(i)
        if (not_scalar_p) ptmp = p(i)


        if ((ptmp .LE. 0.0D0) .OR. (ptmp .GE. 1.0D0)) then
			gradlike(i) = 0d0
		else
			gradlike(i) = x(i) / ptmp - (ntmp - x(i))/(1d0 -ptmp)
		endif

      enddo
      return
      END


      SUBROUTINE bernoulli(x,p,nx,np,like)

c Modified on Jan 16 2007 by D. Huard to allow scalar p.

cf2py logical dimension(nx),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(p),check(len(p)==1 || len(p)==len(x)):: np=len(p)
cf2py double precision intent(out) :: like
cf2py threadsafe
      IMPLICIT NONE

      INTEGER np,nx,i
      DOUBLE PRECISION p(np), ptmp, like
      LOGICAL x(nx)
      LOGICAL not_scalar_p
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

C     Check parameter size
      not_scalar_p = (np .NE. 1)

      like = 0.0
      ptmp = p(1)
      do i=1,nx
        if (not_scalar_p) ptmp = p(i)
        if (ptmp .LT. 0.0) then
          like = -infinity
          RETURN
        endif

        if (x(i)) then
          like = like + dlog(ptmp)
        else
          like = like + dlog(1.0D0 - ptmp)
        endif

      enddo
      return
      END

      SUBROUTINE bern_grad_p(x,p,nx,np,grad_like)

c Modified on Jan 16 2007 by D. Huard to allow scalar p.

cf2py logical dimension(nx),intent(in) :: x
cf2py double precision dimension(np),intent(in) :: p
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(p),check(len(p)==1 || len(p)==len(x)):: np=len(p)
cf2py double precision dimension(np),intent(out) :: grad_like
cf2py threadsafe
      IMPLICIT NONE

      INTEGER np,nx,i
      DOUBLE PRECISION p(np), ptmp, glike, grad_like(np)
      LOGICAL x(nx)
      LOGICAL not_scalar_p
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

C     Check parameter size
      not_scalar_p = (np .NE. 1)

      ptmp = p(1)

      do i=1,np
        if (p(i) .LT. 0.0 .OR. p(i) .GT. 1.0) return
      enddo


      do i=1,nx
        if (not_scalar_p) ptmp = p(i)

        if (x(i)) then
          glike = 1.0D0 / ptmp
        else
          glike = -1.0D0 / (1.0D0 - ptmp)
        endif

        if (not_scalar_p) then
        	grad_like(i) = glike
        else
        	grad_like(1) = grad_like(1) + glike
        endif

      enddo
      return
      END

      SUBROUTINE beta_like(x,alpha,beta,nx,na,nb,like)

c Beta log-likelihood function
c Modified by D. Huard on Jan 17 2007 to accept scalar parameters.
c Renamed to use alpha and beta arguments for compatibility with
c random.beta.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision intent(out) :: like
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION x(nx),alpha(na),beta(nb), atmp, btmp
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity, e, zero, one
      PARAMETER (infinity = 1.7976931348623157d308)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      like = 0.0
      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        if ((atmp .LE. 0.0) .OR. (btmp .LE. 0.0)) then
          like = -infinity
          RETURN
        endif
        if ((x(i) .LE. 0.0) .OR. (x(i) .GE. 1.0)) then
          like = -infinity
          RETURN
        endif
        like =like + (gammln(atmp+btmp) - gammln(atmp) - gammln(btmp))
        like =like + (atmp-one)*dlog(x(i)) + (btmp-one)*dlog(one-x(i))
      enddo

      return
      END

      SUBROUTINE beta_grad_x(x,alpha,beta,nx,na,nb,gradlikex)

c Beta log-likelihood function
c Modified by D. Huard on Jan 17 2007 to accept scalar parameters.
c Renamed to use alpha and beta arguments for compatibility with
c random.beta.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(nx),intent(out):: gradlikex
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb
      DOUBLE PRECISION gradlikex(nx), gradx
      DOUBLE PRECISION x(nx),alpha(na),beta(nb), atmp, btmp
      DOUBLE PRECISION psi
      DOUBLE PRECISION infinity, e, zero, one
      PARAMETER (infinity = 1.7976931348623157d308)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      gradx = 0.0

      do i=1,na
      	if (alpha(i) .LE. 0.0) return
      enddo

      do i=1,nb
      	if (beta(i) .LE. 0.0) return
      enddo

      do i=1,nx
      	if ((x(i) .LE. 0.0) .OR. (x(i) .GE. 1.0)) return
      enddo

      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)

        gradlikex(i) = (atmp - 1)/x(i) - (btmp - 1)/(1 - x(i))

      enddo

      return
      END

      SUBROUTINE beta_grad_a(x,alpha,beta,nx,na,nb,gradlikea)

c Beta log-likelihood function
c Modified by D. Huard on Jan 17 2007 to accept scalar parameters.
c Renamed to use alpha and beta arguments for compatibility with
c random.beta.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(na),intent(out):: gradlikea
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb
      DOUBLE PRECISION gradlikea(na), grada
      DOUBLE PRECISION x(nx),alpha(na),beta(nb), atmp, btmp
      DOUBLE PRECISION psi
      DOUBLE PRECISION infinity, e, zero, one
      PARAMETER (infinity = 1.7976931348623157d308)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      grada = 0.0

      do i=1,na
      	if (alpha(i) .LE. 0.0) return
      enddo

      do i=1,nb
      	if (beta(i) .LE. 0.0) return
      enddo

      do i=1,nx
      	if ((x(i) .LE. 0.0) .OR. (x(i) .GE. 1.0)) return
      enddo

      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)

        grada = dlog(x(i)) - psi(atmp) + psi(atmp + btmp)
        if (na .NE. 1) then
        	gradlikea(i) = grada
        else
        	gradlikea(1) = gradlikea(1) + grada
        endif

      enddo

      return
      END

            SUBROUTINE beta_grad_b(x,alpha,beta,nx,na,nb,gradlikeb)

c Beta log-likelihood function
c Modified by D. Huard on Jan 17 2007 to accept scalar parameters.
c Renamed to use alpha and beta arguments for compatibility with
c random.beta.

cf2py double precision dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py double precision dimension(nb),intent(out):: gradlikeb
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb
      DOUBLE PRECISION gradlikeb(nb), gradb
      DOUBLE PRECISION x(nx),alpha(na),beta(nb), atmp, btmp
      DOUBLE PRECISION psi
      DOUBLE PRECISION infinity, e, zero, one
      PARAMETER (infinity = 1.7976931348623157d308)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      gradb = 0.0

      do i=1,na
      	if (alpha(i) .LE. 0.0) return

      enddo

      do i=1,nb
      	if (beta(i) .LE. 0.0) return

      enddo

      do i=1,nx
      	if ((x(i) .LE. 0.0) .OR. (x(i) .GE. 1.0)) then
          RETURN
        endif
      enddo

      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)

        gradb = dlog(1 - x(i)) - psi(btmp) + psi(atmp + btmp)
        if (nb .NE. 1) then
        	gradlikeb(i) = gradb
        else
        	gradlikeb(1) = gradlikeb(1) + gradb
        endif

      enddo

      return
      END

      SUBROUTINE betabin_like(x,alpha,beta,n,nx,na,nb,nn,like)

c Beta-binomial log-likelihood function

cf2py integer dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer dimension(nn),intent(in) :: n
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py double precision intent(out) :: like
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb,nn
      DOUBLE PRECISION like
      DOUBLE PRECISION alpha(na),beta(nb)
      INTEGER x(nx),n(nn)
      DOUBLE PRECISION atmp,btmp,ntmp
      DOUBLE PRECISION gammln
      DOUBLE PRECISION infinity,one
      PARAMETER (infinity = 1.7976931348623157d308)
      data one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      ntmp = n(1)
      like = 0.0
      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        if (nn .NE. 1) ntmp = n(i)
        if ((atmp.LE.0.0).OR.(btmp.LE.0.0).OR.(ntmp.LE.0)) then
          like = -infinity
          RETURN
        endif
        if (x(i) .LT. 0) then
          like = -infinity
          RETURN
        endif
        like =like + gammln(atmp+btmp)
        like =like - gammln(atmp) - gammln(btmp)

        like =like + gammln(ntmp+one)
        like =like - gammln(x(i)+one) - gammln(ntmp-x(i)+one)

        like =like + gammln(atmp+x(i)) + gammln(ntmp+btmp-x(i))
        like =like - gammln(atmp+btmp+ntmp)

      enddo

      return
      END

      SUBROUTINE betabin_ga(x,alpha,beta,n,nx,na,nb,nn,gradlike)

c Beta-binomial log-likelihood function

cf2py integer dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer dimension(nn),intent(in) :: n
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py double precision dimension(na), intent(out) :: gradlike
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb,nn
      DOUBLE PRECISION gradlike(na), grad
      DOUBLE PRECISION alpha(na),beta(nb)
      INTEGER x(nx),n(nn)
      DOUBLE PRECISION atmp,btmp,ntmp
      DOUBLE PRECISION gammln, psi
      DOUBLE PRECISION infinity,one
      PARAMETER (infinity = 1.7976931348623157d308)
      data one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      ntmp = n(1)

	  do i = 1,na
	  	if (alpha(i) .LE. 0.0) return
	  enddo

	  do i = 1,nb
	  	if (beta(i) .LE. 0.0) return
	  enddo

	  do i = 1,nn
	  	if (n(i) .LE. 0) return
	  enddo

	  do i = 1,nx
	  	if (x(i) .LT. 0) return
	  enddo


      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        if (nn .NE. 1) ntmp = n(i)

        grad=psi(atmp+btmp)-psi(atmp)+
     +       psi(atmp+x(i))-psi(atmp+btmp+ntmp)

        if (na .NE. 1) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif

      enddo

      return
      END

      SUBROUTINE betabin_gb(x,alpha,beta,n,nx,na,nb,nn,gradlike)

c Beta-binomial log-likelihood function

cf2py integer dimension(nx),intent(in) :: x
cf2py double precision dimension(na),intent(in) :: alpha
cf2py double precision dimension(nb),intent(in) :: beta
cf2py integer dimension(nn),intent(in) :: n
cf2py integer intent(hide),depend(x) :: nx=len(x)
cf2py integer intent(hide),depend(alpha),check(na==1 || na==len(x)) :: na=len(alpha)
cf2py integer intent(hide),depend(beta),check(nb==1 || nb==len(x)) :: nb=len(beta)
cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
cf2py double precision dimension(nb), intent(out) :: gradlike
cf2py threadsafe
      IMPLICIT NONE
      INTEGER i,nx,na,nb,nn
      DOUBLE PRECISION gradlike(nb), grad
      DOUBLE PRECISION alpha(na),beta(nb)
      INTEGER x(nx),n(nn)
      DOUBLE PRECISION atmp,btmp,ntmp
      DOUBLE PRECISION gammln, psi
      DOUBLE PRECISION infinity,one
      PARAMETER (infinity = 1.7976931348623157d308)
      data one/1.0d0/


      atmp = alpha(1)
      btmp = beta(1)
      ntmp = n(1)

	  do i = 1,na
	  	if (alpha(i) .LE. 0.0) return
	  enddo

	  do i = 1,nb
	  	if (beta(i) .LE. 0.0) return
	  enddo

	  do i = 1,nn
	  	if (n(i) .LE. 0) return
	  enddo

	  do i = 1,nx
	  	if (x(i) .LT. 0) return
	  enddo


      do i=1,nx
        if (na .NE. 1) atmp = alpha(i)
        if (nb .NE. 1) btmp = beta(i)
        if (nn .NE. 1) ntmp = n(i)

        grad =psi(atmp+btmp)+ psi(ntmp + btmp - x(i))
     +        -psi(atmp +btmp+ntmp)

        if (na .NE. 1) then
        	gradlike(i) = grad
        else
        	gradlike(1) = gradlike(1) + grad
        endif

      enddo

      return
      END


      SUBROUTINE mvhyperg(x,color,k,like)

c Multivariate hypergeometric log-likelihood function
c Using the analogy of an urn filled with balls of different colors,
c the mv hypergeometric distribution describes the probability of
c drawing x(i) balls of a given color.
c
c x : (array) Number of draws for each color.
c color : (array) Number of balls of each color.

c Total number of draws = sum(x)
c Total number of balls in the urn = sum(color)

cf2py integer dimension(k),intent(in) :: x,color
cf2py integer intent(hide),depend(x) :: k=len(x)
cf2py double precision intent(out) :: like
cf2py threadsafe

      INTEGER x(k),color(k)
      INTEGER d,total,i,k
      DOUBLE PRECISION like
      DOUBLE PRECISION factln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      total = 0
      d = 0
      like = 0.0
      do i=1,k
c Combinations of x balls of color i
        like = like + factln(color(i))-factln(x(i))
     +-factln(color(i)-x(i))
        if ((color(i) .LT. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
        d = d + x(i)
        total = total + color(i)
      enddo
      if (total .LE. 0.0) then
        like = -infinity
        RETURN
      endif
c Combinations of d draws from total
      like = like - (factln(total)-factln(d)-factln(total-d))
      return
      END


      SUBROUTINE dirmultinom(x,theta,k,like)

c Dirichlet-multinomial log-likelihood function

cf2py integer dimension(k),intent(in) :: x
cf2py double precision dimension(k),intent(in) :: theta
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: k=len(x)
cf2py threadsafe

      INTEGER i,k,sumx
      INTEGER x(k)
      DOUBLE PRECISION like,sumt
      DOUBLE PRECISION theta(k)
      DOUBLE PRECISION factln, gammln
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      like = 0.0
      sumt = 0.0
      sumx = 0
      do 222 i=1,k
c kernel of distribution
        like = like + dlog(x(i) + theta(i)) - dlog(theta(i))
        sumt = sumt + theta(i)
        sumx = sumx + x(i)
        if ((theta(i) .LT. 0.0) .OR. (x(i) .LT. 0.0)) then
          like = -infinity
          RETURN
        endif
  222 continue
c normalizing constant

      if ((sumx .LE. 0.0) .OR. (sumt .LE. 0.0)) then
        like = -infinity
        RETURN
      endif

      like = like + factln(sumx)
      like = like + gammln(sumt)
      like = like - gammln(sumx + sumt)

      return
      END


      SUBROUTINE wishart(X,k,n,sigma,like)

c Wishart log-likelihood function

cf2py double precision dimension(k,k),intent(in) :: X,sigma
cf2py double precision intent(in) :: n
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(X) :: k=len(X)
cf2py threadsafe

      INTEGER i,k
      DOUBLE PRECISION X(k,k),sigma(k,k),bx(k,k)
      DOUBLE PRECISION dx,n,db,tbx,a,g,like
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)


c determinants
      call dtrm(X,k,dx)
      call dtrm(sigma,k,db)
c trace of sigma*X
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



      SUBROUTINE trace(mat,k,tr)

c matrix trace (sum of diagonal elements)

      INTEGER k,i
      DOUBLE PRECISION mat(k,k),tr

      tr = 0.0
      do i=1,k
        tr = tr + mat(k,k)
      enddo
      return
      END


      SUBROUTINE gamfun(xx,gx)

c Return the logarithm of the gamma function
c Corresponds to scipy.special.gammaln

cf2py double precision intent(in) :: xx
cf2py double precision intent(out) :: gx
cf2py threadsafe

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


      double precision function gammds (y,p,ifault)

cf2py double precision intent(in) :: y,p,ifault
cf2py double precision intent(out) :: gammds
cf2py threadsafe

c
c        Algorithm AS 147  Appl. Statist. (1980) Vol. 29, No. 1
c
c        Computes the incomplete gamma integral for positive
c        parameters y,p using an infinite series
c
c        Auxiliary function required: ALNGAM = CACM algorithm 291
c
c	 AS239 should be considered as an alternative to AS147
c
      implicit double precision (a-h,o-z)
      data e/1.0d-9/, zero/0.0d0/, one/1.0d0/, uflo/1.0d-37/
c
c        Checks admissibility of arguments and value of f
c
      ifault = 1
      gammds = zero
      if(y.le.zero .or. p.le.zero) return
      ifault = 2
c
c        alngam is natural log of gamma function
c
      arg = p*log(y)-gammln(p+one)-y
      if(arg.lt.log(uflo)) return
      f = exp(arg)
      if(f.eq.zero) return
      ifault = 0
c
c          Series begins
c
      c = one
      gammds = one
      a = p
    1 a = a+one
      c = c*y/a
      gammds = gammds+c
      if (c/gammds.gt.e) goto 1
      gammds = gammds*f
      return
      end

      double precision function psi(x)
c taken from
c Bernardo, J. M. (1976). Algorithm AS 103: Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
c http://www.uv.es/~bernardo/1976AppStatist.pdf
cf2py     double precision intent(in) :: x
cf2py 	  double precision intent(out) :: psi
	      double precision x, y, R

	      DATA S /1.0e-5/, C /8.5/ S3 /8.333333333e-2/
	      DATA S4 /8.333333333e-3/, S5 /3.968253968e-3/
	      DATA D1 /-0.5772156649/

	      psi = 0

	      y = x

	      if (y .LE. 0.0) return

	      if (y .LE. S) then
	      	 psi = d1 - 1.0/y
	      	 return
	      endif

	      do while (y .LT. C)
	      	psi = psi - 1.0 / y
	      	y = y + 1
	      enddo

	      R= 1.0 / y
	      psi = psi + dlog(y) - .5 * R
	      R= R*R
	      psi = psi - R * (S3 - R * (S4 - R * S5))
	      return


      end

      SUBROUTINE gser(gamser,a,x,gln)
      INTEGER ITMAX
      DOUBLE PRECISION a,gamser,gln,x,EPS
      PARAMETER (ITMAX=100,EPS=3.e-7)
C USES gammln
C Returns the incomplete gamma function P (a, x) evaluated by its series representation as
C gamser. Also returns ln Γ(a) as gln.
      INTEGER n
      DOUBLE PRECISION ap,del,sum,gammln

      gln=gammln(a)
      if(x.le.0.)then
        if(x.lt.0.) write (*,*) 'x < 0 in gser'
        gamser=0.
        return
      endif
      ap=a
      sum=1./a
      del=sum
      do n=1,ITMAX
        ap=ap+1.
        del=del*x/ap
        sum=sum+del
        if (abs(del).lt.abs(sum)*EPS) goto 1
      enddo
      write (*,*) 'a too large, ITMAX too small in gser'
    1 gamser=sum*exp(-x+a*log(x)-gln)
      return
      END


      SUBROUTINE gcf(gammcf,a,x,gln)
      INTEGER ITMAX
      DOUBLE PRECISION a,gammcf,gln,x,EPS,FPMIN
      PARAMETER (ITMAX=100,EPS=3.e-7,FPMIN=1.e-30)
C USES gammln
C Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction
C representation as gammcf. Also returns ln Γ(a) as gln.
C Parameters: ITMAX is the maximum allowed number of iterations; EPS is the relative
C accuracy; FPMIN is a number near the smallest representable ﬂoating-point number.
      INTEGER i
      DOUBLE PRECISION an,b,c,d,del,h,gammln
      gln=gammln(a)
      b=x+1.-a
      c=1./FPMIN
      d=1./b
      h=d
      do i=1,ITMAX
        an=-i*(i-a)
        b=b+2.
        d=an*d+b
        if(abs(d).lt.FPMIN)d=FPMIN
        c=b+an/c
        if(abs(c).lt.FPMIN)c=FPMIN
        d=1./d
        del=d*c
        h=h*del
        if(abs(del-1.).lt.EPS)goto 1
      enddo
      write (*,*) 'a too large, ITMAX too small in gcf'
    1 gammcf=exp(-x+a*log(x)-gln)*h
      return
      END


      FUNCTION gammq(a,x)
cf2py double precision intent(in) :: a,x
cf2py double precision intent(out) :: gammaq
cf2py threadsafe
      DOUBLE PRECISION a,gammq,x
C USES gcf,gser
C Returns the incomplete gamma function Q(a, x) ≡ 1 − P (a, x).
      DOUBLE PRECISION gammcf,gamser,gln
      if(x.lt.0..or.a.le.0.) write (*,*) 'bad arguments in gammq'
      if(x.lt.a+1.)then
C Use the series representation
        call gser(gamser,a,x,gln)
        gammq=1.-gamser
      else
C Use the continued fraction representation.
        call gcf(gammcf,a,x,gln)
      gammq=gammcf
      endif
      return
      END


      SUBROUTINE trans(mat,tmat,m,n)

c matrix transposition

cf2py double precision dimension(m,n),intent(in) :: mat
cf2py double precision dimension(n,m),intent(out) :: tmat
cf2py integer intent(hide),depend(mat) :: m=len(mat)
cf2py integer intent(hide),depend(mat) :: n=shape(mat,1)
cf2py threadsafe

      INTEGER i,j,m,n
      DOUBLE PRECISION mat(m,n),tmat(n,m)

      do 88 i=1,m
        do 99 j=1,n
          tmat(j,i) = mat(i,j)
 99     continue
 88   continue

      return
      END


      SUBROUTINE matmult(mat1, mat2, prod, m, n, p, q)

c matrix multiplication

cf2py double precision dimension(m,q),intent(out) :: prod
cf2py double precision dimension(m,n),intent(in) :: mat1
cf2py double precision dimension(p,q),intent(in) :: mat2
cf2py integer intent(hide),depend(mat1) :: m=len(mat1),n=shape(mat1,1)
cf2py integer intent(hide),depend(mat2) :: p=len(mat2),q=shape(mat2,1)
cf2py threadsafe


      INTEGER i,j,k,m,n,p,q
      DOUBLE PRECISION mat1(m,n), mat2(p,q), prod(m,q)
      DOUBLE PRECISION sum

      if (n.eq.p) then
        do 30 i = 1,m
          do 20 j = 1,q
            sum = 0.0
            do 10 k = 1,n
              sum = sum + mat1(i,k) * mat2(k,j)
10          continue
            prod(i,j) = sum
20        continue
30      continue
      else
        write (*,*) 'Matrix dimensions do not match'
      end if
      return
      END


c Updated 10/24/2001.
c
ccccccccccccccccccccccccc     Program 4.2     cccccccccccccccccccccccccc
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c Please Note:                                                         c
c                                                                      c
c (1) This computer program is part of the book, "An Introduction to   c
c     Computational Physics," written by Tao Pang and published and    c
c     copyrighted by Cambridge University Press in 1997.               c
c                                                                      c
c (2) No warranties, express or implied, are made for this program.    c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      SUBROUTINE DTRM(A,N,D)
C
C Subroutine for evaluating the determinant of a matrix using
C the partial-pivoting Gaussian elimination scheme.
C

cf2py double precision dimension(N,N),intent(in) :: A
cf2py double precision intent(out) :: D
cf2py integer intent(hide),depend(A) :: N=len(A)
cf2py threadsafe

      DOUBLE PRECISION A(N,N), D
      INTEGER INDX(N)

      CALL ELGS(A,N,INDX)
C
      D    = 1.0
      DO     100 I = 1, N
         D = D*A(INDX(I),I)
  100 CONTINUE
C
      MSGN = 1
      DO     200 I = 1, N
        DO   150 WHILE (I.NE.INDX(I))
          MSGN = -MSGN
          J = INDX(I)
          INDX(I) = INDX(J)
          INDX(J) = J
  150   END DO
  200 CONTINUE
      D = MSGN*D

      RETURN
      END


      SUBROUTINE ELGS(A,N,INDX)

C Subroutine to perform the partial-pivoting Gaussian elimination.
C A(N,N) is the original matrix in the input and transformed
C matrix plus the pivoting element ratios below the diagonal in
C the output.  INDX(N) records the pivoting order.


      DOUBLE PRECISION A(N,N),C(N),C1,PI,PJ,PI1
      INTEGER N
      INTEGER INDX(N)

C Initialize the index

      DO     50    I = 1, N
        INDX(I) = I
   50 CONTINUE

C Find the rescaling factors, one from each row

        DO     100   I = 1, N
          C1= 0.0
          DO    90   J = 1, N
            C1 = MAX(C1,ABS(A(I,J)))
   90     CONTINUE
          C(I) = C1
  100   CONTINUE

C Search the pivoting (largest) element from each column

      DO     200   J = 1, N-1
        PI1 = 0.0
        DO   150   I = J, N
          PI = ABS(A(INDX(I),J))/C(INDX(I))
          IF (PI.GT.PI1) THEN
            PI1 = PI
            K   = I
          ELSE
          ENDIF
  150   CONTINUE

C Interchange the rows via INDX(N) to record pivoting order

        ITMP    = INDX(J)
        INDX(J) = INDX(K)
        INDX(K) = ITMP
        DO   170   I = J+1, N
          PJ  = A(INDX(I),J)/A(INDX(J),J)

C Record pivoting ratios below the diagonal

          A(INDX(I),J) = PJ

C Modify other elements accordingly

          DO 160   K = J+1, N
            A(INDX(I),K) = A(INDX(I),K)-PJ*A(INDX(J),K)
  160     CONTINUE
  170   CONTINUE
  200 CONTINUE

      RETURN
      END


      FUNCTION bico(n,k)
C USES factln Returns the binomial coefficient as a
C floating point number.
      INTEGER k,n
      DOUBLE PRECISION bico
      DOUBLE PRECISION factln
C The nearest-integer function cleans up roundoff error
C for smaller values of n and k.
      bico=nint(dexp(factln(n)-factln(k)-factln(n-k)))
      return
      END

      subroutine chol(n,a,c)
c...perform a Cholesky decomposition of matrix a, returned as c
      implicit double precision (a-h,o-z)
      double precision c(n,n),a(n,n)

cf2py double precision dimension(n,n),intent(in) :: a
cf2py double precision dimension(n,n),intent(out) :: c
cf2py integer intent(in),depend(a) :: n=len(a)
cf2py threadsafe

      c(1,1) = sqrt(a(1,1))
      do i=2,n
         c(i,1) = a(i,1) / c(1,1)
      enddo
      do j=2,n
         do i=j,n
            s = a(i,j)
            do k=1,j-1
               s = s - c(i,k) * c(j,k)
            enddo
            if(i .eq. j) then
               c(j,j) = sqrt(s)
            else
               c(i,j) = s / c(j,j)
               c(j,i) = 0.d0
            endif
         enddo
      enddo
      return
      end

      SUBROUTINE hermpoly( n, x, cx )

C*******************************************************************************
C
CC HERMPOLY evaluates the Hermite polynomials at X.
C
C  Differential equation:
C
C    Y'' - 2 X Y' + 2 N Y = 0
C
C  First terms:
C
C      1
C      2 X
C      4 X**2     -  2
C      8 X**3     - 12 X
C     16 X**4     - 48 X**2     + 12
C     32 X**5    - 160 X**3    + 120 X
C     64 X**6    - 480 X**4    + 720 X**2    - 120
C    128 X**7   - 1344 X**5   + 3360 X**3   - 1680 X
C    256 X**8   - 3584 X**6  + 13440 X**4  - 13440 X**2   + 1680
C    512 X**9   - 9216 X**7  + 48384 X**5  - 80640 X**3  + 30240 X
C   1024 X**10 - 23040 X**8 + 161280 X**6 - 403200 X**4 + 302400 X**2 - 30240
C
C  Recursion:
C
C    H(0,X) = 1,
C    H(1,X) = 2*X,
C    H(N,X) = 2*X * H(N-1,X) - 2*(N-1) * H(N-2,X)
C
C  Norm:
C
C    Integral ( -Infinity < X < Infinity ) exp ( - X**2 ) * H(N,X)**2 dX
C    = sqrt ( PI ) * 2**N * N!
C
C    H(N,X) = (-1)**N * exp ( X**2 ) * dn/dXn ( exp(-X**2 ) )
C
C  Modified:
C
C    01 October 2002
C
C  Author:
C
C    John Burkardt
C
C  Reference:
C
C    Milton Abramowitz and Irene Stegun,
C    Handbook of Mathematical Functions,
C    US Department of Commerce, 1964.
C
C    Larry Andrews,
C    Special Functions of Mathematics for Engineers,
C    Second Edition,
C    Oxford University Press, 1998.
C
C  Parameters:
C
C    Input, integer N, the highest order polynomial to compute.
C    Note that polynomials 0 through N will be computed.
C
C    Input, double precision ( kind = 8 ) X, the point at which the polynomials are
C    to be evaluated.
C
C    Output, double precision ( kind = 8 ) CX(0:N), the values of the first N+1 Hermite
C    polynomials at the point X.
C

cf2py double precision intent(in) :: x
cf2py integer intent(in) :: n
cf2py double precision dimension(n+1),intent(out) :: cx
cf2py threadsafe

      integer n,i
      double precision cx(n+1)
      double precision x

      if ( n < 0 ) then
        return
      end if

      cx(1) = 1.0

      if ( n == 0 ) then
        return
      end if

      cx(2) = 2.0 * x

      do i = 3, n+1
        cx(i) = 2.0 * x * cx(i-1) - 2.0 * real(i - 1) * cx(i-2)
      end do

      return
      end

!         double precision function uniform()
! c
! c    Generate uniformly distributed random numbers using the 32-bit
! c    generator from figure 3 of:
! c    L`Ecuyer, P. Efficient and portable combined random number
! c    generators, C.A.C.M., vol. 31, 742-749 & 774-?, June 1988.
! c
! c    The cycle length is claimed to be 2.30584E+18
! c
! c    Seeds can be set by calling the routine set_uniform
! c
! c    It is assumed that the Fortran compiler supports long variable
! c    names, and integer*4.
! c
!         integer*4 z, k, s1, s2
!         common /unif_seeds/ s1, s2
!         save /unif_seeds/
! c
!         k = s1 / 53668
!         s1 = 40014 * (s1 - k * 53668) - k * 12211
!         if (s1 .lt. 0) s1 = s1 + 2147483563
! c
!         k = s2 / 52774
!         s2 = 40692 * (s2 - k * 52774) - k * 3791
!         if (s2 .lt. 0) s2 = s2 + 2147483399
! c
!         if (z .lt. 1) z = z + 2147483562
! c
!         uniform = z / 2147483563.
!         return
!         end


        subroutine set_uniform(seed1, seed2)
c
c    Set seeds for the uniform random number generator.
c
        integer*4 s1, s2, seed1, seed2
        common /unif_seeds/ s1, s2
        save /unif_seeds/

        s1 = seed1
        s2 = seed2
        return
      END

!
!       SUBROUTINE categorical(x,n,hist,k,mn,step,logp)
!
! cf2py intent(out) logp
! cf2py intent(hide) n,k
!
!       DOUBLE PRECISION hist(k),logp,x(n),mn,step,nrm
!       INTEGER n,k,i,j
!       DOUBLE PRECISION infinity
!       PARAMETER (infinity = 1.7976931348623157d308)
!       LOGICAL match
!
!       logp = 0.0D0
!       nrm = 0.0D0
!       do i=1,k
!           nrm = nrm + hist(k)
!       end do
!       if (dabs(nrm-0.0D0).GT.1.0D-7) then
!           logp = -infinity
!           return
!       end if
!
!       do i=1,n
!           match = .FALSE.
!
!           j = int(x(i)-mn/step)+1
!           logp = logp + dlog(hist(j))
!
!       end do
!
!       return
!       END
!

!       SUBROUTINE categorical(x,p,n,k,like)
!
! c Categorical log-likelihood function
! c Need to return -Infs when appropriate
!
! cf2py integer dimension(n),intent(in) :: x
! cf2py double precision dimension(k-1),intent(in) :: p
! cf2py integer intent(hide),depend(x) :: n=len(x)
! cf2py integer intent(hide),depend(p) :: k=len(p)+1
! cf2py double precision intent(out) :: like
! cf2py threadsafe
!
!       DOUBLE PRECISION p(k),val,like
!       INTEGER x(n)
!       INTEGER n,k,i,j
!       DOUBLE PRECISION infinity, sump
!       PARAMETER (infinity = 1.7976931348623157d308)
!
!       like = 0.0
!       sump = 0.0
!       do j=1,k-1
!           sump = sump + p(j)
!       end do
! c loop over number of elements in x
!       do i=1,n
! c elements should not be larger than the largest index
!         if ((x(i).GT.(k-1)).OR.(x(i).LT.0)) then
!           like = -infinity
!           RETURN
!         endif
! c increment log-likelihood
!         if (x(i).eq.(k-1)) then
! c likelihood of the kth element
!           like = like + dlog(1.0D0-sump)
!         else
!           like = like + dlog(p(x(i)+1))
!         endif
!       enddo
!       return
!       END


      SUBROUTINE categorical(x,p,nx,np,k,like)

c Categorical log-likelihood function

cf2py integer dimension(nx), intent(in) :: x
cf2py double precision dimension(np,k), intent(in) :: p
cf2py integer intent(hide), depend(p) :: np=shape(p, 0)
cf2py integer intent(hide), depend(x) :: nx=len(x)
cf2py integer intent(hide), depend(p) :: k=shape(p, 1)
cf2py double precision intent(out) :: like
cf2py threadsafe

      DOUBLE PRECISION like, factln, infinity, sump
      DOUBLE PRECISION p(np,k), p_tmp(k)
      INTEGER i,j,n_tmp
      INTEGER x(nx)
      PARAMETER (infinity = 1.7976931348623157d308)

      sump = 0.0
      do i=1,k
        p_tmp(i) = p(1,i)
        sump = sump + p_tmp(i)
      enddo

      like = 0.0
      do j=1,nx
        if (np .NE. 1) then
          sump = 0.0
          do i=1,k
            p_tmp(i) = p(j,i)
            sump = sump + p_tmp(i)
          enddo
        endif
c       Protect against zero p[x]
        if (p_tmp(x(j)+1).LE.0.0D0) then
          like = -infinity
          RETURN
        end if
c       Category outside of set
        if ((x(j) .GT. k-1) .OR. (x(j) .LT. 0)) then
          like = -infinity
          RETURN
        end if
        like = like + dlog(p_tmp(x(j)+1))

      enddo
      RETURN
      END


      SUBROUTINE rcat(p,s,k,n,rands)

c Returns n samples from categorical random variable (histogram)

cf2py double precision dimension(k-1),intent(in) :: p
cf2py double precision dimension(n),intent(in) :: rands
cf2py integer dimension(n),intent(out) :: s
cf2py integer intent(hide),depend(p) :: k=len(p)+1
cf2py integer intent(hide),depend(rands) :: n=len(rands)
cf2py threadsafe


      DOUBLE PRECISION p(k-1),sump,u,rands(n)
      INTEGER s(n)
      INTEGER n,k,i,j

c repeat for n samples
      do i=1,n
c initialize sum
        sump = p(1)
c random number
        u = rands(i)
c initialize index
        j = 0

c find index to value
        do while (u.gt.sump)
          j = j + 1
          if (j.eq.(k-1)) then
            goto 1
          endif
          sump = sump + p(j+1)

      enddo
c assign value to array
    1 s(i) = j
      enddo
      return
      END



      subroutine logit(theta,n,ltheta)
c Maps (0,1) -> R.
cf2py intent(hide) n
cf2py intent(out) ltheta
cf2py threadsafe
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      INTEGER n, i
      do i=1,n
          if (theta(i).LE.0.0D0) then
              ltheta(i) = -infinity
          else if (theta(i).GE.1.0D0) then
              ltheta(i) = infinity
          else
              ltheta(i) = dlog(theta(i) / (1.0D0-theta(i)))
          endif
      end do
      RETURN
      END
c


      subroutine invlogit(ltheta,n,theta)
c Maps R -> (0,1).
cf2py intent(hide) n
cf2py intent(out) theta
cf2py threadsafe
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      INTEGER n, i
      do i=1,n
          theta(i) = 1.0D0 / (1.0D0 + dexp(-ltheta(i)))
      end do
      RETURN
      END

c
      subroutine stukel_logit(theta,n,ltheta,a1,a2,na1,na2)
!
! Reference: Therese A. Stukel, 'Generalized Logistic Models',
! JASA vol 83 no 402, pp.426-431 (June 1988)
!
cf2py intent(hide) n, na1, na2
cf2py intent(out) ltheta
cf2py intent(copy) theta
cf2py threadsafe
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION a1(na1), a2(na2), a1t, a2t
      LOGICAL a1_isscalar, a2_isscalar
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      INTEGER n, i, na1, na2

      a1t = a1(1)
      a2t = a2(1)

      CALL logit(theta,n,ltheta)

      a1_isscalar = (na1.LT.n)
      a2_isscalar = (na2.LT.n)

      do i=1,n

          if (ltheta(i).GT.0.0D0) then
              if (.NOT.a1_isscalar) then
                  a1t = a1(i)
              end if
              if (a1t.GT.0.0D0) then
                  ltheta(i)=dlog(ltheta(i)*a1t+1.0D0)/a1t
              else if (a1t.LT.0.0D0) then
                  ltheta(i) = (1.0D0-dexp(-ltheta(i)*a1t))/a1t
              end if

          else if (ltheta(i).LT.0.0D0) then
              if (.NOT.a2_isscalar) then
                  a2t = a2(i)
              end if
              if (a2t.GT.0.0D0) then
                  ltheta(i)=-dlog(-ltheta(i)*a2t+1.0D0)/a2t
              else if (a2t.LT.0.0D0) then
                  ltheta(i)=-(1.0D0-dexp(ltheta(i)*a2t))/a2t
              end if

          else
              ltheta(i) = 0.0D0
          end if

      end do

      RETURN
      END

c

      subroutine stukel_invlogit(ltheta,n,theta,a1,a2,na1,na2)
!
! Reference: Therese A. Stukel, 'Generalized Logistic Models',
! JASA vol 83 no 402, pp.426-431 (June 1988)
!
cf2py intent(hide) n, na1, na2
cf2py intent(out) theta
cf2py intent(copy) ltheta
cf2py threadsafe
      DOUBLE PRECISION theta(n), ltheta(n)
      DOUBLE PRECISION a1(na1), a2(na2), a1t, a2t
      LOGICAL a1_isscalar, a2_isscalar
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      INTEGER n, i, na1, na2

      a1t = a1(1)
      a2t = a2(1)

      a1_isscalar = (na1.LT.n)
      a2_isscalar = (na2.LT.n)

      do i=1,n
          if (ltheta(i).GT.0.0D0) then
              if (.NOT.a1_isscalar) then
                  a1t = a1(i)
              end if
              if (a1t.GT.0.0D0) then
                  ltheta(i) = (dexp(a1t*ltheta(i))-1.0D0)/a1t
              else if (a1t.LT.0.0D0) then
                  ltheta(i) = -dlog(1.0D0-a1t*ltheta(i))/a1t
              end if

          else if (ltheta(i).LT.0.0D0) then
              if (.NOT.a2_isscalar) then
                  a2t = a2(i)
              end if
              if (a2t.GT.0.0D0) then
                  ltheta(i) = -(dexp(-a2t*ltheta(i))-1.0D0)/a2t
              else if (a2t.LT.0.0D0) then
                  ltheta(i) = dlog(1.0D0+a2t*ltheta(i))/a2t
              end if

!           else
!               ltheta(i) = 0.5D0
          end if

      end do


      CALL invlogit(ltheta,n,theta)

      RETURN
      END


      SUBROUTINE vonmises(x,mu,kappa,n,nmu, nkappa, like)

c von Mises log-likelihood function

c Written 13/01/2009 ADS.

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nmu),intent(in) :: mu
cf2py double precision dimension(nkappa),intent(in) :: kappa
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(mu,n),check(nmu==1||nmu==n) :: nmu=len(mu)
cf2py integer intent(hide),depend(kappa,n),check(nkappa==1||nkappa==n) :: nkappa=len(kappa)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nkappa,nmu
      DOUBLE PRECISION like
      DOUBLE PRECISION tmp
      DOUBLE PRECISION x(n),mu(nmu),kappa(nkappa)
      DOUBLE PRECISION mu_tmp, kappa_tmp
      LOGICAL not_scalar_mu, not_scalar_kappa
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141592653589793238462643d0)
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      DOUBLE PRECISION i0

      not_scalar_mu = (nmu .NE. 1)
      not_scalar_kappa = (nkappa .NE. 1)

      mu_tmp = mu(1)
      kappa_tmp = kappa(1)
      like = 0.0
      do i=1,n
        if (not_scalar_mu) mu_tmp=mu(i)
        if (not_scalar_kappa) kappa_tmp=kappa(i)
        if (kappa_tmp .LT. 0.0) then
          like = -infinity
          RETURN
        endif
        like = like - dlog(2*PI*i0(kappa_tmp))
        like = like + kappa_tmp*dcos(x(i)-mu_tmp)
      enddo
      return
      END


      SUBROUTINE pareto(x,alpha,m,n,nalpha,nm,like)

c Pareto log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nm),intent(in) :: m
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha,n),check(nalpha==1||nalpha==n) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(m,n),check(nm==1||nm==n) :: nm=len(m)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nalpha,nm
      DOUBLE PRECISION like
      DOUBLE PRECISION tmp
      DOUBLE PRECISION x(n),m(nm),alpha(nalpha)
      DOUBLE PRECISION m_tmp, alpha_tmp
      LOGICAL not_scalar_m, not_scalar_alpha
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      DOUBLE PRECISION i0

      not_scalar_m = (nm .NE. 1)
      not_scalar_alpha = (nalpha .NE. 1)

      m_tmp = m(1)
      alpha_tmp = alpha(1)
      like = 0.0
      do i=1,n
        if (not_scalar_m) m_tmp=m(i)
        if (not_scalar_alpha) alpha_tmp=alpha(i)
        if ((alpha_tmp .LE. 0.0) .OR. (m_tmp .LE. 0.0) .OR.
     +(x(i) .LT. m_tmp)) then
          like = -infinity
          RETURN
        endif
        like = like + dlog(alpha_tmp) + alpha_tmp*dlog(m_tmp)
        like = like - (alpha_tmp + 1)*dlog(x(i))
      enddo
      return
      END

      SUBROUTINE truncated_pareto(x,alpha,m,b,n,nalpha,nm,nb,like)

c Truncated Pareto log-likelihood function

cf2py double precision dimension(n),intent(in) :: x
cf2py double precision dimension(nalpha),intent(in) :: alpha
cf2py double precision dimension(nm),intent(in) :: m
cf2py double precision dimension(nb),intent(in) :: b
cf2py double precision intent(out) :: like
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha,n),check(nalpha==1||nalpha==n) :: nalpha=len(alpha)
cf2py integer intent(hide),depend(m,n),check(nm==1||nm==n) :: nm=len(m)
cf2py integer intent(hide),depend(b,n),check(nb==1||nb==n) :: nb=len(b)
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n,i,nalpha,nm,nb
      DOUBLE PRECISION like
      DOUBLE PRECISION tmp
      DOUBLE PRECISION x(n),m(nm),alpha(nalpha),b(nb)
      DOUBLE PRECISION m_tmp, alpha_tmp, b_tmp
      LOGICAL not_scalar_m, not_scalar_alpha, not_scalar_b
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

      DOUBLE PRECISION i0

      not_scalar_m = (nm .NE. 1)
      not_scalar_alpha = (nalpha .NE. 1)
      not_scalar_b = (nb .NE. 1)

      m_tmp = m(1)
      alpha_tmp = alpha(1)
      b_tmp = b(1)
      like = 0.0
      do i=1,n
        if (not_scalar_m) m_tmp=m(i)
        if (not_scalar_alpha) alpha_tmp=alpha(i)
        if (not_scalar_b) b_tmp=b(i)
        if ((alpha_tmp .LE. 0.0) .OR. (m_tmp .LE. 0.0) .OR.
     +(x(i) .LT. m_tmp) .OR. (b_tmp .LT. x(i))) then
          like = -infinity
          RETURN
        endif
        like = like + dlog(alpha_tmp) + alpha_tmp*dlog(m_tmp)
        like = like - (alpha_tmp + 1)*dlog(x(i))
        like = like - dlog(1 - (m_tmp/b_tmp)**alpha_tmp)
      enddo
      return
      END
