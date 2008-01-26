      REAL FUNCTION SDSDOT(N,SB,SX,INCX,SY,INCY)
*     .. Scalar Arguments ..
      REAL SB
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      REAL SX(*),SY(*)
*     ..
*
*  PURPOSE
*  =======
*
*  Compute the inner product of two vectors with extended
*  precision accumulation.
*
*  Returns S.P. result with dot product accumulated in D.P.
*  SDSDOT = SB + sum for I = 0 to N-1 of SX(LX+I*INCX)*SY(LY+I*INCY),
*  where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
*  defined in a similar way using INCY.
*
*  AUTHOR
*  ======
*  Lawson, C. L., (JPL), Hanson, R. J., (SNLA),
*  Kincaid, D. R., (U. of Texas), Krogh, F. T., (JPL)
*
*  ARGUMENTS 
*  =========
*
*  N      (input) INTEGER
*         number of elements in input vector(s)
*
*  SB     (input) REAL
*         single precision scalar to be added to inner product
*
*  SX     (input) REAL array, dimension (N)
*         single precision vector with N elements
*
*  INCX   (input) INTEGER
*         storage spacing between elements of SX
*
*  SY     (input) REAL array, dimension (N)
*         single precision vector with N elements
*
*  INCY   (input) INTEGER
*         storage spacing between elements of SY
*
*  SDSDOT (output) REAL
*         single precision dot product (SB if N .LE. 0)
*
*  REFERENCES
*  ==========
*
*  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
*  Krogh, Basic linear algebra subprograms for Fortran
*  usage, Algorithm No. 539, Transactions on Mathematical
*  Software 5, 3 (September 1979), pp. 308-323.
*
*  REVISION HISTORY  (YYMMDD)
*  ==========================
*      
*  791001  DATE WRITTEN
*  890531  Changed all specific intrinsics to generic.  (WRB)
*  890831  Modified array declarations.  (WRB)
*  890831  REVISION DATE from Version 3.2
*  891214  Prologue converted to Version 4.0 format.  (BAB)
*  920310  Corrected definition of LX in DESCRIPTION.  (WRB)
*  920501  Reformatted the REFERENCES section.  (WRB)
*  070118  Reformat to LAPACK coding style
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DSDOT
      INTEGER I,KX,KY,NS
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC DBLE
*     ..
      DSDOT = SB
      IF (N.LE.0) GO TO 30
      IF (INCX.EQ.INCY .AND. INCX.GT.0) GO TO 40
*
*     Code for unequal or nonpositive increments.
*
      KX = 1
      KY = 1
      IF (INCX.LT.0) KX = 1 + (1-N)*INCX
      IF (INCY.LT.0) KY = 1 + (1-N)*INCY
      DO 10 I = 1,N
          DSDOT = DSDOT + DBLE(SX(KX))*DBLE(SY(KY))
          KX = KX + INCX
          KY = KY + INCY
   10 CONTINUE
   30 SDSDOT = DSDOT
      RETURN
*
*     Code for equal and positive increments.
*
   40 NS = N*INCX
      DO 50 I = 1,NS,INCX
          DSDOT = DSDOT + DBLE(SX(I))*DBLE(SY(I))
   50 CONTINUE
      SDSDOT = DSDOT
      RETURN
      END
