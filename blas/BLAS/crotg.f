      SUBROUTINE CROTG(CA,CB,C,S)
*     .. Scalar Arguments ..
      COMPLEX CA,CB,S
      REAL C
*     ..
*
*  Purpose
*  =======
*
*  CROTG determines a complex Givens rotation.
*
*     .. Local Scalars ..
      COMPLEX ALPHA
      REAL NORM,SCALE
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC CABS,CONJG,SQRT
*     ..
      IF (CABS(CA).NE.0.) GO TO 10
      C = 0.
      S = (1.,0.)
      CA = CB
      GO TO 20
   10 CONTINUE
      SCALE = CABS(CA) + CABS(CB)
      NORM = SCALE*SQRT((CABS(CA/SCALE))**2+ (CABS(CB/SCALE))**2)
      ALPHA = CA/CABS(CA)
      C = CABS(CA)/NORM
      S = ALPHA*CONJG(CB)/NORM
      CA = ALPHA*NORM
   20 CONTINUE
      RETURN
      END
