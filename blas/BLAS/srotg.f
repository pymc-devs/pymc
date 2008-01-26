      SUBROUTINE SROTG(SA,SB,C,S)
*     .. Scalar Arguments ..
      REAL C,S,SA,SB
*     ..
*
*  Purpose
*  =======
*
*     construct givens plane rotation.
*     jack dongarra, linpack, 3/11/78.
*
*
*     .. Local Scalars ..
      REAL R,ROE,SCALE,Z
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS,SIGN,SQRT
*     ..
      ROE = SB
      IF (ABS(SA).GT.ABS(SB)) ROE = SA
      SCALE = ABS(SA) + ABS(SB)
      IF (SCALE.NE.0.0) GO TO 10
      C = 1.0
      S = 0.0
      R = 0.0
      Z = 0.0
      GO TO 20
   10 R = SCALE*SQRT((SA/SCALE)**2+ (SB/SCALE)**2)
      R = SIGN(1.0,ROE)*R
      C = SA/R
      S = SB/R
      Z = 1.0
      IF (ABS(SA).GT.ABS(SB)) Z = S
      IF (ABS(SB).GE.ABS(SA) .AND. C.NE.0.0) Z = 1.0/C
   20 SA = R
      SB = Z
      RETURN
      END
