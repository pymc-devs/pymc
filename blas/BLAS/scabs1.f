      REAL FUNCTION SCABS1(Z)
*     .. Scalar Arguments ..
      COMPLEX Z
*     ..
*
*  Purpose
*  =======
*
*  SCABS1 computes absolute value of a complex number
*
*     .. Intrinsic Functions ..
      INTRINSIC ABS,AIMAG,REAL
*     ..
      SCABS1 = ABS(REAL(Z)) + ABS(AIMAG(Z))
      RETURN
      END
