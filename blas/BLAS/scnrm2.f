      REAL FUNCTION SCNRM2(N,X,INCX)
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      COMPLEX X(*)
*     ..
*
*  Purpose
*  =======
*
*  SCNRM2 returns the euclidean norm of a vector via the function
*  name, so that
*
*     SCNRM2 := sqrt( conjg( x' )*x )
*
*
*
*  -- This version written on 25-October-1982.
*     Modified on 14-October-1993 to inline the call to CLASSQ.
*     Sven Hammarling, Nag Ltd.
*
*
*     .. Parameters ..
      REAL ONE,ZERO
      PARAMETER (ONE=1.0E+0,ZERO=0.0E+0)
*     ..
*     .. Local Scalars ..
      REAL NORM,SCALE,SSQ,TEMP
      INTEGER IX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS,AIMAG,REAL,SQRT
*     ..
      IF (N.LT.1 .OR. INCX.LT.1) THEN
          NORM = ZERO
      ELSE
          SCALE = ZERO
          SSQ = ONE
*        The following loop is equivalent to this call to the LAPACK
*        auxiliary routine:
*        CALL CLASSQ( N, X, INCX, SCALE, SSQ )
*
          DO 10 IX = 1,1 + (N-1)*INCX,INCX
              IF (REAL(X(IX)).NE.ZERO) THEN
                  TEMP = ABS(REAL(X(IX)))
                  IF (SCALE.LT.TEMP) THEN
                      SSQ = ONE + SSQ* (SCALE/TEMP)**2
                      SCALE = TEMP
                  ELSE
                      SSQ = SSQ + (TEMP/SCALE)**2
                  END IF
              END IF
              IF (AIMAG(X(IX)).NE.ZERO) THEN
                  TEMP = ABS(AIMAG(X(IX)))
                  IF (SCALE.LT.TEMP) THEN
                      SSQ = ONE + SSQ* (SCALE/TEMP)**2
                      SCALE = TEMP
                  ELSE
                      SSQ = SSQ + (TEMP/SCALE)**2
                  END IF
              END IF
   10     CONTINUE
          NORM = SCALE*SQRT(SSQ)
      END IF
*
      SCNRM2 = NORM
      RETURN
*
*     End of SCNRM2.
*
      END
