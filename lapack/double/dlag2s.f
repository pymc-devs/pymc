      SUBROUTINE DLAG2S( M, N, A, LDA, SA, LDSA, INFO)
*
*  -- LAPACK PROTOTYPE auxiliary routine (version 3.1.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     January 2007
*
*     ..
*     .. WARNING: PROTOTYPE ..
*     This is an LAPACK PROTOTYPE routine which means that the
*     interface of this routine is likely to be changed in the future
*     based on community feedback.
*
*     .. Scalar Arguments ..
      INTEGER INFO,LDA,LDSA,M,N
*     ..
*     .. Array Arguments ..
      REAL SA(LDSA,*)
      DOUBLE PRECISION A(LDA,*)
*     ..
*
*  Purpose
*  =======
*
*  DLAG2S converts a DOUBLE PRECISION matrix, SA, to a SINGLE
*  PRECISION matrix, A.
*
*  RMAX is the overflow for the SINGLE PRECISION arithmetic
*  DLAG2S checks that all the entries of A are between -RMAX and
*  RMAX. If not the convertion is aborted and a flag is raised.
*
*  This is a helper routine so there is no argument checking.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of lines of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M-by-N coefficient matrix A.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  SA      (output) REAL array, dimension (LDSA,N)
*          On exit, if INFO=0, the M-by-N coefficient matrix SA.
*
*  LDSA    (input) INTEGER
*          The leading dimension of the array SA.  LDSA >= max(1,M).
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          > 0:  if INFO = k, the (i,j) entry of the matrix A has
*                overflowed when moving from DOUBLE PRECISION to SINGLE
*                k is given by k = (i-1)*LDA+j
*
*  =========
*
*     .. Local Scalars ..
      INTEGER I,J
      DOUBLE PRECISION RMAX
*     ..
*     .. External Functions ..
      REAL SLAMCH
      EXTERNAL SLAMCH
*     ..
*     .. Executable Statements ..
*
      RMAX = SLAMCH('O')
      DO 20 J = 1,N
          DO 30 I = 1,M
              IF ((A(I,J).LT.-RMAX) .OR. (A(I,J).GT.RMAX)) THEN
                  INFO = (I-1)*LDA + J
                  GO TO 10
              END IF
              SA(I,J) = A(I,J)
   30     CONTINUE
   20 CONTINUE
   10 CONTINUE
      RETURN
*
*     End of DLAG2S
*
      END
