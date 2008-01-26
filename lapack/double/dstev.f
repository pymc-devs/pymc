      SUBROUTINE DSTEV( JOBZ, N, D, E, Z, LDZ, WORK, INFO )
*
*  -- LAPACK driver routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ
      INTEGER            INFO, LDZ, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), E( * ), WORK( * ), Z( LDZ, * )
*     ..
*
*  Purpose
*  =======
*
*  DSTEV computes all eigenvalues and, optionally, eigenvectors of a
*  real symmetric tridiagonal matrix A.
*
*  Arguments
*  =========
*
*  JOBZ    (input) CHARACTER*1
*          = 'N':  Compute eigenvalues only;
*          = 'V':  Compute eigenvalues and eigenvectors.
*
*  N       (input) INTEGER
*          The order of the matrix.  N >= 0.
*
*  D       (input/output) DOUBLE PRECISION array, dimension (N)
*          On entry, the n diagonal elements of the tridiagonal matrix
*          A.
*          On exit, if INFO = 0, the eigenvalues in ascending order.
*
*  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
*          On entry, the (n-1) subdiagonal elements of the tridiagonal
*          matrix A, stored in elements 1 to N-1 of E.
*          On exit, the contents of E are destroyed.
*
*  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
*          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
*          eigenvectors of the matrix A, with the i-th column of Z
*          holding the eigenvector associated with D(i).
*          If JOBZ = 'N', then Z is not referenced.
*
*  LDZ     (input) INTEGER
*          The leading dimension of the array Z.  LDZ >= 1, and if
*          JOBZ = 'V', LDZ >= max(1,N).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (max(1,2*N-2))
*          If JOBZ = 'N', WORK is not referenced.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  if INFO = i, the algorithm failed to converge; i
*                off-diagonal elements of E did not converge to zero.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            WANTZ
      INTEGER            IMAX, ISCALE
      DOUBLE PRECISION   BIGNUM, EPS, RMAX, RMIN, SAFMIN, SIGMA, SMLNUM,
     $                   TNRM
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      DOUBLE PRECISION   DLAMCH, DLANST
      EXTERNAL           LSAME, DLAMCH, DLANST
*     ..
*     .. External Subroutines ..
      EXTERNAL           DSCAL, DSTEQR, DSTERF, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          SQRT
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      WANTZ = LSAME( JOBZ, 'V' )
*
      INFO = 0
      IF( .NOT.( WANTZ .OR. LSAME( JOBZ, 'N' ) ) ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDZ.LT.1 .OR. ( WANTZ .AND. LDZ.LT.N ) ) THEN
         INFO = -6
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSTEV ', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
      IF( N.EQ.1 ) THEN
         IF( WANTZ )
     $      Z( 1, 1 ) = ONE
         RETURN
      END IF
*
*     Get machine constants.
*
      SAFMIN = DLAMCH( 'Safe minimum' )
      EPS = DLAMCH( 'Precision' )
      SMLNUM = SAFMIN / EPS
      BIGNUM = ONE / SMLNUM
      RMIN = SQRT( SMLNUM )
      RMAX = SQRT( BIGNUM )
*
*     Scale matrix to allowable range, if necessary.
*
      ISCALE = 0
      TNRM = DLANST( 'M', N, D, E )
      IF( TNRM.GT.ZERO .AND. TNRM.LT.RMIN ) THEN
         ISCALE = 1
         SIGMA = RMIN / TNRM
      ELSE IF( TNRM.GT.RMAX ) THEN
         ISCALE = 1
         SIGMA = RMAX / TNRM
      END IF
      IF( ISCALE.EQ.1 ) THEN
         CALL DSCAL( N, SIGMA, D, 1 )
         CALL DSCAL( N-1, SIGMA, E( 1 ), 1 )
      END IF
*
*     For eigenvalues only, call DSTERF.  For eigenvalues and
*     eigenvectors, call DSTEQR.
*
      IF( .NOT.WANTZ ) THEN
         CALL DSTERF( N, D, E, INFO )
      ELSE
         CALL DSTEQR( 'I', N, D, E, Z, LDZ, WORK, INFO )
      END IF
*
*     If matrix was scaled, then rescale eigenvalues appropriately.
*
      IF( ISCALE.EQ.1 ) THEN
         IF( INFO.EQ.0 ) THEN
            IMAX = N
         ELSE
            IMAX = INFO - 1
         END IF
         CALL DSCAL( IMAX, ONE / SIGMA, D, 1 )
      END IF
*
      RETURN
*
*     End of DSTEV
*
      END
