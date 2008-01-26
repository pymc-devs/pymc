      SUBROUTINE DSBGV( JOBZ, UPLO, N, KA, KB, AB, LDAB, BB, LDBB, W, Z,
     $                  LDZ, WORK, INFO )
*
*  -- LAPACK driver routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ, UPLO
      INTEGER            INFO, KA, KB, LDAB, LDBB, LDZ, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AB( LDAB, * ), BB( LDBB, * ), W( * ),
     $                   WORK( * ), Z( LDZ, * )
*     ..
*
*  Purpose
*  =======
*
*  DSBGV computes all the eigenvalues, and optionally, the eigenvectors
*  of a real generalized symmetric-definite banded eigenproblem, of
*  the form A*x=(lambda)*B*x. Here A and B are assumed to be symmetric
*  and banded, and B is also positive definite.
*
*  Arguments
*  =========
*
*  JOBZ    (input) CHARACTER*1
*          = 'N':  Compute eigenvalues only;
*          = 'V':  Compute eigenvalues and eigenvectors.
*
*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangles of A and B are stored;
*          = 'L':  Lower triangles of A and B are stored.
*
*  N       (input) INTEGER
*          The order of the matrices A and B.  N >= 0.
*
*  KA      (input) INTEGER
*          The number of superdiagonals of the matrix A if UPLO = 'U',
*          or the number of subdiagonals if UPLO = 'L'. KA >= 0.
*
*  KB      (input) INTEGER
*          The number of superdiagonals of the matrix B if UPLO = 'U',
*          or the number of subdiagonals if UPLO = 'L'. KB >= 0.
*
*  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
*          On entry, the upper or lower triangle of the symmetric band
*          matrix A, stored in the first ka+1 rows of the array.  The
*          j-th column of A is stored in the j-th column of the array AB
*          as follows:
*          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka).
*
*          On exit, the contents of AB are destroyed.
*
*  LDAB    (input) INTEGER
*          The leading dimension of the array AB.  LDAB >= KA+1.
*
*  BB      (input/output) DOUBLE PRECISION array, dimension (LDBB, N)
*          On entry, the upper or lower triangle of the symmetric band
*          matrix B, stored in the first kb+1 rows of the array.  The
*          j-th column of B is stored in the j-th column of the array BB
*          as follows:
*          if UPLO = 'U', BB(kb+1+i-j,j) = B(i,j) for max(1,j-kb)<=i<=j;
*          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=min(n,j+kb).
*
*          On exit, the factor S from the split Cholesky factorization
*          B = S**T*S, as returned by DPBSTF.
*
*  LDBB    (input) INTEGER
*          The leading dimension of the array BB.  LDBB >= KB+1.
*
*  W       (output) DOUBLE PRECISION array, dimension (N)
*          If INFO = 0, the eigenvalues in ascending order.
*
*  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
*          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
*          eigenvectors, with the i-th column of Z holding the
*          eigenvector associated with W(i). The eigenvectors are
*          normalized so that Z**T*B*Z = I.
*          If JOBZ = 'N', then Z is not referenced.
*
*  LDZ     (input) INTEGER
*          The leading dimension of the array Z.  LDZ >= 1, and if
*          JOBZ = 'V', LDZ >= N.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  if INFO = i, and i is:
*             <= N:  the algorithm failed to converge:
*                    i off-diagonal elements of an intermediate
*                    tridiagonal form did not converge to zero;
*             > N:   if INFO = N + i, for 1 <= i <= N, then DPBSTF
*                    returned INFO = i: B is not positive definite.
*                    The factorization of B could not be completed and
*                    no eigenvalues or eigenvectors were computed.
*
*  =====================================================================
*
*     .. Local Scalars ..
      LOGICAL            UPPER, WANTZ
      CHARACTER          VECT
      INTEGER            IINFO, INDE, INDWRK
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DPBSTF, DSBGST, DSBTRD, DSTEQR, DSTERF, XERBLA
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      WANTZ = LSAME( JOBZ, 'V' )
      UPPER = LSAME( UPLO, 'U' )
*
      INFO = 0
      IF( .NOT.( WANTZ .OR. LSAME( JOBZ, 'N' ) ) ) THEN
         INFO = -1
      ELSE IF( .NOT.( UPPER .OR. LSAME( UPLO, 'L' ) ) ) THEN
         INFO = -2
      ELSE IF( N.LT.0 ) THEN
         INFO = -3
      ELSE IF( KA.LT.0 ) THEN
         INFO = -4
      ELSE IF( KB.LT.0 .OR. KB.GT.KA ) THEN
         INFO = -5
      ELSE IF( LDAB.LT.KA+1 ) THEN
         INFO = -7
      ELSE IF( LDBB.LT.KB+1 ) THEN
         INFO = -9
      ELSE IF( LDZ.LT.1 .OR. ( WANTZ .AND. LDZ.LT.N ) ) THEN
         INFO = -12
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSBGV ', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
*     Form a split Cholesky factorization of B.
*
      CALL DPBSTF( UPLO, N, KB, BB, LDBB, INFO )
      IF( INFO.NE.0 ) THEN
         INFO = N + INFO
         RETURN
      END IF
*
*     Transform problem to standard eigenvalue problem.
*
      INDE = 1
      INDWRK = INDE + N
      CALL DSBGST( JOBZ, UPLO, N, KA, KB, AB, LDAB, BB, LDBB, Z, LDZ,
     $             WORK( INDWRK ), IINFO )
*
*     Reduce to tridiagonal form.
*
      IF( WANTZ ) THEN
         VECT = 'U'
      ELSE
         VECT = 'N'
      END IF
      CALL DSBTRD( VECT, UPLO, N, KA, AB, LDAB, W, WORK( INDE ), Z, LDZ,
     $             WORK( INDWRK ), IINFO )
*
*     For eigenvalues only, call DSTERF.  For eigenvectors, call SSTEQR.
*
      IF( .NOT.WANTZ ) THEN
         CALL DSTERF( N, W, WORK( INDE ), INFO )
      ELSE
         CALL DSTEQR( JOBZ, N, W, WORK( INDE ), Z, LDZ, WORK( INDWRK ),
     $                INFO )
      END IF
      RETURN
*
*     End of DSBGV
*
      END
