      DOUBLE PRECISION FUNCTION DLANGB( NORM, N, KL, KU, AB, LDAB,
     $                 WORK )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          NORM
      INTEGER            KL, KU, LDAB, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AB( LDAB, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DLANGB  returns the value of the one norm,  or the Frobenius norm, or
*  the  infinity norm,  or the element of  largest absolute value  of an
*  n by n band matrix  A,  with kl sub-diagonals and ku super-diagonals.
*
*  Description
*  ===========
*
*  DLANGB returns the value
*
*     DLANGB = ( max(abs(A(i,j))), NORM = 'M' or 'm'
*              (
*              ( norm1(A),         NORM = '1', 'O' or 'o'
*              (
*              ( normI(A),         NORM = 'I' or 'i'
*              (
*              ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
*
*  where  norm1  denotes the  one norm of a matrix (maximum column sum),
*  normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
*  normF  denotes the  Frobenius norm of a matrix (square root of sum of
*  squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.
*
*  Arguments
*  =========
*
*  NORM    (input) CHARACTER*1
*          Specifies the value to be returned in DLANGB as described
*          above.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.  When N = 0, DLANGB is
*          set to zero.
*
*  KL      (input) INTEGER
*          The number of sub-diagonals of the matrix A.  KL >= 0.
*
*  KU      (input) INTEGER
*          The number of super-diagonals of the matrix A.  KU >= 0.
*
*  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
*          The band matrix A, stored in rows 1 to KL+KU+1.  The j-th
*          column of A is stored in the j-th column of the array AB as
*          follows:
*          AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(n,j+kl).
*
*  LDAB    (input) INTEGER
*          The leading dimension of the array AB.  LDAB >= KL+KU+1.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)),
*          where LWORK >= N when NORM = 'I'; otherwise, WORK is not
*          referenced.
*
* =====================================================================
*
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J, K, L
      DOUBLE PRECISION   SCALE, SUM, VALUE
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLASSQ
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN, SQRT
*     ..
*     .. Executable Statements ..
*
      IF( N.EQ.0 ) THEN
         VALUE = ZERO
      ELSE IF( LSAME( NORM, 'M' ) ) THEN
*
*        Find max(abs(A(i,j))).
*
         VALUE = ZERO
         DO 20 J = 1, N
            DO 10 I = MAX( KU+2-J, 1 ), MIN( N+KU+1-J, KL+KU+1 )
               VALUE = MAX( VALUE, ABS( AB( I, J ) ) )
   10       CONTINUE
   20    CONTINUE
      ELSE IF( ( LSAME( NORM, 'O' ) ) .OR. ( NORM.EQ.'1' ) ) THEN
*
*        Find norm1(A).
*
         VALUE = ZERO
         DO 40 J = 1, N
            SUM = ZERO
            DO 30 I = MAX( KU+2-J, 1 ), MIN( N+KU+1-J, KL+KU+1 )
               SUM = SUM + ABS( AB( I, J ) )
   30       CONTINUE
            VALUE = MAX( VALUE, SUM )
   40    CONTINUE
      ELSE IF( LSAME( NORM, 'I' ) ) THEN
*
*        Find normI(A).
*
         DO 50 I = 1, N
            WORK( I ) = ZERO
   50    CONTINUE
         DO 70 J = 1, N
            K = KU + 1 - J
            DO 60 I = MAX( 1, J-KU ), MIN( N, J+KL )
               WORK( I ) = WORK( I ) + ABS( AB( K+I, J ) )
   60       CONTINUE
   70    CONTINUE
         VALUE = ZERO
         DO 80 I = 1, N
            VALUE = MAX( VALUE, WORK( I ) )
   80    CONTINUE
      ELSE IF( ( LSAME( NORM, 'F' ) ) .OR. ( LSAME( NORM, 'E' ) ) ) THEN
*
*        Find normF(A).
*
         SCALE = ZERO
         SUM = ONE
         DO 90 J = 1, N
            L = MAX( 1, J-KU )
            K = KU + 1 - J + L
            CALL DLASSQ( MIN( N, J+KL )-L+1, AB( K, J ), 1, SCALE, SUM )
   90    CONTINUE
         VALUE = SCALE*SQRT( SUM )
      END IF
*
      DLANGB = VALUE
      RETURN
*
*     End of DLANGB
*
      END
