      DOUBLE PRECISION FUNCTION DLANHS( NORM, N, A, LDA, WORK )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          NORM
      INTEGER            LDA, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DLANHS  returns the value of the one norm,  or the Frobenius norm, or
*  the  infinity norm,  or the  element of  largest absolute value  of a
*  Hessenberg matrix A.
*
*  Description
*  ===========
*
*  DLANHS returns the value
*
*     DLANHS = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
*          Specifies the value to be returned in DLANHS as described
*          above.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.  When N = 0, DLANHS is
*          set to zero.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
*          The n by n upper Hessenberg matrix A; the part of A below the
*          first sub-diagonal is not referenced.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(N,1).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)),
*          where LWORK >= N when NORM = 'I'; otherwise, WORK is not
*          referenced.
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J
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
            DO 10 I = 1, MIN( N, J+1 )
               VALUE = MAX( VALUE, ABS( A( I, J ) ) )
   10       CONTINUE
   20    CONTINUE
      ELSE IF( ( LSAME( NORM, 'O' ) ) .OR. ( NORM.EQ.'1' ) ) THEN
*
*        Find norm1(A).
*
         VALUE = ZERO
         DO 40 J = 1, N
            SUM = ZERO
            DO 30 I = 1, MIN( N, J+1 )
               SUM = SUM + ABS( A( I, J ) )
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
            DO 60 I = 1, MIN( N, J+1 )
               WORK( I ) = WORK( I ) + ABS( A( I, J ) )
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
            CALL DLASSQ( MIN( N, J+1 ), A( 1, J ), 1, SCALE, SUM )
   90    CONTINUE
         VALUE = SCALE*SQRT( SUM )
      END IF
*
      DLANHS = VALUE
      RETURN
*
*     End of DLANHS
*
      END
