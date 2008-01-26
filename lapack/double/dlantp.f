      DOUBLE PRECISION FUNCTION DLANTP( NORM, UPLO, DIAG, N, AP, WORK )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          DIAG, NORM, UPLO
      INTEGER            N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   AP( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DLANTP  returns the value of the one norm,  or the Frobenius norm, or
*  the  infinity norm,  or the  element of  largest absolute value  of a
*  triangular matrix A, supplied in packed form.
*
*  Description
*  ===========
*
*  DLANTP returns the value
*
*     DLANTP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
*          Specifies the value to be returned in DLANTP as described
*          above.
*
*  UPLO    (input) CHARACTER*1
*          Specifies whether the matrix A is upper or lower triangular.
*          = 'U':  Upper triangular
*          = 'L':  Lower triangular
*
*  DIAG    (input) CHARACTER*1
*          Specifies whether or not the matrix A is unit triangular.
*          = 'N':  Non-unit triangular
*          = 'U':  Unit triangular
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.  When N = 0, DLANTP is
*          set to zero.
*
*  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
*          The upper or lower triangular matrix A, packed columnwise in
*          a linear array.  The j-th column of A is stored in the array
*          AP as follows:
*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
*          Note that when DIAG = 'U', the elements of the array AP
*          corresponding to the diagonal elements of the matrix A are
*          not referenced, but are assumed to be one.
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
      LOGICAL            UDIAG
      INTEGER            I, J, K
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
      INTRINSIC          ABS, MAX, SQRT
*     ..
*     .. Executable Statements ..
*
      IF( N.EQ.0 ) THEN
         VALUE = ZERO
      ELSE IF( LSAME( NORM, 'M' ) ) THEN
*
*        Find max(abs(A(i,j))).
*
         K = 1
         IF( LSAME( DIAG, 'U' ) ) THEN
            VALUE = ONE
            IF( LSAME( UPLO, 'U' ) ) THEN
               DO 20 J = 1, N
                  DO 10 I = K, K + J - 2
                     VALUE = MAX( VALUE, ABS( AP( I ) ) )
   10             CONTINUE
                  K = K + J
   20          CONTINUE
            ELSE
               DO 40 J = 1, N
                  DO 30 I = K + 1, K + N - J
                     VALUE = MAX( VALUE, ABS( AP( I ) ) )
   30             CONTINUE
                  K = K + N - J + 1
   40          CONTINUE
            END IF
         ELSE
            VALUE = ZERO
            IF( LSAME( UPLO, 'U' ) ) THEN
               DO 60 J = 1, N
                  DO 50 I = K, K + J - 1
                     VALUE = MAX( VALUE, ABS( AP( I ) ) )
   50             CONTINUE
                  K = K + J
   60          CONTINUE
            ELSE
               DO 80 J = 1, N
                  DO 70 I = K, K + N - J
                     VALUE = MAX( VALUE, ABS( AP( I ) ) )
   70             CONTINUE
                  K = K + N - J + 1
   80          CONTINUE
            END IF
         END IF
      ELSE IF( ( LSAME( NORM, 'O' ) ) .OR. ( NORM.EQ.'1' ) ) THEN
*
*        Find norm1(A).
*
         VALUE = ZERO
         K = 1
         UDIAG = LSAME( DIAG, 'U' )
         IF( LSAME( UPLO, 'U' ) ) THEN
            DO 110 J = 1, N
               IF( UDIAG ) THEN
                  SUM = ONE
                  DO 90 I = K, K + J - 2
                     SUM = SUM + ABS( AP( I ) )
   90             CONTINUE
               ELSE
                  SUM = ZERO
                  DO 100 I = K, K + J - 1
                     SUM = SUM + ABS( AP( I ) )
  100             CONTINUE
               END IF
               K = K + J
               VALUE = MAX( VALUE, SUM )
  110       CONTINUE
         ELSE
            DO 140 J = 1, N
               IF( UDIAG ) THEN
                  SUM = ONE
                  DO 120 I = K + 1, K + N - J
                     SUM = SUM + ABS( AP( I ) )
  120             CONTINUE
               ELSE
                  SUM = ZERO
                  DO 130 I = K, K + N - J
                     SUM = SUM + ABS( AP( I ) )
  130             CONTINUE
               END IF
               K = K + N - J + 1
               VALUE = MAX( VALUE, SUM )
  140       CONTINUE
         END IF
      ELSE IF( LSAME( NORM, 'I' ) ) THEN
*
*        Find normI(A).
*
         K = 1
         IF( LSAME( UPLO, 'U' ) ) THEN
            IF( LSAME( DIAG, 'U' ) ) THEN
               DO 150 I = 1, N
                  WORK( I ) = ONE
  150          CONTINUE
               DO 170 J = 1, N
                  DO 160 I = 1, J - 1
                     WORK( I ) = WORK( I ) + ABS( AP( K ) )
                     K = K + 1
  160             CONTINUE
                  K = K + 1
  170          CONTINUE
            ELSE
               DO 180 I = 1, N
                  WORK( I ) = ZERO
  180          CONTINUE
               DO 200 J = 1, N
                  DO 190 I = 1, J
                     WORK( I ) = WORK( I ) + ABS( AP( K ) )
                     K = K + 1
  190             CONTINUE
  200          CONTINUE
            END IF
         ELSE
            IF( LSAME( DIAG, 'U' ) ) THEN
               DO 210 I = 1, N
                  WORK( I ) = ONE
  210          CONTINUE
               DO 230 J = 1, N
                  K = K + 1
                  DO 220 I = J + 1, N
                     WORK( I ) = WORK( I ) + ABS( AP( K ) )
                     K = K + 1
  220             CONTINUE
  230          CONTINUE
            ELSE
               DO 240 I = 1, N
                  WORK( I ) = ZERO
  240          CONTINUE
               DO 260 J = 1, N
                  DO 250 I = J, N
                     WORK( I ) = WORK( I ) + ABS( AP( K ) )
                     K = K + 1
  250             CONTINUE
  260          CONTINUE
            END IF
         END IF
         VALUE = ZERO
         DO 270 I = 1, N
            VALUE = MAX( VALUE, WORK( I ) )
  270    CONTINUE
      ELSE IF( ( LSAME( NORM, 'F' ) ) .OR. ( LSAME( NORM, 'E' ) ) ) THEN
*
*        Find normF(A).
*
         IF( LSAME( UPLO, 'U' ) ) THEN
            IF( LSAME( DIAG, 'U' ) ) THEN
               SCALE = ONE
               SUM = N
               K = 2
               DO 280 J = 2, N
                  CALL DLASSQ( J-1, AP( K ), 1, SCALE, SUM )
                  K = K + J
  280          CONTINUE
            ELSE
               SCALE = ZERO
               SUM = ONE
               K = 1
               DO 290 J = 1, N
                  CALL DLASSQ( J, AP( K ), 1, SCALE, SUM )
                  K = K + J
  290          CONTINUE
            END IF
         ELSE
            IF( LSAME( DIAG, 'U' ) ) THEN
               SCALE = ONE
               SUM = N
               K = 2
               DO 300 J = 1, N - 1
                  CALL DLASSQ( N-J, AP( K ), 1, SCALE, SUM )
                  K = K + N - J + 1
  300          CONTINUE
            ELSE
               SCALE = ZERO
               SUM = ONE
               K = 1
               DO 310 J = 1, N
                  CALL DLASSQ( N-J+1, AP( K ), 1, SCALE, SUM )
                  K = K + N - J + 1
  310          CONTINUE
            END IF
         END IF
         VALUE = SCALE*SQRT( SUM )
      END IF
*
      DLANTP = VALUE
      RETURN
*
*     End of DLANTP
*
      END
