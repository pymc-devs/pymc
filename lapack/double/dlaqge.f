      SUBROUTINE DLAQGE( M, N, A, LDA, R, C, ROWCND, COLCND, AMAX,
     $                   EQUED )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          EQUED
      INTEGER            LDA, M, N
      DOUBLE PRECISION   AMAX, COLCND, ROWCND
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), C( * ), R( * )
*     ..
*
*  Purpose
*  =======
*
*  DLAQGE equilibrates a general M by N matrix A using the row and
*  column scaling factors in the vectors R and C.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M by N matrix A.
*          On exit, the equilibrated matrix.  See EQUED for the form of
*          the equilibrated matrix.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(M,1).
*
*  R       (input) DOUBLE PRECISION array, dimension (M)
*          The row scale factors for A.
*
*  C       (input) DOUBLE PRECISION array, dimension (N)
*          The column scale factors for A.
*
*  ROWCND  (input) DOUBLE PRECISION
*          Ratio of the smallest R(i) to the largest R(i).
*
*  COLCND  (input) DOUBLE PRECISION
*          Ratio of the smallest C(i) to the largest C(i).
*
*  AMAX    (input) DOUBLE PRECISION
*          Absolute value of largest matrix entry.
*
*  EQUED   (output) CHARACTER*1
*          Specifies the form of equilibration that was done.
*          = 'N':  No equilibration
*          = 'R':  Row equilibration, i.e., A has been premultiplied by
*                  diag(R).
*          = 'C':  Column equilibration, i.e., A has been postmultiplied
*                  by diag(C).
*          = 'B':  Both row and column equilibration, i.e., A has been
*                  replaced by diag(R) * A * diag(C).
*
*  Internal Parameters
*  ===================
*
*  THRESH is a threshold value used to decide if row or column scaling
*  should be done based on the ratio of the row or column scaling
*  factors.  If ROWCND < THRESH, row scaling is done, and if
*  COLCND < THRESH, column scaling is done.
*
*  LARGE and SMALL are threshold values used to decide if row scaling
*  should be done based on the absolute size of the largest matrix
*  element.  If AMAX > LARGE or AMAX < SMALL, row scaling is done.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, THRESH
      PARAMETER          ( ONE = 1.0D+0, THRESH = 0.1D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J
      DOUBLE PRECISION   CJ, LARGE, SMALL
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DLAMCH
      EXTERNAL           DLAMCH
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( M.LE.0 .OR. N.LE.0 ) THEN
         EQUED = 'N'
         RETURN
      END IF
*
*     Initialize LARGE and SMALL.
*
      SMALL = DLAMCH( 'Safe minimum' ) / DLAMCH( 'Precision' )
      LARGE = ONE / SMALL
*
      IF( ROWCND.GE.THRESH .AND. AMAX.GE.SMALL .AND. AMAX.LE.LARGE )
     $     THEN
*
*        No row scaling
*
         IF( COLCND.GE.THRESH ) THEN
*
*           No column scaling
*
            EQUED = 'N'
         ELSE
*
*           Column scaling
*
            DO 20 J = 1, N
               CJ = C( J )
               DO 10 I = 1, M
                  A( I, J ) = CJ*A( I, J )
   10          CONTINUE
   20       CONTINUE
            EQUED = 'C'
         END IF
      ELSE IF( COLCND.GE.THRESH ) THEN
*
*        Row scaling, no column scaling
*
         DO 40 J = 1, N
            DO 30 I = 1, M
               A( I, J ) = R( I )*A( I, J )
   30       CONTINUE
   40    CONTINUE
         EQUED = 'R'
      ELSE
*
*        Row and column scaling
*
         DO 60 J = 1, N
            CJ = C( J )
            DO 50 I = 1, M
               A( I, J ) = CJ*R( I )*A( I, J )
   50       CONTINUE
   60    CONTINUE
         EQUED = 'B'
      END IF
*
      RETURN
*
*     End of DLAQGE
*
      END
