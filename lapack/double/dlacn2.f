      SUBROUTINE DLACN2( N, V, X, ISGN, EST, KASE, ISAVE )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            KASE, N
      DOUBLE PRECISION   EST
*     ..
*     .. Array Arguments ..
      INTEGER            ISGN( * ), ISAVE( 3 )
      DOUBLE PRECISION   V( * ), X( * )
*     ..
*
*  Purpose
*  =======
*
*  DLACN2 estimates the 1-norm of a square, real matrix A.
*  Reverse communication is used for evaluating matrix-vector products.
*
*  Arguments
*  =========
*
*  N      (input) INTEGER
*         The order of the matrix.  N >= 1.
*
*  V      (workspace) DOUBLE PRECISION array, dimension (N)
*         On the final return, V = A*W,  where  EST = norm(V)/norm(W)
*         (W is not returned).
*
*  X      (input/output) DOUBLE PRECISION array, dimension (N)
*         On an intermediate return, X should be overwritten by
*               A * X,   if KASE=1,
*               A' * X,  if KASE=2,
*         and DLACN2 must be re-called with all the other parameters
*         unchanged.
*
*  ISGN   (workspace) INTEGER array, dimension (N)
*
*  EST    (input/output) DOUBLE PRECISION
*         On entry with KASE = 1 or 2 and ISAVE(1) = 3, EST should be
*         unchanged from the previous call to DLACN2.
*         On exit, EST is an estimate (a lower bound) for norm(A). 
*
*  KASE   (input/output) INTEGER
*         On the initial call to DLACN2, KASE should be 0.
*         On an intermediate return, KASE will be 1 or 2, indicating
*         whether X should be overwritten by A * X  or A' * X.
*         On the final return from DLACN2, KASE will again be 0.
*
*  ISAVE  (input/output) INTEGER array, dimension (3)
*         ISAVE is used to save variables between calls to DLACN2
*
*  Further Details
*  ======= =======
*
*  Contributed by Nick Higham, University of Manchester.
*  Originally named SONEST, dated March 16, 1988.
*
*  Reference: N.J. Higham, "FORTRAN codes for estimating the one-norm of
*  a real or complex matrix, with applications to condition estimation",
*  ACM Trans. Math. Soft., vol. 14, no. 4, pp. 381-396, December 1988.
*
*  This is a thread safe version of DLACON, which uses the array ISAVE
*  in place of a SAVE statement, as follows:
*
*     DLACON     DLACN2
*      JUMP     ISAVE(1)
*      J        ISAVE(2)
*      ITER     ISAVE(3)
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            ITMAX
      PARAMETER          ( ITMAX = 5 )
      DOUBLE PRECISION   ZERO, ONE, TWO
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0, TWO = 2.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, JLAST
      DOUBLE PRECISION   ALTSGN, ESTOLD, TEMP
*     ..
*     .. External Functions ..
      INTEGER            IDAMAX
      DOUBLE PRECISION   DASUM
      EXTERNAL           IDAMAX, DASUM
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, NINT, SIGN
*     ..
*     .. Executable Statements ..
*
      IF( KASE.EQ.0 ) THEN
         DO 10 I = 1, N
            X( I ) = ONE / DBLE( N )
   10    CONTINUE
         KASE = 1
         ISAVE( 1 ) = 1
         RETURN
      END IF
*
      GO TO ( 20, 40, 70, 110, 140 )ISAVE( 1 )
*
*     ................ ENTRY   (ISAVE( 1 ) = 1)
*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
*
   20 CONTINUE
      IF( N.EQ.1 ) THEN
         V( 1 ) = X( 1 )
         EST = ABS( V( 1 ) )
*        ... QUIT
         GO TO 150
      END IF
      EST = DASUM( N, X, 1 )
*
      DO 30 I = 1, N
         X( I ) = SIGN( ONE, X( I ) )
         ISGN( I ) = NINT( X( I ) )
   30 CONTINUE
      KASE = 2
      ISAVE( 1 ) = 2
      RETURN
*
*     ................ ENTRY   (ISAVE( 1 ) = 2)
*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
*
   40 CONTINUE
      ISAVE( 2 ) = IDAMAX( N, X, 1 )
      ISAVE( 3 ) = 2
*
*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
*
   50 CONTINUE
      DO 60 I = 1, N
         X( I ) = ZERO
   60 CONTINUE
      X( ISAVE( 2 ) ) = ONE
      KASE = 1
      ISAVE( 1 ) = 3
      RETURN
*
*     ................ ENTRY   (ISAVE( 1 ) = 3)
*     X HAS BEEN OVERWRITTEN BY A*X.
*
   70 CONTINUE
      CALL DCOPY( N, X, 1, V, 1 )
      ESTOLD = EST
      EST = DASUM( N, V, 1 )
      DO 80 I = 1, N
         IF( NINT( SIGN( ONE, X( I ) ) ).NE.ISGN( I ) )
     $      GO TO 90
   80 CONTINUE
*     REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED.
      GO TO 120
*
   90 CONTINUE
*     TEST FOR CYCLING.
      IF( EST.LE.ESTOLD )
     $   GO TO 120
*
      DO 100 I = 1, N
         X( I ) = SIGN( ONE, X( I ) )
         ISGN( I ) = NINT( X( I ) )
  100 CONTINUE
      KASE = 2
      ISAVE( 1 ) = 4
      RETURN
*
*     ................ ENTRY   (ISAVE( 1 ) = 4)
*     X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
*
  110 CONTINUE
      JLAST = ISAVE( 2 )
      ISAVE( 2 ) = IDAMAX( N, X, 1 )
      IF( ( X( JLAST ).NE.ABS( X( ISAVE( 2 ) ) ) ) .AND.
     $    ( ISAVE( 3 ).LT.ITMAX ) ) THEN
         ISAVE( 3 ) = ISAVE( 3 ) + 1
         GO TO 50
      END IF
*
*     ITERATION COMPLETE.  FINAL STAGE.
*
  120 CONTINUE
      ALTSGN = ONE
      DO 130 I = 1, N
         X( I ) = ALTSGN*( ONE+DBLE( I-1 ) / DBLE( N-1 ) )
         ALTSGN = -ALTSGN
  130 CONTINUE
      KASE = 1
      ISAVE( 1 ) = 5
      RETURN
*
*     ................ ENTRY   (ISAVE( 1 ) = 5)
*     X HAS BEEN OVERWRITTEN BY A*X.
*
  140 CONTINUE
      TEMP = TWO*( DASUM( N, X, 1 ) / DBLE( 3*N ) )
      IF( TEMP.GT.EST ) THEN
         CALL DCOPY( N, X, 1, V, 1 )
         EST = TEMP
      END IF
*
  150 CONTINUE
      KASE = 0
      RETURN
*
*     End of DLACN2
*
      END
