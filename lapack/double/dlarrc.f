      SUBROUTINE DLARRC( JOBT, N, VL, VU, D, E, PIVMIN,
     $                            EIGCNT, LCNT, RCNT, INFO )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          JOBT
      INTEGER            EIGCNT, INFO, LCNT, N, RCNT
      DOUBLE PRECISION   PIVMIN, VL, VU
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), E( * )
*     ..
*
*  Purpose
*  =======
*
*  Find the number of eigenvalues of the symmetric tridiagonal matrix T
*  that are in the interval (VL,VU] if JOBT = 'T', and of L D L^T
*  if JOBT = 'L'.
*
*  Arguments
*  =========
*
*  JOBT    (input) CHARACTER*1
*          = 'T':  Compute Sturm count for matrix T.
*          = 'L':  Compute Sturm count for matrix L D L^T.
*
*  N       (input) INTEGER
*          The order of the matrix. N > 0.
*
*  VL      (input) DOUBLE PRECISION
*  VU      (input) DOUBLE PRECISION
*          The lower and upper bounds for the eigenvalues.
*
*  D       (input) DOUBLE PRECISION array, dimension (N)
*          JOBT = 'T': The N diagonal elements of the tridiagonal matrix T.
*          JOBT = 'L': The N diagonal elements of the diagonal matrix D.
*
*  E       (input) DOUBLE PRECISION array, dimension (N)
*          JOBT = 'T': The N-1 offdiagonal elements of the matrix T.
*          JOBT = 'L': The N-1 offdiagonal elements of the matrix L.
*
*  PIVMIN  (input) DOUBLE PRECISION
*          The minimum pivot in the Sturm sequence for T.
*
*  EIGCNT  (output) INTEGER
*          The number of eigenvalues of the symmetric tridiagonal matrix T
*          that are in the interval (VL,VU]
*
*  LCNT    (output) INTEGER
*  RCNT    (output) INTEGER
*          The left and right negcounts of the interval.
*
*  INFO    (output) INTEGER
*
*  Further Details
*  ===============
*
*  Based on contributions by
*     Beresford Parlett, University of California, Berkeley, USA
*     Jim Demmel, University of California, Berkeley, USA
*     Inderjit Dhillon, University of Texas, Austin, USA
*     Osni Marques, LBNL/NERSC, USA
*     Christof Voemel, University of California, Berkeley, USA
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I
      LOGICAL            MATT
      DOUBLE PRECISION   LPIVOT, RPIVOT, SL, SU, TMP, TMP2

*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      LCNT = 0
      RCNT = 0
      EIGCNT = 0
      MATT = LSAME( JOBT, 'T' )


      IF (MATT) THEN
*        Sturm sequence count on T
         LPIVOT = D( 1 ) - VL
         RPIVOT = D( 1 ) - VU
         IF( LPIVOT.LE.ZERO ) THEN
            LCNT = LCNT + 1
         ENDIF
         IF( RPIVOT.LE.ZERO ) THEN
            RCNT = RCNT + 1
         ENDIF
         DO 10 I = 1, N-1
            TMP = E(I)**2
            LPIVOT = ( D( I+1 )-VL ) - TMP/LPIVOT
            RPIVOT = ( D( I+1 )-VU ) - TMP/RPIVOT
            IF( LPIVOT.LE.ZERO ) THEN
               LCNT = LCNT + 1
            ENDIF
            IF( RPIVOT.LE.ZERO ) THEN
               RCNT = RCNT + 1
            ENDIF
 10      CONTINUE
      ELSE
*        Sturm sequence count on L D L^T
         SL = -VL
         SU = -VU
         DO 20 I = 1, N - 1
            LPIVOT = D( I ) + SL
            RPIVOT = D( I ) + SU
            IF( LPIVOT.LE.ZERO ) THEN
               LCNT = LCNT + 1
            ENDIF
            IF( RPIVOT.LE.ZERO ) THEN
               RCNT = RCNT + 1
            ENDIF
            TMP = E(I) * D(I) * E(I)
*
            TMP2 = TMP / LPIVOT
            IF( TMP2.EQ.ZERO ) THEN
               SL =  TMP - VL
            ELSE
               SL = SL*TMP2 - VL
            END IF
*
            TMP2 = TMP / RPIVOT
            IF( TMP2.EQ.ZERO ) THEN
               SU =  TMP - VU
            ELSE
               SU = SU*TMP2 - VU
            END IF
 20      CONTINUE
         LPIVOT = D( N ) + SL
         RPIVOT = D( N ) + SU
         IF( LPIVOT.LE.ZERO ) THEN
            LCNT = LCNT + 1
         ENDIF
         IF( RPIVOT.LE.ZERO ) THEN
            RCNT = RCNT + 1
         ENDIF
      ENDIF
      EIGCNT = RCNT - LCNT

      RETURN
*
*     end of DLARRC
*
      END
