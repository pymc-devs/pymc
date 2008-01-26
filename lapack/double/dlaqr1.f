      SUBROUTINE DLAQR1( N, H, LDH, SR1, SI1, SR2, SI2, V )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   SI1, SI2, SR1, SR2
      INTEGER            LDH, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   H( LDH, * ), V( * )
*     ..
*
*       Given a 2-by-2 or 3-by-3 matrix H, DLAQR1 sets v to a
*       scalar multiple of the first column of the product
*
*       (*)  K = (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I)
*
*       scaling to avoid overflows and most underflows. It
*       is assumed that either
*
*               1) sr1 = sr2 and si1 = -si2
*           or
*               2) si1 = si2 = 0.
*
*       This is useful for starting double implicit shift bulges
*       in the QR algorithm.
*
*
*       N      (input) integer
*              Order of the matrix H. N must be either 2 or 3.
*
*       H      (input) DOUBLE PRECISION array of dimension (LDH,N)
*              The 2-by-2 or 3-by-3 matrix H in (*).
*
*       LDH    (input) integer
*              The leading dimension of H as declared in
*              the calling procedure.  LDH.GE.N
*
*       SR1    (input) DOUBLE PRECISION
*       SI1    The shifts in (*).
*       SR2
*       SI2
*
*       V      (output) DOUBLE PRECISION array of dimension N
*              A scalar multiple of the first column of the
*              matrix K in (*).
*
*     ================================================================
*     Based on contributions by
*        Karen Braman and Ralph Byers, Department of Mathematics,
*        University of Kansas, USA
*
*     ================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0d0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   H21S, H31S, S
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
*     ..
*     .. Executable Statements ..
      IF( N.EQ.2 ) THEN
         S = ABS( H( 1, 1 )-SR2 ) + ABS( SI2 ) + ABS( H( 2, 1 ) )
         IF( S.EQ.ZERO ) THEN
            V( 1 ) = ZERO
            V( 2 ) = ZERO
         ELSE
            H21S = H( 2, 1 ) / S
            V( 1 ) = H21S*H( 1, 2 ) + ( H( 1, 1 )-SR1 )*
     $               ( ( H( 1, 1 )-SR2 ) / S ) - SI1*( SI2 / S )
            V( 2 ) = H21S*( H( 1, 1 )+H( 2, 2 )-SR1-SR2 )
         END IF
      ELSE
         S = ABS( H( 1, 1 )-SR2 ) + ABS( SI2 ) + ABS( H( 2, 1 ) ) +
     $       ABS( H( 3, 1 ) )
         IF( S.EQ.ZERO ) THEN
            V( 1 ) = ZERO
            V( 2 ) = ZERO
            V( 3 ) = ZERO
         ELSE
            H21S = H( 2, 1 ) / S
            H31S = H( 3, 1 ) / S
            V( 1 ) = ( H( 1, 1 )-SR1 )*( ( H( 1, 1 )-SR2 ) / S ) -
     $               SI1*( SI2 / S ) + H( 1, 2 )*H21S + H( 1, 3 )*H31S
            V( 2 ) = H21S*( H( 1, 1 )+H( 2, 2 )-SR1-SR2 ) +
     $               H( 2, 3 )*H31S
            V( 3 ) = H31S*( H( 1, 1 )+H( 3, 3 )-SR1-SR2 ) +
     $               H21S*H( 3, 2 )
         END IF
      END IF
      END
