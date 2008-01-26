      SUBROUTINE DLAGS2( UPPER, A1, A2, A3, B1, B2, B3, CSU, SNU, CSV,
     $                   SNV, CSQ, SNQ )
*
*  -- LAPACK auxiliary routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      LOGICAL            UPPER
      DOUBLE PRECISION   A1, A2, A3, B1, B2, B3, CSQ, CSU, CSV, SNQ,
     $                   SNU, SNV
*     ..
*
*  Purpose
*  =======
*
*  DLAGS2 computes 2-by-2 orthogonal matrices U, V and Q, such
*  that if ( UPPER ) then
*
*            U'*A*Q = U'*( A1 A2 )*Q = ( x  0  )
*                        ( 0  A3 )     ( x  x  )
*  and
*            V'*B*Q = V'*( B1 B2 )*Q = ( x  0  )
*                        ( 0  B3 )     ( x  x  )
*
*  or if ( .NOT.UPPER ) then
*
*            U'*A*Q = U'*( A1 0  )*Q = ( x  x  )
*                        ( A2 A3 )     ( 0  x  )
*  and
*            V'*B*Q = V'*( B1 0  )*Q = ( x  x  )
*                        ( B2 B3 )     ( 0  x  )
*
*  The rows of the transformed A and B are parallel, where
*
*    U = (  CSU  SNU ), V = (  CSV SNV ), Q = (  CSQ   SNQ )
*        ( -SNU  CSU )      ( -SNV CSV )      ( -SNQ   CSQ )
*
*  Z' denotes the transpose of Z.
*
*
*  Arguments
*  =========
*
*  UPPER   (input) LOGICAL
*          = .TRUE.: the input matrices A and B are upper triangular.
*          = .FALSE.: the input matrices A and B are lower triangular.
*
*  A1      (input) DOUBLE PRECISION
*  A2      (input) DOUBLE PRECISION
*  A3      (input) DOUBLE PRECISION
*          On entry, A1, A2 and A3 are elements of the input 2-by-2
*          upper (lower) triangular matrix A.
*
*  B1      (input) DOUBLE PRECISION
*  B2      (input) DOUBLE PRECISION
*  B3      (input) DOUBLE PRECISION
*          On entry, B1, B2 and B3 are elements of the input 2-by-2
*          upper (lower) triangular matrix B.
*
*  CSU     (output) DOUBLE PRECISION
*  SNU     (output) DOUBLE PRECISION
*          The desired orthogonal matrix U.
*
*  CSV     (output) DOUBLE PRECISION
*  SNV     (output) DOUBLE PRECISION
*          The desired orthogonal matrix V.
*
*  CSQ     (output) DOUBLE PRECISION
*  SNQ     (output) DOUBLE PRECISION
*          The desired orthogonal matrix Q.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   A, AUA11, AUA12, AUA21, AUA22, AVB11, AVB12,
     $                   AVB21, AVB22, B, C, CSL, CSR, D, R, S1, S2,
     $                   SNL, SNR, UA11, UA11R, UA12, UA21, UA22, UA22R,
     $                   VB11, VB11R, VB12, VB21, VB22, VB22R
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLARTG, DLASV2
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
*     ..
*     .. Executable Statements ..
*
      IF( UPPER ) THEN
*
*        Input matrices A and B are upper triangular matrices
*
*        Form matrix C = A*adj(B) = ( a b )
*                                   ( 0 d )
*
         A = A1*B3
         D = A3*B1
         B = A2*B1 - A1*B2
*
*        The SVD of real 2-by-2 triangular C
*
*         ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
*         ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
*
         CALL DLASV2( A, B, D, S1, S2, SNR, CSR, SNL, CSL )
*
         IF( ABS( CSL ).GE.ABS( SNL ) .OR. ABS( CSR ).GE.ABS( SNR ) )
     $        THEN
*
*           Compute the (1,1) and (1,2) elements of U'*A and V'*B,
*           and (1,2) element of |U|'*|A| and |V|'*|B|.
*
            UA11R = CSL*A1
            UA12 = CSL*A2 + SNL*A3
*
            VB11R = CSR*B1
            VB12 = CSR*B2 + SNR*B3
*
            AUA12 = ABS( CSL )*ABS( A2 ) + ABS( SNL )*ABS( A3 )
            AVB12 = ABS( CSR )*ABS( B2 ) + ABS( SNR )*ABS( B3 )
*
*           zero (1,2) elements of U'*A and V'*B
*
            IF( ( ABS( UA11R )+ABS( UA12 ) ).NE.ZERO ) THEN
               IF( AUA12 / ( ABS( UA11R )+ABS( UA12 ) ).LE.AVB12 /
     $             ( ABS( VB11R )+ABS( VB12 ) ) ) THEN
                  CALL DLARTG( -UA11R, UA12, CSQ, SNQ, R )
               ELSE
                  CALL DLARTG( -VB11R, VB12, CSQ, SNQ, R )
               END IF
            ELSE
               CALL DLARTG( -VB11R, VB12, CSQ, SNQ, R )
            END IF
*
            CSU = CSL
            SNU = -SNL
            CSV = CSR
            SNV = -SNR
*
         ELSE
*
*           Compute the (2,1) and (2,2) elements of U'*A and V'*B,
*           and (2,2) element of |U|'*|A| and |V|'*|B|.
*
            UA21 = -SNL*A1
            UA22 = -SNL*A2 + CSL*A3
*
            VB21 = -SNR*B1
            VB22 = -SNR*B2 + CSR*B3
*
            AUA22 = ABS( SNL )*ABS( A2 ) + ABS( CSL )*ABS( A3 )
            AVB22 = ABS( SNR )*ABS( B2 ) + ABS( CSR )*ABS( B3 )
*
*           zero (2,2) elements of U'*A and V'*B, and then swap.
*
            IF( ( ABS( UA21 )+ABS( UA22 ) ).NE.ZERO ) THEN
               IF( AUA22 / ( ABS( UA21 )+ABS( UA22 ) ).LE.AVB22 /
     $             ( ABS( VB21 )+ABS( VB22 ) ) ) THEN
                  CALL DLARTG( -UA21, UA22, CSQ, SNQ, R )
               ELSE
                  CALL DLARTG( -VB21, VB22, CSQ, SNQ, R )
               END IF
            ELSE
               CALL DLARTG( -VB21, VB22, CSQ, SNQ, R )
            END IF
*
            CSU = SNL
            SNU = CSL
            CSV = SNR
            SNV = CSR
*
         END IF
*
      ELSE
*
*        Input matrices A and B are lower triangular matrices
*
*        Form matrix C = A*adj(B) = ( a 0 )
*                                   ( c d )
*
         A = A1*B3
         D = A3*B1
         C = A2*B3 - A3*B2
*
*        The SVD of real 2-by-2 triangular C
*
*         ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
*         ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
*
         CALL DLASV2( A, C, D, S1, S2, SNR, CSR, SNL, CSL )
*
         IF( ABS( CSR ).GE.ABS( SNR ) .OR. ABS( CSL ).GE.ABS( SNL ) )
     $        THEN
*
*           Compute the (2,1) and (2,2) elements of U'*A and V'*B,
*           and (2,1) element of |U|'*|A| and |V|'*|B|.
*
            UA21 = -SNR*A1 + CSR*A2
            UA22R = CSR*A3
*
            VB21 = -SNL*B1 + CSL*B2
            VB22R = CSL*B3
*
            AUA21 = ABS( SNR )*ABS( A1 ) + ABS( CSR )*ABS( A2 )
            AVB21 = ABS( SNL )*ABS( B1 ) + ABS( CSL )*ABS( B2 )
*
*           zero (2,1) elements of U'*A and V'*B.
*
            IF( ( ABS( UA21 )+ABS( UA22R ) ).NE.ZERO ) THEN
               IF( AUA21 / ( ABS( UA21 )+ABS( UA22R ) ).LE.AVB21 /
     $             ( ABS( VB21 )+ABS( VB22R ) ) ) THEN
                  CALL DLARTG( UA22R, UA21, CSQ, SNQ, R )
               ELSE
                  CALL DLARTG( VB22R, VB21, CSQ, SNQ, R )
               END IF
            ELSE
               CALL DLARTG( VB22R, VB21, CSQ, SNQ, R )
            END IF
*
            CSU = CSR
            SNU = -SNR
            CSV = CSL
            SNV = -SNL
*
         ELSE
*
*           Compute the (1,1) and (1,2) elements of U'*A and V'*B,
*           and (1,1) element of |U|'*|A| and |V|'*|B|.
*
            UA11 = CSR*A1 + SNR*A2
            UA12 = SNR*A3
*
            VB11 = CSL*B1 + SNL*B2
            VB12 = SNL*B3
*
            AUA11 = ABS( CSR )*ABS( A1 ) + ABS( SNR )*ABS( A2 )
            AVB11 = ABS( CSL )*ABS( B1 ) + ABS( SNL )*ABS( B2 )
*
*           zero (1,1) elements of U'*A and V'*B, and then swap.
*
            IF( ( ABS( UA11 )+ABS( UA12 ) ).NE.ZERO ) THEN
               IF( AUA11 / ( ABS( UA11 )+ABS( UA12 ) ).LE.AVB11 /
     $             ( ABS( VB11 )+ABS( VB12 ) ) ) THEN
                  CALL DLARTG( UA12, UA11, CSQ, SNQ, R )
               ELSE
                  CALL DLARTG( VB12, VB11, CSQ, SNQ, R )
               END IF
            ELSE
               CALL DLARTG( VB12, VB11, CSQ, SNQ, R )
            END IF
*
            CSU = SNR
            SNU = CSR
            CSV = SNL
            SNV = CSL
*
         END IF
*
      END IF
*
      RETURN
*
*     End of DLAGS2
*
      END
