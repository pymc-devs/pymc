      SUBROUTINE ILAVER( VERS_MAJOR, VERS_MINOR, VERS_PATCH )
*
*  -- LAPACK routine (version 3.1.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     January 2007
*
*     .. Scalar Arguments ..
      INTEGER VERS_MAJOR, VERS_MINOR, VERS_PATCH
*     ..
*
*  Purpose
*  =======
*
*  This subroutine return the Lapack version.
*
*  Arguments
*  =========
*
*  VERS_MAJOR   (output) INTEGER
*      return the lapack major version
*  VERS_MINOR   (output) INTEGER
*      return the lapack minor version from the major version
*  VERS_PATCH   (output) INTEGER
*      return the lapack patch version from the minor version
*
*     .. Executable Statements ..
*
      VERS_MAJOR = 3
      VERS_MINOR = 1
      VERS_PATCH = 1
*  =====================================================================
*
      RETURN
      END
