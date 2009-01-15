/* c2f.c - call C functions from fortran */

/* Fortran and C have different calling conventions. This file does
   impedance matching between the two. */

extern double cephes_i0(double);

double i0_(x)
  double* x;
{
  double result;
  result = cephes_i0(*x);
  return result;
}
