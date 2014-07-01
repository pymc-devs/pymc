from numpy.testing import *
from numpy import *
from pymc.examples import disaster_model as DM
from pymc import Container, Normal


class test_Container(TestCase):

    def test_container_parents(self):
        A = Normal('A', 0, 1)
        B = Normal('B', 0, 1)

        C = Normal('C', [A, B], 1)

        assert_equal(Container([A, B]).value, [A.value, B.value])
        assert_equal(C.parents.value['mu'], [A.value, B.value])

    def test_nested_tuple_container(self):
        A = Normal('A', 0, 1)
        try:
            Container(([A],))
            raise AssertionError('A NotImplementedError should have resulted.')
        except NotImplementedError:
            pass

    def test(self):

        # Test set container:

        S = [DM.early_mean, DM.switchpoint, DM.late_mean]
        R = Container(S)

        for item in R:
            assert(item in S)

        for item in R.value:
            val_in_S = False
            for S_item in S:
                if S_item.value is item:
                    val_in_S = True

            assert(val_in_S)

        # Test list/dictionary container:

        A = [[DM.early_mean, DM.switchpoint],
             [DM.late_mean, DM.disasters, 3.], 54.323]
        C = Container(A)

        for i in range(2):
            for j in range(2):
                assert(C[i][j] == A[i][j])
                assert(all(C.value[i][j] == A[i][j].value))

            assert(C[1][2] == A[1][2])
            assert(C.value[1][2] == A[1][2])

            assert(C[2] == A[2])
            assert(C.value[2] == A[2])

        # Test array container:

        B = ndarray((3, 3), dtype=object)
        B[0, :] = DM.early_mean
        B[1, :] = 1.
        B[2, :] = A
        D = Container(B)

        for i in range(2):
            assert(D[0, i] == DM.early_mean)
            assert(D.value[0, i] == DM.early_mean.value)
            assert(D[1, i] == 1.)
            assert(D.value[1, i] == 1.)

        P = D[2, :]
        Q = D.value[2, :]

        for i in range(2):
            for j in range(2):
                assert(P[i][j] == A[i][j])
                assert(all(Q[i][j] == A[i][j].value))

            assert(P[1][2] == A[1][2])
            assert(Q[1][2] == A[1][2])

            assert(P[2] == A[2])
            assert(Q[2] == A[2])


if __name__ == '__main__':
    import unittest
    unittest.main()
