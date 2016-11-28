import threading
import unittest

from pymc3 import Model, Normal


class TestModelContext(unittest.TestCase):
    def test_thread_safety(self):
        """ Regression test for issue #1552: Thread safety of model context manager

        This test creates two threads that attempt to construct two
        unrelated models at the same time.
        For repeatable testing, the two threads are syncronised such
        that thread A enters the context manager first, then B,
        then A attempts to declare a variable while B is still in the context manager.
        """
        aInCtxt,bInCtxt,aDone = [threading.Event() for k in range(3)]
        modelA = Model()
        modelB = Model()
        def make_model_a():
            with modelA:
                aInCtxt.set()
                bInCtxt.wait()
                a = Normal('a',0,1)
            aDone.set()
        def make_model_b():
            aInCtxt.wait()
            with modelB:
                bInCtxt.set()
                aDone.wait()
                b = Normal('b', 0, 1)
        threadA = threading.Thread(target=make_model_a)
        threadB = threading.Thread(target=make_model_b)
        threadA.start()
        threadB.start()
        threadA.join()
        threadB.join()
        # now let's see which model got which variable
        # previous to #1555, the variables would be swapped:
        # - B enters it's model context after A, but before a is declared -> a goes into B
        # - A leaves it's model context before B attempts to declare b. A's context manager
        #   takes B from the stack, such that b ends up in model A
        self.assertEqual(
            (
                list(modelA.named_vars),
                list(modelB.named_vars),
            ), (['a'],['b'])
        )
