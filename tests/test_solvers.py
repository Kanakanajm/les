import unittest
from numpy import array
from numpy.linalg import inv
from solvers import ClassicalSolver

class TestClassicalSolver(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName) 
        self.solver = ClassicalSolver()


    def test_solve(self):
        M = array([[1., 2.], [3., 4.]])
        c = array([[5.],[6.]])
        x = inv(M) @ c
        result = self.solver.solve(M, c)
        self.assertEqual(result, x)


if __name__ == '__main__':
    unittest.main()