import unittest
import numpy as np
from solvers import JacobiSolver

class TestJacobiSolver(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName) 
        self.solver = JacobiSolver()


    def test_solve(self):
        M = np.array([[3., 2.], [2., 6.]])
        b = np.array([[2.],[-8.]])

        x_true = np.linalg.inv(M) @ b

        x = self.solver.solve(M, b)

        np.testing.assert_allclose(x, x_true, atol=1e-6)


if __name__ == '__main__':
    unittest.main()