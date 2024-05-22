import numpy as np
from .base_solver import Solver

class JacobiSolver(Solver):
    def solve(self, A: np.ndarray, b: np.ndarray, num_iters = 100) -> np.ndarray:
        """
        Solve the system Ax = b using Jacobi iteration.
        A = D + E
        x = -inv(D)Ex + inv(D)b is the termination condition

        B = -inv(D)E
        z = inv(D)b
        
        """
        assert A.shape[0] == A.shape[1] # M must be a square matrix
        assert A.shape[0] == b.shape[0] # Mx = c must be valid


        D = np.diag(np.diag(A))
        Dinv = np.diag(1/np.diag(A)) # cannot use 1/D because off-diagonal elements will be divided by zero i.e. 1/0 = inf
        E = A - D
        B = -Dinv @ E

        # spectral radius of B must be less than 1 for convergence
        # this can be observed as following
        # suppose xbar is the true solution, let x_k = xbar + e_k
        # then x_{k+1}  = B(xbar + e_k) + z
        #               = B xbar + B e_k + z
        #               = xbar + B e_k (since B xbar + z = xbar)
        # we got x_{k+1} - xbar = B e_k
        # we got e_{k+1} = B e_k
        assert np.linalg.eigvals(B).max() < 1

        z = Dinv @ b

        x0 = np.zeros_like(b)
        x_kp1 = x0
        print("initial error: {0:.4f}".format(np.linalg.norm(A @ x_kp1 - b)))

        for i in range(num_iters):
            x_k = x_kp1
            x_kp1 = B @ x_k + z

            if i % 1 == 0:
                print("error after iteration {0}: {1:.4f}".format(i, np.linalg.norm(A @ x_kp1 - b)))

        return x_kp1
