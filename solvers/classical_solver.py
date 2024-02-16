from numpy import ndarray, diag, zeros
from numpy.linalg import norm
from .base_solver import Solver

class ClassicalSolver(Solver):
    def solve(self, M: ndarray, c: ndarray) -> ndarray:
        M_diag = diag(M)

        S = diag(M_diag)
        S_inv = diag(1 / M_diag)
    
        # Subtract S from M to get another matrix T
        T = M - S
    
        # Initialize x with zeros
        x = zeros(c.shape)
        
        # Residual at timestep 0
        res0 = norm(c - M @ x)
    
        # Iterate 100 times
        for i in range(100000):
            x = S_inv @ (c - T @ x)
            if norm(c - M @ x) < res0:
                break
            
        return x
