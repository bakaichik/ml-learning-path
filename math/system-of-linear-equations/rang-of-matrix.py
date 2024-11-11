import numpy as np
import pandas as pd

a = np.array([
    [1, 2, 3],
    [1, -2, 3],
    [-1, -2, -3]
])

r = np.linalg.matrix_rank(a)

print(f"Rang of the matrix A: {r}")