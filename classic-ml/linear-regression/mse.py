import numpy as np

X = np.array([
    [1, 1],
    [1, 1],
    [2, 1],
    [3, 1]
])
w = np.array([2, -2])
y = np.array([-1, 2, 4, 3])

e = y - X @ w

# MSE
mse = (e ** 2).mean()

print(mse)
