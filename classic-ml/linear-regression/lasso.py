import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2 + x[2]**4

def grad(x):
    return np.array([2*x[0], 4*x[1], 4*x[2]**3])

# Parameters
x = np.array([1.5, 2, 3])

gamma = 1e-2
max_iter = 10
eps = 1e-4

for i in range(max_iter):
    gradient = grad(x)
    # Check stopping condition based on gradient norm
    if np.linalg.norm(gradient) < eps:
        break

    # Update x
    x = x - gamma * gradient

    # Print function value at new x
    print(f(x))

print("Optimized x:", x)
print("Function value at optimized x:", f(x))
