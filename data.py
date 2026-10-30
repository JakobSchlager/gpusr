# simple script as a template to use for data generation

import numpy as np

rng = np.random.default_rng(42)

# generate 100 observations with three variables
X = rng.uniform(-2, 2, size=(100, 3))

x1, x2, x3 = X.T
y = x1**2 + x2 + x3

data = np.column_stack((X, y))

np.savetxt("data/poly.txt", data, fmt="%.8f")