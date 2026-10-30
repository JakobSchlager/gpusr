from pathlib import Path
import numpy as np

rng = np.random.default_rng(42)

N = 1000
out = Path("../data")
out.mkdir(exist_ok=True)

def save(name, fn, n_vars=3, low=-2.0, high=2.0):
    X = rng.uniform(low, high, size=(N, n_vars))

    y = fn(*X.T)

    data = np.column_stack((X, y))
    np.savetxt(out / f"{name}.txt", data, fmt="%.12g")

    print(f"{name:12} -> {data.shape}")

save("linear",   lambda x, y, z: x + y)
save("poly",     lambda x, y, z: x*x + y + z)
save("product",  lambda x, y, z: x*y + z)

save("trig",     lambda x, y, z: np.sin(x) + y)
save("rational", lambda x, y, z: x / (1 + y*y))
save("exp",      lambda x, y, z: np.exp(x) + y)
save("sinxyzq",  lambda x, y, z: np.sin(x * y) + z**2)
save("poly4",    lambda x, y, z: x**4 + x**3 + x**2 + x)
