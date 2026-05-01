import numpy as np
import pyswarms as ps

# -----------------------------
# SIMPLE PSO TEST
# -----------------------------

def sphere(x):
    return np.sum(x**2, axis=1)

if __name__ == "__main__":

    dim = 5

    bounds = (
        -5 * np.ones(dim),
         5 * np.ones(dim)
    )

    options = {
        "c1": 1.5,
        "c2": 1.5,
        "w": 0.7
    }

    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=dim,
        options=options,
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        sphere,
        iters=100
    )

    print("Best cost:", best_cost)
    print("Best position:", best_pos)