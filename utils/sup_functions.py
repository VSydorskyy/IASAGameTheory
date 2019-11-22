import numpy as np

def random_walk(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return unif
    else:
        return p

def boundary_visiting(m):
    return np.array([random_walk(el) for el in np.random.uniform(size=m)])
