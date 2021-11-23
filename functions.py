import numpy as np
import math
from numpy.random import default_rng
rng = default_rng()

def random_vector(n):
    """Generate an uniformly distributed random vector."""
    components = [rng.standard_normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = np.array([x/r for x in components])
    return v

def random_vectors(m, n):
    array = np.random.normal(size=(m,n))
    norm = np.linalg.norm(array, axis=1)
    return array/norm[:, None]

def random_joint_vectors(n):
    """Generate a list of random 3D unit vectors."""
    a, b = [], []
    for _ in range(n):
        vector_a = random_vector(3)
        vector_b = random_vector(3)
        a.append(vector_a)
        b.append(vector_b)
    return (a, b)
