from typing import Tuple
from vispy import app, scene

import numpy as np


def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0., 1.)):
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0., 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    boids[:, 2] = v * np.cos(alpha)
    boids[:, 3] = v * np.sin(alpha)


def directions(boids: np.ndarray, dt: float):
    """

    Parameters
    ----------
    boids
    dt

    Returns
    -------
    array of N x (x0, y0, x1, y1) for array painting
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))
