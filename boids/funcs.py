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


def vclip(v: np.ndarray, vrange: tuple[float, float]):
    norm = np.linalg.norm(v, axis=1)
    mask = norm > vrange[1]
    if np.any(mask):
        # np.newaxis # np.reshape # norm[mask].reshape(-1, -1) -1 — автоматически посчитать
        v[mask] *= vrange[1] / norm[mask, None]


def propagate(boids: np.ndarray, dt: float, vrange: tuple[float, float]):
    boids[:, 2:4] += 0.5 * dt * boids[:, 4:6]  # скорости # v = v*dt # возможно тут нужно убрать умножение на 0.5
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]  # координаты # s = s_0 + v_0*dt + 0.5*a*dtˆ2


def distances(vecs: np.ndarray) -> np.ndarray:
    n, m = vecs.shape
    delta = vecs.reshape((n, 1, m)) - vecs.reshape((1, n, m))
    D = np.linalg.norm(delta, axis=2)
    return D


def flocking(boids: np.ndarray,
             perception: float,
             coeff: np.array,
             asp: float,
             vrange: tuple):
    D = distances(boids[:, 0:2])
    N = boids.shape[0]
    D[range(N), range(N)] = perception + 1  # выкидываем из D растояние между i и i
    mask = D < perception
    wal = walls(boids, asp)
    for i in range(N):
        if not np.any(mask[i]):  # нет соседей
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            coh = alignment(boids, i, mask[i], vrange)
            coh = separation(boids, i, mask[i], perception)
        boids[i, 4:6] = coeff[0] * coh + coeff[1] * alg + coeff[2] * sep + coeff[3] * wal[i]
