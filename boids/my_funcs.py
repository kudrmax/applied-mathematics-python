from typing import Tuple
from vispy import app, scene
from numba import njit, prange

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

# global maxSpeed
# global maxDeltaVelocity
maxSpeed = 4
maxDeltaVelocity = 10

@njit
def njit_norm(arr: np.array):
    return np.sqrt(np.sum(arr**2))


def njit_mean_ax_0(arr: np.array):
    return np.sum(arr, axis=0) / arr.shape[0]


@njit
def get_normal_vec(vec: np.array):
    new_vec = np.empty(2)
    new_vec[0] = vec[1].copy()
    new_vec[1] = -vec[0].copy()
    return new_vec / np.linalg.norm(new_vec)


def get_normal_acceleration(vec: np.array):
    normal = get_normal_vec(vec)
    if np.dot(vec, normal) < 0:
        return -normal
    return normal


def paint_arrows(arrows, boids, dt):
    arrows.set_data(arrows=directions(boids, dt))  # отрисовка стрелок


def init_boids(boids: np.ndarray, field_size: tuple, vrange: tuple = (0., 1.)):
    """
    Функция, отвечающая за создание птиц
    """
    n = boids.shape[0]
    rng = np.random.default_rng()

    # задаем рандомные начальные положения
    boids[:, 0] = rng.uniform(0., field_size[0], size=n)  # координата x
    boids[:, 1] = rng.uniform(0., field_size[1], size=n)  # координата y

    # задаем рандомные начальные скорости
    alpha = rng.uniform(0., 2 * np.pi, size=n)  # угол наклона вектора
    v = rng.uniform(*vrange, size=n)  # рандомный модуль скорости
    boids[:, 2] = v * np.cos(alpha)  # координата x
    boids[:, 3] = v * np.sin(alpha)  # координата y


def directions(boids: np.ndarray, dt: float):
    """
    Функция для отрисовки векторов стрелок, т.е. для отрисовки в каком направлении движутся птицы

    Returns
    -------
    Массив вида N x (x0, y0, x, y) для отрисовки векторов стрелок
    """
    pos = boids[:, :2]  # положение в момент времени t
    delta_pos = dt * boids[:, 2:4]  # как изменилось положение за dt
    pos0 = pos - delta_pos  # положение в момент времени t-dt
    return np.hstack((pos0, pos))


def vclip(v: np.ndarray, vrange: tuple[float, float]):
    """
    Если скорость выходит за разрешенные скорости, то мы обрезаем скорость
    """
    norm = np.linalg.norm(v, axis=1)  # модуль скорости
    mask = norm > vrange[1]  # маска
    # @todo разобраться что делает эта штука (кажется обрезает вектор просто до значения модуля скорости vrange[1])
    if np.any(mask):
        # np.newaxis # np.reshape # norm[mask].reshape(-1, -1) -1 — автоматически посчитать
        v[mask] *= vrange[1] / norm[mask, None]


def propagate(boids: np.ndarray, dt: float, vrange: tuple[float, float]):
    """
    Пересчет скоростей за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    vclip(boids[:, 2:4], vrange)  # обрезаем скорости, если они вышли за vrange
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt


@njit(parallel=True)
def compute_distances(boids: np.ndarray) -> np.ndarray:
    """
    Вычисляет матрицу, где в ячейке (i, j) записано расстояние между птицей i и птицей j
    """
    # n, m = boids.shape
    # vector_distance_difference = boids.reshape((n, 1, m)) - boids.reshape((1, n, m))
    # norm_of_distance_difference = np.linalg.norm(vector_distance_difference, axis=2)
    # return norm_of_distance_difference

    p = boids[:, :2]
    n = p.shape[0]
    dist = np.empty(shape=(n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            v = p[i] - p[j]
            d = np.sqrt(np.sum(v * v))
            dist[i, j] = d
    dist = np.sqrt(dist)
    return dist


@njit
def compute_cohesion(boids: np.ndarray, id: int, mask: np.array) -> np.array:
    """
    Steer to move towards the average position (center of mass) of local flockmates
    """
    steering_pos = np.sum(boids[mask][:, 0:2], axis=0) / boids[mask].shape[0]
    delta_steering_pos = steering_pos - boids[id][0:2]
    delta_steering_pos = delta_steering_pos / np.linalg.norm(delta_steering_pos)
    delta_steering_pos *= maxSpeed
    delta_steering_v = delta_steering_pos - boids[id, 2:4]
    if np.linalg.norm(delta_steering_v) > maxDeltaVelocity:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= maxDeltaVelocity
    return delta_steering_v
    # steering_pos = np.mean(boids[mask], axis=0)[0:2]
    # steering_pos = steering_pos / np.linalg.norm(steering_pos)
    # delta_steering_pos = steering_pos - boids[id][0:2]
    # steering_v = delta_steering_pos / np.linalg.norm(delta_steering_pos)
    # return steering_v


@njit
def compute_separation(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer to avoid crowding local flockmates
    """
    steering_pos = np.sum((boids[id][0:2] - boids[mask][:, 0:2]) / np.linalg.norm(boids[id][0:2] - boids[mask][:, 0:2])**2, axis=0)
    steering_pos = steering_pos / np.linalg.norm(steering_pos)
    steering_pos *= maxSpeed
    delta_steering_v = steering_pos - boids[id, 2:4]
    if np.linalg.norm(delta_steering_v) > maxDeltaVelocity:
        delta_steering_v = delta_steering_v /np.linalg.norm(delta_steering_v)
        delta_steering_v *= maxDeltaVelocity
    return delta_steering_v


@njit
def compute_alignment(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer towards the average heading of local flockmates
    """
    # обычное решение
    steering_v = np.sum(boids[mask][:, 2:4], axis=0) / boids[mask].shape[0]
    # steering_v = njit_mean_ax_0(boids[mask][:, 2:4])
    # steering_v = boids[mask].mean(axis=0)[2:4]
    steering_v = steering_v / np.linalg.norm(steering_v)
    steering_v *= maxSpeed
    delta_steering_v = steering_v - boids[id][2:4]
    if np.linalg.norm(delta_steering_v) > maxDeltaVelocity:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= maxDeltaVelocity
    return delta_steering_v

    # расширение
    # steering_v = boids[mask].mean(axis=0)[2:4]
    # avegare_pos = boids[mask].mean(axis=0)[0:2]
    # delta_pos = boids[id][0:2] - avegare_pos
    # delta_pos = 0
    # steering_v += delta_pos * 2
    # steering_v = steering_v / np.linalg.norm(steering_v)
    # return steering_v - boids[id][2:4]


@njit(parallel=True)
def compute_walls_interations(boids, mask, field_size):
    for i in prange(boids.shape[0]):
        if mask[0][i]:
            boids[i][3] = -boids[i][3]
            # boids[i][3] *= 200000
            boids[i][1] = field_size[1] - 0.001

        if mask[1][i]:
            boids[i][2] = -boids[i][2]
            # boids[i][2] *= 200000
            boids[i][0] = field_size[0] - 0.001

        if mask[2][i]:
            boids[i][3] = -boids[i][3]
            # boids[i][3] *= 200000
            boids[i][1] = 0.001

        if mask[3][i]:
            boids[i][2] = -boids[i][2]
            # boids[i][2] *= 200000
            boids[i][0] = 0.001


@njit(parallel=True)
def flocking(boids: np.ndarray,
             radius: float,
             coeff: np.array,
             field_size,
             vrange: np.array,
             dt: float):
    """
    Функция, отвечающая за взаимодействие птиц между собой
    """
    distances = compute_distances(boids)  # матрица с расстояниями между всеми птицами
    N = boids.shape[0]
    for i in range(N):
        distances[i, i] = np.inf  # выкидываем расстояния между i и i

    mask_cohesion = distances < (radius * 2) * (distances > radius / 2)
    mask_separation = distances < radius / 2
    mask_alignment = distances < radius

    # mask_walls = np.empty((4, boids.shape[0]))
    # mask_walls[0] = boids[:, 1] > field_size[1]
    # mask_walls[1] = boids[:, 0] > field_size[0]
    # mask_walls[2] = boids[:, 1] < 0
    # mask_walls[3] = boids[:, 0] < 0
    # compute_walls_interations(boids, mask_walls, field_size)  # if np.any(mask_walls, axis=0) else np.zeros(2)

    for i in prange(N):
        if not np.any(mask_cohesion[i]):
            cohesion = np.zeros(2)
        else:
            cohesion = compute_cohesion(boids, i, mask_cohesion[i])
        if not np.any(mask_separation[i]):
            separation = np.zeros(2)
        else:
            separation = compute_separation(boids, i, mask_separation[i])
        if not np.any(mask_alignment[i]):
            alignment = np.zeros(2)
        else:
            alignment = compute_alignment(boids, i, mask_alignment[i])

        # separation = compute_separation(boids, i, mask_separation[i], dt, radius) if np.any(mask_separation[i]) else np.zeros(2)
        # alignment = compute_alignment(boids, i, mask_alignment[i], dt) if np.any(mask_alignment[i]) else np.zeros(2)
        # cohesion = compute_cohesion(boids, i, mask_cohesion[i], dt) if np.any(mask_cohesion[i]) else np.zeros(2)

        a = coeff[0] * cohesion + coeff[1] * alignment + coeff[2] * separation
        noise = 0
        # noise = get_normal_vec(boids[i, 2:4]) * np.random.uniform(-0.1, 0.1)
        boids[i, 4:6] = 100 * a + noise

    # for mask_wall in mask_walls:
    #     for i in range(N):
    #         if mask_wall[i]:
    #             boids[i, 4:6] = [0.0, 0.0]
