from typing import Tuple
from vispy import app, scene

import numpy as np


def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0., 1.)):
    """
    Функция, отвечающая за создание птиц

    Parameters
    ----------
    boids
    asp
    vrange
    """
    n = boids.shape[0]
    rng = np.random.default_rng()

    # задаем рандомные начальные положения
    boids[:, 0] = rng.uniform(0., asp, size=n)  # координата x
    boids[:, 1] = rng.uniform(0., 1., size=n)  # координата y

    # задаем рандомные начальные скорости
    alpha = rng.uniform(0., 2 * np.pi, size=n)  # угол наклона вектора
    v = rng.uniform(*vrange, size=n)  # рандомный модуль скорости
    boids[:, 2] = v * np.cos(alpha)  # координата x
    boids[:, 3] = v * np.sin(alpha)  # координата y


def directions(boids: np.ndarray, dt: float):
    """
    Функция для отрисовки векторов стрелок, т.е. для отрисовки в каком направлении движутся птицы

    Parameters
    ----------
    boids
    dt

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

    Parameters
    ----------
    v
    vrange
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

    Parameters
    ----------
    boids
    dt
    vrange
    """
    # @todo разобраться
    boids[:, 2:4] += 0.5 * dt * boids[:, 4:6]  # скорости # v = v*dt # возможно тут нужно убрать умножение на 0.5
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]  # координаты # s = s_0 + v_0*dt + 0.5*a*dtˆ2


def distances(vecs: np.ndarray) -> np.ndarray:
    """
    Вычисляет матрицу, где в ячейке (i, j) записано расстояние между птицей i и птицей j

    Parameters
    ----------
    vecs

    Returns
    -------
    Матрица c расстояниями
    """
    n, m = vecs.shape
    vector_distance_difference = vecs.reshape((n, 1, m)) - vecs.reshape((1, n, m))
    norm_of_distance_difference = np.linalg.norm(vector_distance_difference, axis=2)
    return norm_of_distance_difference


def compute_cohesion(boids: np.ndarray,
             id: int,
             mask_row: np.ndarray,
             perception: float) -> np.ndarray:
    """
    Steer to move towards the average position (center of mass) of local flockmates

    @remove compute_separation(boids, i, mask[i], perception)

    Parameters
    ----------
    boids: массив птиц
    id: индекс птицы в массиве boids
    mask_row: если mask_row[j] == True, то значит птица j взаиможействует с данной птицей id
    perception

    Returns
    -------

    """
    center_pos = boids[id][0:2]
    k = 1
    for j in range(boids.shape[0]):
        if mask_row[j]:
            center_pos += boids[j][0:2]
            k += 1
    center_pos /= k
    return center_pos


def flocking(boids: np.ndarray,
             perception: float,
             coeff: np.array,
             asp: float,
             vrange: tuple):
    """
    Функция, отвечающая за взаимодействие птиц между собой

    Parameters
    ----------
    boids
    perception
    coeff
    asp
    vrange
    """
    D = distances(boids[:, 0:2])  # матрица с расстояниями между всеми птицами
    N = boids.shape[0]
    D[range(N), range(N)] = np.inf  # выкидываем расстояния между i и i
    mask = D < perception  # если расстояние достаточно близкое (в круге радиуса perception), то True
    # walls = create_walls(boids, asp)  # создание стен
    for i in range(N):

        # вычисляем насколько должны поменяться ускорения
        if not np.any(mask[i]):  # если нет соседей, то ничего не менять, то есть птица как двигалась, так и движется
            separation = np.zeros(2)
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
        else:
            # separation = compute_separation(boids, i, mask[i], perception)
            # alignment = compute_alignment(boids, i, mask[i], vrange)
            cohesion = compute_cohesion(boids, i, mask[i], perception)

        # @todo временно:
        separation = np.zeros(2)
        alignment = np.zeros(2)
        # cohesion = np.zeros(2)

        # меняем ускорения птиц
        boids[i, 4:6] = coeff[0] * cohesion + coeff[1] * alignment + coeff[2] * separation  # + coeff[3] * walls[i]
