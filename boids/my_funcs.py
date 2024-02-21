from typing import Tuple
from vispy import app, scene

import numpy as np


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

    Parameters
    ----------
    boids
    ratio
    vrange
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
    boids[:, 2:4] += boids[:, 4:6]
    # boids[:, 2:4] += 0.05 * dt * boids[:, 4:6]  # скорости # v = v*dt
    vclip(boids[:, 2:4], vrange)  # обрезаем скорости, если они вышли за vrange
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


def compute_cohesion(boids: np.ndarray, id: int, mask: np.array, dt: float) -> np.array:
    """
    Steer to move towards the average position (center of mass) of local flockmates

    Parameters
    ----------
    dt
    boids: массив птиц
    id: индекс птицы в массиве boids
    mask: если mask[j] == True, то значит птица j взаиможействует с данной птицей id
    perception

    Returns
    -------
    Вектор новых ускорений
    """
    new_pos = (boids[id][0:2] + np.sum(boids[mask], axis=0)[0:2]) / (1 + boids[mask].shape[0]) # радиус-вектор точки, куда мы хотим чтобы переместилась птица
    delta_pos = new_pos - boids[id][0:2]
    normal = get_normal_vec(boids[id][2:4]) # нормаль к вектору скорости
    normal_acceleration = normal if np.dot(delta_pos, normal) > 0 else -normal # нормальное усорение
    delta_v = normal_acceleration * dt
    return delta_v


def compute_separation(boids, id, mask, perception):
    """
    steer to avoid crowding local flockmates

    Parameters
    ----------
    boids
    i
    mask
    perception

    Returns
    -------

    """
    n = boids[mask].shape[0]
    new_pos = n * boids[id][0:2] - np.sum(boids[mask], axis=0)[0:2]  # == ((r - r1) + (r - r2) +...+ (r - rn)
    return new_pos / ((new_pos[0] ** 2 + new_pos[1] ** 2) + 1)


# @todo сделать слайдер

def compute_alignment(boids, id, mask, vrange):
    """
    steer towards the average heading of local flockmates

    Parameters
    ----------
    boids
    i
    mask
    vrange

    Returns
    -------

    # """
    # avarage_velocity = boids[mask, 2:4].mean(axis=0)
    # old_velocity = boids[id, 2:4]
    # return (avarage_velocity - old_velocity)

    avarage_v = boids[mask].mean(axis=0)
    this_v = boids[id,].copy()
    n_this_v = this_v.copy()
    n_this_v[2:3] = this_v[3:4]
    n_this_v[3:4] = -this_v[2:3]
    if np.dot(avarage_v[2:4], n_this_v[2:4]) < 0:
        n_this_v = -n_this_v
    return n_this_v[2:4]


def flocking(boids: np.ndarray,
             perception: float,
             coeff: np.array,
             field_size: tuple,
             vrange: tuple,
             dt: float):
    """
    Функция, отвечающая за взаимодействие птиц между собой

    Parameters
    ----------
    boids
    perception
    coeff
    field_size
    vrange
    """
    D = distances(boids[:, 0:2])  # матрица с расстояниями между всеми птицами
    N = boids.shape[0]
    D[range(N), range(N)] = np.inf  # выкидываем расстояния между i и i
    mask = D < perception  # если расстояние достаточно близкое (в круге радиуса perception), то True
    # walls = create_walls(boids, ratio)  # создание стен
    for i in range(N):

        # вычисляем насколько должны поменяться ускорения
        if not np.any(mask[i]):  # если нет соседей, то ничего не менять, то есть птица как двигалась, так и движется
            separation = np.zeros(2)
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
        else:
            # separation = compute_separation(boids, i, mask[i], dt)
            # alignment = compute_alignment(boids, i, mask[i], dt)
            cohesion = compute_cohesion(boids, i, mask[i], dt)

        # @todo временно:
        separation = np.zeros(2)
        alignment = np.zeros(2)
        # cohesion = np.zeros(2)

        # меняем ускорения птиц
        boids[i, 4:6] = coeff[0] * cohesion
        # boids[i, 4:6] =
        # boids[i, 4:6] = coeff[0] * cohesion + coeff[1] * alignment + coeff[2] * separation  # + coeff[3] * walls[i]
