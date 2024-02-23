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
    boids[:, 2:4] += boids[:, 4:6]  # меняем скорости: v += dv, где dv — изменение скорости за dt
    vclip(boids[:, 2:4], vrange)  # обрезаем скорости, если они вышли за vrange
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt


def compute_distances(vecs: np.ndarray) -> np.ndarray:
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
    steering_pos = np.mean(boids[mask], axis=0)[0:2]
    # steering_pos = boids[id][0:2] + np.sum(boids[mask], axis=0)[0:2]
    # steering_pos = steering_pos / np.linalg.norm(steering_pos)
    delta_steering_pos = steering_pos - boids[id][0:2]
    steering_v = delta_steering_pos / np.linalg.norm(delta_steering_pos)
    return steering_v * dt
    # intention_pos = (boids[id][0:2] + np.sum(boids[mask], axis=0)[0:2]) / (1 + boids[mask].shape[0])  # радиус-вектор точки, куда мы хотим чтобы переместилась птица
    # intention_delta_pos = intention_pos - boids[id][0:2]
    # normal = get_normal_vec(boids[id][2:4])  # нормаль к вектору скорости
    # normal_acceleration = normal if np.dot(intention_delta_pos, normal) > 0 else -normal  # нормальное усорение
    # delta_v = normal_acceleration * dt
    # return delta_v


def compute_separation(boids, id, mask, dt, radius):
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
    steering_pos = np.sum(
        (boids[id][0:2] - boids[mask][:, 0:2])
        / (1 + np.linalg.norm(boids[id][0:2] - boids[mask][:, 0:2])**2),
        axis=0)
    steering_v = steering_pos / np.linalg.norm(steering_pos)
    return steering_v * dt
    # # intention_delta_pos = np.sum(
    # #     (boids[id][0:2] - boids[mask][:, 0:2])
    # #     * (1 / (1 + np.linalg.norm(boids[mask][0:2]))),
    # #     axis=0)
    #
    # temp = boids[mask][:, 0:2] - boids[id][0:2]
    # D = np.linalg.norm(boids[mask][:, 0:2] - boids[id][0:2], axis=1)
    # # k_arr = D / radius
    # k_arr = 1 - D / radius
    # intention_delta_pos = -np.sum(
    #     (boids[mask][:, 0:2] - boids[id][0:2])
    #     * (1 / k_arr[:][0]),
    #     axis=0)
    #
    # # intention_delta_pos = -np.sum(
    # #     (boids[mask][:, 0:2] - boids[id][0:2])
    # #     * (1 / np.linalg.norm(boids[mask][:, 0:2] - boids[id][0:2])),
    # #     axis=0)
    # # intention_delta_pos = -np.sum(boids[mask][:, 0:2] - boids[id][0:2], axis=0)
    #
    # normal = get_normal_vec(boids[id][2:4])  # нормаль к вектору скорости
    # normal_acceleration = normal if np.dot(intention_delta_pos, normal) > 0 else -normal  # нормальное усорение
    # delta_v = normal_acceleration * dt
    # return delta_v / 100


# @todo сделать слайдер

def compute_alignment(boids, id, mask, dt):
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

    steering_v = boids[mask].mean(axis=0)[2:4]
    steering_v = steering_v / np.linalg.norm(steering_v)
    return steering_v * dt
    # normal = get_normal_vec(boids[id, 2:4])
    # normal_acceleration = normal if np.dot(avarage_v, normal) > 0 else -normal  # нормальное усорение
    # delta_v = normal_acceleration * dt
    # return delta_v


def compute_walls_interations(boids, mask, field_size):
    for i in range(boids.shape[0]):
        if mask[0][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = field_size[1] - 0.001

        if mask[1][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = field_size[0] - 0.001

        if mask[2][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = 0.001

        if mask[3][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = 0.001

    # c = 1
    # x = boids[:, 0]
    # y = boids[:, 1]
    #
    # a_left = 1 / (np.abs(x) + c) ** 2
    # a_right = -1 / (np.abs(x - asp) + c) ** 2
    #
    # a_bottom = 1 / (np.abs(y) + c) ** 2
    # a_top = -1 / (np.abs(y - 1.) + c) ** 2
    #
    # return np.column_stack((a_left + a_right, a_bottom + a_top))


def flocking(boids: np.ndarray,
             radius: float,
             coeff: np.array,
             field_size: tuple,
             vrange: tuple,
             dt: float):
    """
    Функция, отвечающая за взаимодействие птиц между собой

    Parameters
    ----------
    boids
    radius
    coeff
    field_size
    vrange
    """
    distances = compute_distances(boids[:, 0:2])  # матрица с расстояниями между всеми птицами
    N = boids.shape[0]
    distances[range(N), range(N)] = np.inf  # выкидываем расстояния между i и i
    k = 2  # насколько маленкий радиус отличается от большого
    # mask_cohesion = (distances > radius / k) * (distances < radius)
    mask_cohesion = distances < radius / 1
    mask_separation = distances < radius / 2
    mask_alignment = distances < radius / 2
    # mask_alignment = distances < radius / (2 * k)
    mask_walls = np.array([
        boids[:, 1] > field_size[1],
        boids[:, 0] > field_size[0],
        boids[:, 1] < 0,
        boids[:, 0] < 0,
    ])
    # walls = create_walls(boids, field_size[0])  # создание стен
    compute_walls_interations(boids, mask_walls, field_size)  # if np.any(mask_walls, axis=0) else np.zeros(2)
    for i in range(N):
        # вычисляем насколько должны поменяться скорости
        separation = compute_separation(boids, i, mask_separation[i], dt, radius) if np.any(mask_separation[i]) else np.zeros(2)
        alignment = compute_alignment(boids, i, mask_alignment[i], dt) if np.any(mask_alignment[i]) else np.zeros(2)
        cohesion = compute_cohesion(boids, i, mask_cohesion[i], dt) if np.any(mask_cohesion[i]) else np.zeros(2)

        # separation = 0
        # alignment = 0
        # cohesion = 0

        # меняем изменения скорости птиц
        boids[i, 4:6] = coeff[0] * cohesion + coeff[1] * alignment + coeff[2] * separation
    for mask_wall in mask_walls:
        for i in range(N):
            if mask_wall[i]:
                boids[i, 4:6] = [0.0, 0.0]
