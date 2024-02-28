import numpy as np
from numba import njit, prange
import config as config


@njit
def get_normal_vec(vec: np.array) -> np.array:
    vec_rotated = np.array([vec[1], -vec[0]])
    vec_rotated /= np.linalg.norm(vec_rotated)
    return vec_rotated

@njit
def clip_array(array: np.ndarray, range: np.ndarray) -> np.ndarray:
    min_magnitude, max_magnitude = range
    norm = njit_norm_axis1(array)
    mask_max = norm > max_magnitude
    mask_min = norm < min_magnitude
    new_array = array.copy()
    if np.any(mask_max):
        new_array[mask_max] = (array[mask_max] / njit_norm_axis1(array[mask_max]).reshape(-1, 1)) * max_magnitude

    if np.any(mask_min):
        new_array[mask_min] = (array[mask_min] / njit_norm_axis1(array[mask_min]).reshape(-1, 1)) * min_magnitude

    return new_array


@njit
def clip_vector(vector: np.ndarray, range: np.ndarray) -> np.ndarray:
    min_magnitude, max_magnitude = range
    norm = njit_norm_vector(vector)
    mask_max = norm > max_magnitude
    mask_min = norm < min_magnitude
    new_vector = vector.copy()
    if mask_max:
        new_vector = (vector / norm) * max_magnitude

    if mask_min:
        new_vector = (vector / norm) * min_magnitude
    return new_vector


def paint_arrows(arrows, boids, dt):
    arrows.set_data(arrows=directions(boids, dt))  # отрисовка стрелок


def init_boids(boids: np.ndarray, screen_size: tuple, velocity_range: tuple = (0., 1.)):
    """
    Функция, отвечающая за создание птиц
    """
    n = boids.shape[0]
    rng = np.random.default_rng()

    # задаем рандомные начальные положения
    boids[:, 0] = rng.uniform(0., screen_size[0], size=n)  # координата x
    boids[:, 1] = rng.uniform(0., screen_size[1], size=n)  # координата y

    # задаем рандомные начальные скорости
    alpha = rng.uniform(0., 2 * np.pi, size=n)  # угол наклона вектора
    v = rng.uniform(*velocity_range, size=n)  # рандомный модуль скорости
    boids[:, 2] = v * np.cos(alpha)  # координата x
    boids[:, 3] = v * np.sin(alpha)  # координата y


# @todo сравнить
# def directions(boids: np.ndarray, dt: float):
#     """
#     Функция для отрисовки векторов стрелок, т.е. для отрисовки в каком направлении движутся птицы
#
#     Returns
#     -------
#     Массив вида N x (x0, y0, x, y) для отрисовки векторов стрелок
#     """
#     pos = boids[:, :2]  # положение в момент времени t
#     delta_pos = dt * boids[:, 2:4]  # как изменилось положение за dt
#     pos0 = pos - delta_pos  # положение в момент времени t-dt
#     return np.hstack((pos0, pos))

def directions(boids: np.ndarray, dt=float) -> np.ndarray:
    """

    :param boids:
    :param dt:
    :return: array N * (x0, y0, x1, y1) for Arrow
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


@njit
def njit_norm_axis1(vector: np.ndarray):
    norm = np.zeros(vector.shape[0], dtype=np.float64)
    for j in prange(vector.shape[0]):
        norm[j] = np.sqrt(vector[j, 0] * vector[j, 0] + vector[j, 1] * vector[j, 1])
    return norm


@njit
def njit_norm_vector(vector: np.ndarray):
    norm = 0
    for j in prange(vector.shape[0]):
        norm += vector[j] * vector[j]
    return np.sqrt(norm)


@njit
def distance(boids: np.ndarray, i: int):
    difference = boids[i, 0:2] - boids[:, 0:2]
    return njit_norm_axis1(difference)

# @todo сравнить c distance для вектора, а не матрицы
# @njit(parallel=True)
# def compute_distances(boids: np.ndarray) -> np.ndarray:
#     """
#     Вычисляет матрицу, где в ячейке (i, j) записано расстояние между птицей i и птицей j
#     """
#     # n, m = boids.shape
#     # vector_distance_difference = boids.reshape((n, 1, m)) - boids.reshape((1, n, m))
#     # norm_of_distance_difference = np.linalg.norm(vector_distance_difference, axis=2)
#     # return norm_of_distance_difference
#
#     r = boids[:, :2]
#     n = r.shape[0]
#     dist = np.empty(shape=(n, n), dtype=np.float64)
#     for i in prange(n):
#         dr_arr = boids[i, 0:2] - boids[:, 0:2]
#         dist[i] = njit_norm_axis1(dr_arr)
#         # dist[i] = np.sqrt(dr_arr[:, 0] ** 2 + dr_arr[:, 1] ** 2)
#
#         # dist[i] = np.sqrt((boids[i, 0:2] - boids[:, 0:2])[:, 0] ** 2 + (boids[i, 0:2] - boids[:, 0:2])[:, 1] ** 2)
#         # dist[i] = np.sqrt(dr_arr[:, 0] ** 2 + dr_arr[:, 1] ** 2)
#         # dr_arr = boids[i, 0:2] - boids[:, 0:2]
#         # norm = np.empty(n, dtype=np.float64)
#         # for j in prange(n):
#         #     norm[j] = np.sqrt(dr_arr[j, 0] ** 2 + dr_arr[j, 1] ** 2)
#         # dist[i] = norm
#     return dist


# @todo сравнить + мб добавить numba?
def vclip(v: np.ndarray, velocity_range: tuple[float, float]):
    """
    Если скорость выходит за разрешенные скорости, то мы обрезаем скорость
    """
    norm = np.linalg.norm(v, axis=1)  # модуль скорости
    mask = norm > velocity_range[1]  # маска
    if np.any(mask):
        v[mask] *= velocity_range[1] / norm[mask, None]


# @njit
# def clip_array(array: np.ndarray, range: np.ndarray) -> np.ndarray:
#     min_magnitude, max_magnitude = range
#     norm = njit_norm_axis1(array)
#     mask_max = norm > max_magnitude
#     mask_min = norm < min_magnitude
#     new_array = array.copy()
#     if np.any(mask_max):
#         new_array[mask_max] = (array[mask_max] / njit_norm_axis1(array[mask_max]).reshape(-1, 1)) * max_magnitude
#
#     if np.any(mask_min):
#         new_array[mask_min] = (array[mask_min] / njit_norm_axis1(array[mask_min]).reshape(-1, 1)) * min_magnitude
#
#     return new_array


# @njit
# def clip_vector(vector: np.ndarray, range: np.ndarray) -> np.ndarray:
#     min_magnitude, max_magnitude = range
#     norm = njit_norm_vector(vector)
#     mask_max = norm > max_magnitude
#     mask_min = norm < min_magnitude
#     new_vector = vector.copy()
#     if mask_max:
#         new_vector = (vector / norm) * max_magnitude
#
#     if mask_min:
#         new_vector = (vector / norm) * min_magnitude
#     return new_vector


@njit
def cohesion(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    directions = boids[distance_mask][:, :2] - boids[i, :2]
    acceleration = np.sum(directions, axis=0)
    acceleration /= directions.shape[0]
    vec = clip_vector(acceleration - boids[i, 2:4], np.array([0, config.max_delta_velocity_magnitude]))
    return vec

@njit
def compute_cohesion(boids: np.ndarray, id: int, mask: np.array) -> np.array:
    """
    Steer to move towards the average position (center of mass) of local flockmates
    """
    steering_pos = np.mean(boids[mask][:, 0:2])
    delta_steering_pos = steering_pos - boids[id][0:2]
    delta_steering_pos = delta_steering_pos / np.linalg.norm(delta_steering_pos)
    delta_steering_pos *= config.max_speed_magnitude
    delta_steering_v = delta_steering_pos - boids[id, 2:4]
    if np.linalg.norm(delta_steering_v) > config.max_delta_velocity_magnitude:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= config.max_delta_velocity_magnitude
    return delta_steering_v


@njit
def separation(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    distance_mask[i] = False
    directions = boids[i, :2] - boids[distance_mask][:, :2]
    directions *= (1 / (njit_norm_axis1(directions) + 0.0001))
    acceleration = np.sum(directions, axis=0)
    vec = clip_vector(acceleration - boids[i, 2:4], np.array([0, config.max_delta_velocity_magnitude]))
    return vec

@njit
def compute_separation(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer to avoid crowding local flockmates
    """
    dr = boids[id][0:2] - boids[mask][:, 0:2]
    dr *= 1 / (dr[:, 0] ** 2 + dr[:, 1] ** 2)
    steering_pos = np.sum(dr, axis=0)
    steering_pos /= np.linalg.norm(steering_pos)
    steering_pos *= config.max_speed_magnitude
    delta_steering_v = steering_pos - boids[id, 2:4]
    if np.linalg.norm(delta_steering_v) > config.max_delta_velocity_magnitude:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= config.max_delta_velocity_magnitude
    return delta_steering_v


@njit
def compute_alignment(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer towards the average heading of local flockmates
    """
    steering_v = np.sum(boids[mask][:, 2:4], axis=0)
    steering_v /= np.linalg.norm(steering_v)
    steering_v *= config.max_speed_magnitude
    delta_steering_v = steering_v - boids[id][2:4]
    if np.linalg.norm(delta_steering_v) > config.max_delta_velocity_magnitude:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= config.max_delta_velocity_magnitude
    return delta_steering_v

@njit
def alignment(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    velocity = boids[distance_mask][:, 2:4]
    acceleration = np.sum(velocity, axis=0)
    acceleration /= velocity.shape[0]
    return acceleration - boids[i, 2:4]


@njit
def compute_walls_interations(boids: np.ndarray, i: int, aspect_ratio: float):
    # mask_walls = np.empty(4)
    # mask_walls[0] = boids[i, 1] > 1
    # mask_walls[1] = boids[i, 0] > aspect_ratio
    # mask_walls[2] = boids[i, 1] < 0
    # mask_walls[3] = boids[i, 0] < 0
    #
    # if mask_walls[0]:
    #     boids[i, 1] = 0
    # if mask_walls[1]:
    #     boids[i, 0] = 0
    # if mask_walls[2]:
    #     boids[i, 1] = 1
    # if mask_walls[3]:
    #     boids[i, 0] = aspect_ratio
    mask_walls = np.empty(4)
    mask_walls[0] = boids[i, 1] > 1
    mask_walls[1] = boids[i, 0] > aspect_ratio
    mask_walls[2] = boids[i, 1] < 0
    mask_walls[3] = boids[i, 0] < 0

    if mask_walls[0]:
        boids[i, 3] = -boids[i, 3]
        boids[i][1] = 1 - 0.001

    if mask_walls[1]:
        boids[i, 2] = -boids[i, 2]
        boids[i, 0] = aspect_ratio - 0.001

    if mask_walls[2]:
        boids[i, 3] = -boids[i, 3]
        boids[i, 1] = 0.001

    if mask_walls[3]:
        boids[i, 2] = -boids[i, 2]
        boids[i, 0] = 0.001

# @njit(parallel=True)
# def compute_walls_interations(boids: np.ndarray, mask: np.ndarray, screen_size: np.array):
#     for i in prange(boids.shape[0]):
#         if mask[0][i]:
#             boids[i][3] = -boids[i][3]
#             # boids[i][3] *= 200000
#             boids[i][1] = screen_size[1] - 0.001
#
#         if mask[1][i]:
#             boids[i][2] = -boids[i][2]
#             # boids[i][2] *= 200000
#             boids[i][0] = screen_size[0] - 0.001
#
#         if mask[2][i]:
#             boids[i][3] = -boids[i][3]
#             # boids[i][3] *= 200000
#             boids[i][1] = 0.001
#
#         if mask[3][i]:
#             boids[i][2] = -boids[i][2]
#             # boids[i][2] *= 200000
#             boids[i][0] = 0.001
#
#     for mask_wall in mask:
#         for i in prange(boids.shape[0]):
#             if mask_wall[i]:
#                 boids[i, 4:6] = np.zeros(2)


@njit(parallel=True)
def flocking(boids: np.ndarray, perseption: float, coeffitients: np.ndarray, aspect_ratio: float,
             a_range: np.ndarray):
    """
    Функция, отвечающая за взаимодействие птиц между собой
    """

    a_separation = np.zeros(2)
    a_cohesion = np.zeros(2)
    a_alignment = np.zeros(2)
    for i in prange(boids.shape[0]):
        d = distance(boids, i)
        perception_mask = d < perseption
        separation_mask = d < perseption / 2
        separation_mask[i] = False
        cohesion_mask = np.logical_xor(perception_mask, separation_mask)

        compute_walls_interations(boids, i, aspect_ratio)

        if np.any(perception_mask):
            if np.any(separation_mask):
                # a_separation = separation(boids, i, separation_mask)
                a_separation = compute_separation(boids, i, separation_mask)
            if np.any(cohesion_mask):
                # a_cohesion = cohesion(boids, i, cohesion_mask)
                a_cohesion = compute_cohesion(boids, i, cohesion_mask)
            # a_alignment = alignment(boids, i, perception_mask)
            a_alignment = compute_alignment(boids, i, perception_mask)

        # if np.any(perception_mask):
        #     if np.any(separation_mask):
        #         a_separation = separation(boids, i, separation_mask)
        #     # if np.any(cohesion_mask):
        #     #     a_cohesion = cohesion(boids, i, cohesion_mask)
        #     # a_alignment = compute_alignment(boids, i, perception_mask)

        acceleration = coeffitients[0] * a_cohesion \
                       + coeffitients[1] * a_separation \
                       + coeffitients[2] * a_alignment
        boids[i, 4:6] = acceleration


def propagate(boids: np.ndarray, dt: float, velocity_range: tuple[float, float]):
    """
    Пересчет скоростей за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    vclip(boids[:, 2:4], velocity_range)  # обрезаем скорости, если они вышли за velocity_range
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt

# def propagate(boids, delta_time, v_range):
#     boids[:, 2:4] += boids[:, 4:6] * delta_time
#     boids[:, 2:4] = clip_array(boids[:, 2:4], v_range)
#     boids[:, 0:2] += boids[:, 2:4] * delta_time
