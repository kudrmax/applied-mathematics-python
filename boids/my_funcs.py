import numpy as np
from numba import njit, prange
import config as config


@njit
def get_normal_vec(vec: np.array) -> np.array:
    vec_rotated = np.array([vec[1], -vec[0]])
    vec_rotated /= np.linalg.norm(vec_rotated)
    return vec_rotated


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

    boids[0][2:4] = [*velocity_range]


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


# @njit
# def njit_norm_axis1(vector: np.ndarray):
#     norm = np.zeros(vector.shape[0], dtype=np.float64)
#     for j in prange(vector.shape[0]):
#         norm[j] = np.sqrt(vector[j, 0] * vector[j, 0] + vector[j, 1] * vector[j, 1])
#     return norm
#
#
# @njit
# def njit_norm_vector(vector: np.ndarray):
#     norm = 0
#     for j in range(vector.shape[0]):
#         norm += vector[j] * vector[j]
#     return np.sqrt(norm)


@njit
def compute_distance(boids: np.ndarray, i: int):
    # arr = np.full(boids.shape[0], fill_value=np.inf, dtype=float)
    # dr = boids[i, 0:2] - boids[mask_grid][:, 0:2]
    # arr[mask_grid] = np.sqrt(dr[:, 0] ** 2 + dr[:, 1] ** 2)
    # return arr
    dr = boids[i, 0:2] - boids[:, 0:2]
    return np.sqrt(dr[:, 0] ** 2 + dr[:, 1] ** 2)


def vclip(v: np.ndarray, velocity_range: np.array):
    """
    Если скорость выходит за разрешенные скорости, то мы обрезаем скорость
    """
    norm = np.linalg.norm(v, axis=1)  # модуль скорости
    mask = norm > velocity_range[1]  # маска
    if np.any(mask):
        v[mask] *= velocity_range[1] / norm[mask, None]


@njit
def compute_cohesion(boids: np.ndarray, id: int, mask: np.array) -> np.array:
    """
    Steer to move towards the average position (center of mass) of local flockmates
    """
    if boids[mask].shape[0] > 1:
        steering_pos = np.sum(boids[mask][:, 0:2], axis=0)
        steering_pos /= boids[mask].shape[0]
        delta_steering_pos = steering_pos - boids[id][0:2]
        delta_steering_pos /= np.linalg.norm(delta_steering_pos)
        delta_steering_pos *= config.max_speed_magnitude
        delta_steering_v = delta_steering_pos - boids[id, 2:4]
        if np.linalg.norm(delta_steering_v) > config.max_delta_velocity_magnitude:
            delta_steering_v /= np.linalg.norm(delta_steering_v)
            delta_steering_v *= config.max_delta_velocity_magnitude
        return delta_steering_v
    else:
        return np.zeros(2)


@njit
def compute_separation(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer to avoid crowding local flockmates
    """
    dr = boids[id][0:2] - boids[mask][:, 0:2]
    dr *= 1 / ((dr[:, 0] ** 2 + dr[:, 1] ** 2) + 0.001)
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


@njit(parallel=True)
def compute_walls_interations(boids: np.ndarray, screen_size: np.array):
    mask_walls = np.empty((4, boids.shape[0]))
    mask_walls[0] = boids[:, 1] > screen_size[1]
    mask_walls[1] = boids[:, 0] > screen_size[0]
    mask_walls[2] = boids[:, 1] < 0
    mask_walls[3] = boids[:, 0] < 0

    for i in prange(boids.shape[0]):
        if mask_walls[0][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = screen_size[1] - 0.001

        if mask_walls[1][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = screen_size[0] - 0.001

        if mask_walls[2][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = 0.001

        if mask_walls[3][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = 0.001

    for mask_wall in mask_walls:
        for i in prange(boids.shape[0]):
            if mask_wall[i]:
                boids[i, 4:6] = np.zeros(2)


@njit
def compute_mask_sector(boids: np.ndarray, mask: np.array, id: int, alpha: float):
    mask[id] = False
    alpha_radians = np.radians(alpha)

    this_v = boids[id, 2:4]
    dr = boids[mask][:, 0:2] - boids[id, 0:2]

    cos = np.empty(dr.shape[0])
    for j in range(dr.shape[0]):
        cos[j] = np.dot(this_v, dr[j]) / (np.linalg.norm(dr[j]) * np.linalg.norm(this_v))
    angle = np.arccos(cos)

    new_mask = np.full(boids.shape[0], False)
    new_mask[mask] = angle < alpha_radians
    new_mask[id] = True

    mask[id] = True
    return new_mask


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception_radius: float,
             coeff: np.array,
             screen_size: np.array, indexes_in_grid, grid, grid_size):
    """
    Функция, отвечающая за взаимодействие птиц между собой
    """
    neighbours = np.full(boids.shape[0], False)
    for i in prange(boids.shape[0]):

        row = indexes_in_grid[i][0]
        col = indexes_in_grid[i][1]
        mask_grid = grid[row, col][:grid_size[row, col]]
        this_index = 0

        # print('i = ', i)
        for j in range(len(mask_grid)):
            if i == mask_grid[j]:
                this_index = j
        # print('this_index = ', this_index)
        # print('boids[i] = ', boids[i])
        # print('boids[mask_grid][this_index] = ', boids[mask_grid][this_index])

        boids_neigh = boids[mask_grid]
        # print('boids_neigh[this_index] = ', boids_neigh[this_index])
        # print('boids[mask_grid]', boids[mask_grid])

        # print('boids_neigh before before', boids_neigh)
        # for j in range(boids_neigh.shape[0]):
        #     if boids_neigh[j] == this_boid:
        #         this_index = j
        # print('this_index', this_index)
        # print('boids_neigh before', boids_neigh)

        D = compute_distance(boids_neigh, this_index)
        # print('D = ', D)

        # D = compute_distance(boids, i)

        # print('boids_neigh after', boids_neigh)

        mask_in_perseption_radius = D < perception_radius
        # print('mask_in_perseption_radius = ', mask_in_perseption_radius)

        # mask_alignment = mask_in_perseption_radius
        # mask_separation = D < perception_radius / 2
        # mask_cohesion = np.logical_xor(mask_separation, mask_alignment)

        mask_sector = mask_in_perseption_radius
        alpha = 30.0
        # mask_sector = compute_mask_sector(boids_neigh, mask_in_perseption_radius, this_index, alpha)
        mask_alignment = np.logical_and(mask_in_perseption_radius, mask_sector)
        mask_separation = np.logical_and(D < perception_radius / 2.0, mask_sector)
        mask_cohesion = np.logical_xor(mask_separation, mask_alignment)

        mask_separation[this_index] = False
        mask_alignment[this_index] = False
        mask_cohesion[this_index] = False

        a_separation = np.zeros(2)
        a_cohesion = np.zeros(2)
        a_alignment = np.zeros(2)

        # print(mask_in_perseption_radius)
        if np.any(mask_cohesion):
            a_cohesion = compute_cohesion(boids_neigh, this_index, mask_cohesion)
        if np.any(mask_separation):
            a_separation = compute_separation(boids_neigh, this_index, mask_separation)
        if np.any(mask_alignment):
            a_alignment = compute_alignment(boids_neigh, this_index, mask_alignment)
        # noise = compute_noise(boids_neigh[i])

        # print('a = ', a_cohesion, a_separation, a_alignment)
        # print()


        acceleration = coeff[0] * a_cohesion \
                       + coeff[1] * a_separation \
                       + coeff[2] * a_alignment
        boids[i, 4:6] = acceleration

        # if i == 0:
        #     for j in range(neighbours.shape[0]):
        #         neighbours[j] = mask_alignment[j]
        # neighbours = mask_alignment[:]

    # коллизия
    compute_walls_interations(boids, screen_size)  # if np.any(mask_walls, axis=0) else np.zeros(2)
    # return boids[neighbours]


# @njit(parallel=True)
def calculate_grid(boids, grid, grid_size, indexes_in_grid, cell_size):
    indexes_in_grid[:] = boids[:, 0:2] // cell_size
    grid_size[:] = 0
    for i in range(indexes_in_grid.shape[0]):
        row = indexes_in_grid[i][0]
        col = indexes_in_grid[i][1]
        index = grid_size[row, col]
        grid[row, col][index] = i
        grid_size[row, col] += 1


def propagate(boids: np.ndarray, dt: float, velocity_range: np.array, grid: np.ndarray, indexes_in_grid: np.ndarray, perception_radius: int, grid_size):
    """
    Пересчет скоростей за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    vclip(boids[:, 2:4], velocity_range)  # обрезаем скорости, если они вышли за velocity_range
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt




