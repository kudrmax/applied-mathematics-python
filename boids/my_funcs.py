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


@njit
def compute_distance(boids: np.ndarray, i: int):
    """
    Вычисление расстояний между птицами
    """
    dr = boids[i, 0:2] - boids[:, 0:2]
    return np.sqrt(dr[:, 0] ** 2 + dr[:, 1] ** 2)


def clip_vector(v: np.ndarray, vector_range: np.array):
    """
    Обрезать вектор, если он выходит за vector_range[1]
    """
    norm = np.linalg.norm(v, axis=1)
    mask = norm > vector_range[1]
    if np.any(mask):
        v[mask] *= vector_range[1] / norm[mask, None]


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
        if np.linalg.norm(delta_steering_v) > config.max_acceleration_magnitude:
            delta_steering_v /= np.linalg.norm(delta_steering_v)
            delta_steering_v *= config.max_acceleration_magnitude
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
    if np.linalg.norm(delta_steering_v) > config.max_acceleration_magnitude:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= config.max_acceleration_magnitude
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
    if np.linalg.norm(delta_steering_v) > config.max_acceleration_magnitude:
        delta_steering_v = delta_steering_v / np.linalg.norm(delta_steering_v)
        delta_steering_v *= config.max_acceleration_magnitude
    return delta_steering_v


@njit(parallel=True)
def compute_walls_interations(boids: np.ndarray, screen_size: np.array):
    """
    Расчет взаимодействия птиц со стенами
    """
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
def get_mask_sector(boids: np.ndarray, mask: np.array, id: int, alpha: float):
    """
    Вычисление макси сектора
    """
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


def get_quarter(boids, cell_size):
    coords = boids[:, 0:2]
    coords_in_cell = coords[:] % cell_size
    x_quarters = coords_in_cell[:, 0] >= cell_size / 2
    y_quarters = coords_in_cell[:, 1] >= cell_size / 2

    quarters = np.empty(boids.shape[0], dtype=int)
    for i in range(x_quarters.shape[0]):
        x_quarter = x_quarters[i]
        y_quarter = y_quarters[i]
        if x_quarter and y_quarter:
            quarters[i] = 1
        elif not x_quarter and y_quarter:
            quarters[i] = 2
        elif not x_quarter and not y_quarter:
            quarters[i] = 2
        elif x_quarter and not y_quarter:
            quarters[i] = 4

    return quarters


# @njit(parallel=True)
def calculate_grid(boids, grid, grid_size, indexes_in_grid, cell_size):
    """
    Заполнение сетки, для вычисления расстояния.

    - Обновляем grid
    - Заполняем заново indexes_in_grid
    - Заполняем заново grid_size

    Parameters
    ----------
    grid
    grid_size
    indexes_in_grid
    cell_size
    """
    indexes_in_grid[:] = boids[:, 0:2] // cell_size
    grid_size[:] = 0
    for i in range(indexes_in_grid.shape[0]):
        row, col = indexes_in_grid[i]
        index = grid_size[row, col]
        grid[row, col][index] = i
        grid_size[row, col] += 1

@njit
def get_mask_grid(grid, grid_size, indexes_in_grid, id):
    row, col = indexes_in_grid[id]
    mask_grid = grid[row, col][:grid_size[row, col]]
    return mask_grid

@njit
def get_index(mask_grid, id):
    # определение индекса боида в новом массиве boids_nearby
    i_nearby = 0
    for j in range(len(mask_grid)):
        if id == mask_grid[j]:
            i_nearby = j
    return i_nearby


@njit(parallel=True)
def calculate_acceleration(boids: np.ndarray,
                           perception_radius: float,
                           coeff: np.array,
                           screen_size: np.array, indexes_in_grid: np.array, grid: np.array, grid_size: np.array):
    """
    Функция, отвечающая за взаимодействие птиц между собой
    """
    # neighbours = np.full(boids.shape[0], False)

    for i in prange(boids.shape[0]):

        # создание макси для боидов, находящихся рядом

        mask_grid = get_mask_grid(grid, grid_size, indexes_in_grid, i)
        boids_nearby = boids[mask_grid]  # боидсы, которые находятся рядом
        i_nearby = get_index(mask_grid, i)

        # расстояния и маски расстояний
        D = compute_distance(boids_nearby, i_nearby)
        mask_in_perception_radius = D < perception_radius

        mask_sector = mask_in_perception_radius
        # mask_sector = get_mask_sector(boids_nearby, mask_in_perception_radius, i_nearby, alpha=30.0)
        mask_alignment = np.logical_and(mask_in_perception_radius, mask_sector)
        mask_separation = np.logical_and(D < perception_radius / 2.0, mask_sector)
        mask_cohesion = np.logical_xor(mask_separation, mask_alignment)

        mask_separation[i_nearby] = False
        mask_alignment[i_nearby] = False
        mask_cohesion[i_nearby] = False

        # считаем ускорения
        a_separation = np.zeros(2)
        a_cohesion = np.zeros(2)
        a_alignment = np.zeros(2)

        if np.any(mask_cohesion):
            a_cohesion = compute_cohesion(boids_nearby, i_nearby, mask_cohesion)
        if np.any(mask_separation):
            a_separation = compute_separation(boids_nearby, i_nearby, mask_separation)
        if np.any(mask_alignment):
            a_alignment = compute_alignment(boids_nearby, i_nearby, mask_alignment)
        # noise = compute_noise(boids_nearby[i])

        acceleration = coeff[0] * a_cohesion \
                       + coeff[1] * a_separation \
                       + coeff[2] * a_alignment
        boids[i, 4:6] = acceleration

        # боиды, которые попали в зону видимости боида с индексом 0
        # if i == 0:
        #     for j in range(neighbours.shape[0]):
        #         neighbours[j] = mask_alignment[j]
        # neighbours = mask_alignment[:]

    # return boids[neighbours]


def calculate_velocity(boids: np.ndarray, dt: float, velocity_range: np.array):
    """
    Пересчет скоростей за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    clip_vector(boids[:, 2:4], velocity_range)  # обрезаем скорости, если они вышли за velocity_range


def calculate_position(boids: np.ndarray, dt: float):
    """
    Пересчет позиции за время dt
    """
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt
