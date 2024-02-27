from numba import njit, prange
import numpy as np
import config as config


@njit
def get_normal_vec(vec: np.array) -> np.array:
    vec_rotated = np.array([vec[1], -vec[0]])
    vec_rotated /= np.linalg.norm(vec_rotated)
    return vec_rotated


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


def vclip(v: np.ndarray, velocity_range: tuple[float, float]):
    """
    Если скорость выходит за разрешенные скорости, то мы обрезаем скорость
    """
    norm = np.linalg.norm(v, axis=1)  # модуль скорости
    mask = norm > velocity_range[1]  # маска
    if np.any(mask):
        v[mask] *= velocity_range[1] / norm[mask, None]


def propagate(boids: np.ndarray, dt: float, velocity_range: tuple[float, float]):
    """
    Пересчет скоростей за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    vclip(boids[:, 2:4], velocity_range)  # обрезаем скорости, если они вышли за velocity_range
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

    r = boids[:, :2]
    n = r.shape[0]
    dist = np.empty(shape=(n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            dist[i, j] = np.sqrt(np.sum((r[i] - r[j]) ** 2))
    return dist


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
def compute_separation(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    steer to avoid crowding local flockmates
    """
    steering_pos = np.sum(
        (boids[id][0:2] - boids[mask][:, 0:2])
        / (np.linalg.norm(boids[id][0:2] - boids[mask][:, 0:2]) ** 2),
        axis=0)
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
def compute_noise(boid: np.array):
    return get_normal_vec(boid[2:4]) * np.random.uniform(-0.1, 0.1)


@njit(parallel=True)
def compute_walls_interations(boids: np.ndarray, mask: np.ndarray, screen_size: np.array):
    for i in prange(boids.shape[0]):
        if mask[0][i]:
            boids[i][3] = -boids[i][3]
            # boids[i][3] *= 200000
            boids[i][1] = screen_size[1] - 0.001

        if mask[1][i]:
            boids[i][2] = -boids[i][2]
            # boids[i][2] *= 200000
            boids[i][0] = screen_size[0] - 0.001

        if mask[2][i]:
            boids[i][3] = -boids[i][3]
            # boids[i][3] *= 200000
            boids[i][1] = 0.001

        if mask[3][i]:
            boids[i][2] = -boids[i][2]
            # boids[i][2] *= 200000
            boids[i][0] = 0.001

    for mask_wall in mask:
        for i in prange(boids.shape[0]):
            if mask_wall[i]:
                boids[i, 4:6] = np.zeros(2)


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception_radius: float,
             coeff: np.array,
             screen_size: np.array):
    """
    Функция, отвечающая за взаимодействие птиц между собой
    """
    # считаем расстояния
    distances = compute_distances(boids)  # матрица с расстояниями между всеми птицами
    N = boids.shape[0]
    for i in prange(N):  # выкидываем расстояния между i и i
        distances[i, i] = np.inf

    # считаем маски
    mask_cohesion = distances < (perception_radius * 2) * (distances > perception_radius / 2)
    mask_separation = distances < perception_radius / 2
    mask_alignment = distances < perception_radius

    mask_walls = np.empty((4, boids.shape[0]))
    mask_walls[0] = boids[:, 1] > screen_size[1]
    mask_walls[1] = boids[:, 0] > screen_size[0]
    mask_walls[2] = boids[:, 1] < 0
    mask_walls[3] = boids[:, 0] < 0

    # вычисляем взаимодействие между птицами
    for i in prange(N):
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        alignment = np.zeros(2)

        if np.any(mask_cohesion[i]):
            cohesion = compute_cohesion(boids, i, mask_cohesion[i])
        if np.any(mask_separation[i]):
            separation = compute_separation(boids, i, mask_separation[i])
        if np.any(mask_alignment[i]):
            alignment = compute_alignment(boids, i, mask_alignment[i])
        noise = compute_noise(boids[i])

        boids[i, 4:6] = \
            coeff[0] * cohesion + \
            coeff[1] * separation + \
            coeff[2] * alignment + \
            noise

    # коллизия
    compute_walls_interations(boids, mask_walls, screen_size)  # if np.any(mask_walls, axis=0) else np.zeros(2)
