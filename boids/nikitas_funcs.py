import numpy as np
from numba import njit, prange


def paint_arrows(arrows, boids, dt):
    arrows.set_data(arrows=directions(boids, dt))  # отрисовка стрелок

def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    n = boids.shape[0]
    random = np.random.default_rng()
    boids[:, 0] = random.uniform(0., asp, size=n)
    boids[:, 1] = random.uniform(0., 1., size=n)

    alpha = random.uniform(0, 2 * np.pi, size=n)
    velocity = random.uniform(*vrange, size=n)
    cos, sin = np.cos(alpha), np.sin(alpha)
    boids[:, 2], boids[:, 3] = velocity * cos, velocity * sin

    # boids[0] = np.array([0.5, 0.5, 0, 0.5, 0, 0])
    # boids[1] = np.array([0.5, 0.55, 0, -0.1, 0, 0])
    # boids[2] = np.array([0.54, 0.525, -0.1, -0.1, 0, 0])
    # print(boids)


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


@njit
def separation(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    distance_mask[i] = False
    directions = boids[i, :2] - boids[distance_mask][:, :2]
    directions *= (1 / (njit_norm_axis1(directions) + 0.0001))
    acceleration = np.sum(directions, axis=0)
    return acceleration - boids[i, 2:4]


@njit
def alignment(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    velocity = boids[distance_mask][:, 2:4]
    acceleration = np.sum(velocity, axis=0)
    acceleration /= velocity.shape[0]
    return acceleration - boids[i, 2:4]


@njit
def cohesion(boids: np.ndarray, i: int, distance_mask: np.ndarray):
    directions = boids[distance_mask][:, :2] - boids[i, :2]
    acceleration = np.sum(directions, axis=0)
    acceleration /= directions.shape[0]
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


@njit(parallel=True)
def flocking(boids: np.ndarray, perseption: float, coeffitients: np.ndarray, aspect_ratio: float,
             a_range: np.ndarray):
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
            # if np.any(separation_mask):
            #     a_separation = separation(boids, i, separation_mask)
            if np.any(cohesion_mask):
                a_cohesion = cohesion(boids, i, cohesion_mask)
            # a_alignment = alignment(boids, i, perception_mask)
            # a_separation = clip_vector(a_separation, a_range)
            # a_cohesion = clip_vector(a_cohesion, a_range)

        acceleration = coeffitients[0] * a_separation \
                       + coeffitients[1] * a_cohesion \
                       + coeffitients[2] * a_alignment
        boids[i, 4:6] = acceleration * 1000



def propagate(boids, delta_time, v_range):
    boids[:, 2:4] += boids[:, 4:6] * delta_time
    boids[:, 2:4] = clip_array(boids[:, 2:4], v_range)
    boids[:, 0:2] += boids[:, 2:4] * delta_time
