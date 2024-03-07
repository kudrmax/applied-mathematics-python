import numpy as np
from numba import njit, prange
import config as config


def init_boids(boids: np.ndarray, screen_size: tuple, velocity_range: tuple = (0., 1.)):
    """
    Функция для создания массивая боидсов

    Parameters
    ----------
    boids: матрица (N, 6), где boids[i] соответствует массиву [x, y, vx, vy, ax, ay], где v и a — скорость и ускорения соответственно
    screen_size: размер области, где screen_size = [ширина, высота]
    velocity_range: максимальная и минимальаня скорости
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
    Функция для нахождения массива с координатами вершин стрелки, соответствующей бойдсу

    Parameters
    ----------
    boids: матрица (N, 6), где boids[i] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    dt: длительность одного кадра

    Returns
    -------
    Массив вида (N, 4) для отрисовки векторов стрелок,
    где i-тая строка соответствует массиву [x0, y0, x, y],
    где [x0, y0] и [x, y] — предыдущее и нынешнее положение i-той птицы
    """
    pos = boids[:, :2]  # положение в момент времени t
    delta_pos = dt * boids[:, 2:4]  # как изменилось положение за dt
    pos0 = pos - delta_pos  # положение в момент времени t-dt
    return np.hstack((pos0, pos))


@njit
def compute_distance(boids: np.ndarray, i: int):
    """
    Функция для нахождения расстояний между i-ым бойтсом и всеми остальными

    Parameters
    ----------
    boids: матрица (N, 6), где boids[i] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    i: номер бойтса

    Returns
    -------
    Массив, где на j-том месте находится расстояние между i-ым и j-ым бойтсом
    """
    dr = boids[i, 0:2] - boids[:, 0:2]
    return np.sqrt(dr[:, 0] ** 2 + dr[:, 1] ** 2)


def clip_ndarray(v: np.ndarray, range: np.array):
    """
    Функция, которая для каждой строки матрицы обрезает строку до range[1] если норма строки больше range[1]
    или увеличивает стркоу до range[1] если ее норма меньше range[1].
    """
    norm = np.linalg.norm(v, axis=1)
    mask_greater = norm > range[1]
    mask_lower = norm < range[0]
    if np.any(mask_greater):
        v[mask_greater] *= range[1] / norm[mask_greater, None]
    if np.any(mask_lower):
        v[mask_lower] *= range[0] / norm[mask_lower, None]


@njit
def clip_array(v: np.array, range: np.array):
    """
    Функция, которая обрезает вектор v до range[1] если его норма больше range[1]
    или увеличивает его до range[1] если его норма меньше range[1].
    """
    norm = np.linalg.norm(v)
    if norm > range[1]:
        v *= range[1] / norm
    elif norm < range[0]:
        v *= range[0] / norm


@njit
def compute_cohesion(boids: np.ndarray, id: int, mask: np.array) -> np.array:
    """
    Вычисление ускорения, соответствующего типу взаимодействия cohesion
    Определение cohesion: steer to move towards the average position (center of mass) of local flockmates

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    id: номер строки, соответствующей данному боидсу
    mask: маска боидсов, который взаимодействуют с id-ым боидсом с типом взаимодействия cohesion

    Returns
    -------
    Вектор ускорения
    """
    if boids[mask].shape[0] > 1:
        steering_pos = np.sum(boids[mask][:, 0:2], axis=0)
        steering_pos /= boids[mask].shape[0]
        delta_steering_pos = steering_pos - boids[id][0:2]
        # delta_steering_pos /= np.linalg.norm(delta_steering_pos)
        # delta_steering_pos *= config.max_speed_magnitude
        delta_steering_v = delta_steering_pos - boids[id, 2:4]
        clip_array(delta_steering_v, range=config.acceleration_range)
        return delta_steering_v
    else:
        return np.zeros(2)


@njit
def compute_separation(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    Вычисление ускорения, соответствующего типу взаимодействия separation
    Определение separation: steer to avoid crowding local flockmates

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    id: номер строки, соответствующей данному боидсу
    mask: маска боидсов, который взаимодействуют с id-ым боидсом с типом взаимодействия separation

    Returns
    -------
    Вектор ускорения
    """
    dr = boids[id][0:2] - boids[mask][:, 0:2]
    dr *= 1 / ((dr[:, 0] ** 2 + dr[:, 1] ** 2) + 0.001)
    steering_pos = np.sum(dr, axis=0)
    # steering_pos /= np.linalg.norm(steering_pos)
    # steering_pos *= config.max_speed_magnitude
    delta_steering_v = steering_pos - boids[id, 2:4]
    clip_array(delta_steering_v, range=config.acceleration_range)
    return delta_steering_v


@njit
def compute_alignment(boids: np.ndarray, id: int, mask: np.ndarray) -> np.array:
    """
    Вычисление ускорения, соответствующего типу взаимодействия alignment
    Определение alignment: steer towards the average heading of local flockmates

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    id: номер строки, соответствующей данному боидсу
    mask: маска боидсов, который взаимодействуют с id-ым боидсом с типом взаимодействия alignment

    Returns
    -------
    Вектор ускорения
    """
    steering_v = np.sum(boids[mask][:, 2:4], axis=0)
    # steering_v /= np.linalg.norm(steering_v)
    # steering_v *= config.max_speed_magnitude
    delta_steering_v = steering_v - boids[id][2:4]
    clip_array(delta_steering_v, range=config.acceleration_range)
    return delta_steering_v


@njit(parallel=True)
def compute_walls_collition(boids: np.ndarray, screen_size: np.array):
    """
    Функция для вычисления коллизии между боидсами и стенами.
    Если боидс оказался внутри стены, то развернуть его вектор скорости в зависимости от того,
    в какой стене он оказался (абсолютно упругое взаиможействие)
    и поменять положение боидса так, чтобы он не находился в стене

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    screen_size: размер области, где screen_size = [ширина, высота]
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
    Функция, для создания маски боидсов, которые попали в сектор с углом alpha

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    id: номер строки, соответствующей данному боидсу
    mask: маска боидсов, которые попадают в зону видимости, если зона видимости — оркужность
    alpha: угол сектора

    Returns
    -------
    Маска боидсов, которые попали в сектор
    """
    mask[id] = False
    alpha = alpha // 2
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


def calculate_grid(boids: np.ndarray, grid: np.ndarray, grid_size: np.ndarray, indexes_in_grid: np.ndarray, cell_size: int):
    """
    Заполнение сетки grid, которая нужна для более эффективного вычисления расстояния между боидсами.

    Что делает функция:
    1. Обновляет grid
    2. Заполняет заново indexes_in_grid
    3. Заполняет заново grid_size

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    grid: матрица, где в ячейке [i, j] хранятся индексы боидсов, которые принадлежал клетке, соответствующей клетке [i, j]
    grid_size: матрица, где в ячейке [i, j] хранится количество индексов в grid (остальное мусор)
    indexes_in_grid: матрица, где в строке k хранятся индексы [i, j] такие, что grid[i, j] содержит индекс k
    cell_size: размер ячеек, на которые делится весь экран
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
    """
    Функция, которая для id-го бойдса возвращает индексы боидсов, которые находится в соседних с ней клетках, т.е. "близко".

    Parameters
    ----------
    id: номер строки, соответствующей данному боидсу
    grid: матрица, где в ячейке [i, j] хранятся индексы боидсов, которые принадлежал клетке, соответствующей клетке [i, j]
    grid_size: матрица, где в ячейке [i, j] хранится количество индексов в grid (остальное мусор)
    indexes_in_grid: матрица, где в строке k хранятся индексы [i, j]

    Returns
    -------

    """
    row, col = indexes_in_grid[id]
    cells = np.empty(shape=(9, 2), dtype=np.int64)
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            this_row, this_col = row + i, col + j
            if this_row >= 0 and this_col >= 0:
                cells[(i + 1) * 3 + (j + 1)] = np.array([this_row, this_col])
            else:
                cells[(i + 1) * 3 + (j + 1)] = np.array([666, 666])
                # 666 — число, которое будет означать, что мы вышли за границы массива и там хранится мусор

    # у меня не получилось сделать это циклом или еще как либо из-за numba, поэтмоу я придумал такой страшный код:

    row, col = cells[0]
    mask0 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[1]
    mask1 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[2]
    mask2 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[3]
    mask3 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[4]
    mask4 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[5]
    mask5 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[6]
    mask6 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[7]
    mask7 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)
    row, col = cells[8]
    mask8 = grid[row, col][:grid_size[row, col]] if row != 666 and col != 666 else np.empty(0, dtype=np.int64)

    mask_grid_flatten = np.array([
        *mask0,
        *mask1,
        *mask2,
        *mask3,
        *mask4,
        *mask5,
        *mask6,
        *mask7,
        *mask8,
    ])

    return mask_grid_flatten


@njit
def get_nearby_index(mask_grid, id):
    """
    Определение индекса боида в новом массиве boids_nearby

    Parameters
    ----------
    mask_grid: маска, которая содержит индексы (индексы, не bool!) боидсов, которые находится "рядом" с id-ым боидсом
    id: номер строки, соответствующей данному боидсу
    """
    i_nearby = 0
    for j in range(len(mask_grid)):
        if id == mask_grid[j]:
            i_nearby = j
    return i_nearby


@njit
def compute_separation_from_walls(id, indexes_in_grid, grid):
    """
    Вычисление ускорения, которое отталкивает птиц от стен.
    Если птица находится в граничной клетке, т.е. рядом со стеной, то ускорение направленно в противоположную сторону от стены.

    Parameters
    ----------
    id: номер строки, соответствующей данному боидсу
    grid: матрица, где в ячейке [i, j] хранятся индексы боидсов, которые принадлежал клетке, соответствующей клетке [i, j]
    indexes_in_grid: матрица, где в строке k хранятся индексы [i, j]

    Returns
    -------
    Вектор ускорения
    """
    max_col, max_row = grid.shape[0], grid.shape[1]
    col, row = indexes_in_grid[id]
    acceleration = np.zeros(2, dtype=float)
    if row == 0:
        acceleration[1] = 1
    if row >= max_row - 2:
        acceleration[1] = -1
    if col == 0:
        acceleration[0] = 1
    if col >= max_col - 2:
        acceleration[0] = -1
    return acceleration


@njit(parallel=True)
def calculate_acceleration(boids: np.ndarray,
                           perception_radius: float,
                           coeff: np.array,
                           grid: np.array,
                           grid_size: np.array,
                           indexes_in_grid: np.array,
                           neighbours_of_main_character,
                           neighbours_of_main_character_size: np.array,
                           main_character_velocity,
                           sector_flag,
                           alpha):
    """
    Алгоритм boids. Вычисление ускорения птиц.

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    perception_radius: радиус, в котором боидсы видят друг друга
    coeff: коэффициенты взаимодействия различных типов ускорений:
        coeff[0] — cohesion
        coeff[1] — separation
        coeff[2] — alignment
        coeff[3] — separation_from_walls
        coeff[4] — noise
    grid: матрица, где в ячейке [i, j] хранятся индексы боидсов, которые принадлежал клетке, соответствующей клетке [i, j]
    grid_size: матрица, где в ячейке [i, j] хранится количество индексов в grid (остальное мусор)
    indexes_in_grid: матрица, где в строке k хранятся индексы [i, j] такие, что grid[i, j] содержит индекс k
    neighbours_of_main_character: массив, в котором хранятся индексы боидсов, которые находятся рядом с боидсом с индексом 0
    neighbours_of_main_character_size: размер массива neighbours_of_main_character (в остальных ячейках мусор)
    main_character_velocity: массив, который содержит координаты начала и конца вектора скорости боидса с индексом 0
    sector_flag: если True, то зона видимости боидсов — сектор угла alpha
    alpha: угол сектора
    """
    for i in prange(boids.shape[0]):

        # создание макси для боидов, находящихся рядом

        mask_grid = get_mask_grid(grid, grid_size, indexes_in_grid, i)
        boids_nearby = boids[mask_grid]  # боидсы, которые находятся рядом
        i_nearby = get_nearby_index(mask_grid, i)

        # расстояния и маски расстояний
        D = compute_distance(boids_nearby, i_nearby)
        mask_in_perception_radius = D < perception_radius

        if sector_flag:
            mask_sector = get_mask_sector(boids_nearby, mask_in_perception_radius, i_nearby, alpha=alpha)
        else:
            mask_sector = mask_in_perception_radius

        # mask_alignment = np.logical_and(mask_in_perception_radius, mask_sector)
        # mask_separation = np.logical_and(D < perception_radius / 2.0, mask_sector)
        # mask_cohesion = np.logical_xor(mask_separation, mask_alignment)

        mask_separation = mask_sector
        mask_alignment = mask_sector
        mask_cohesion = mask_sector

        mask_separation[i_nearby] = False
        mask_alignment[i_nearby] = False
        mask_cohesion[i_nearby] = False

        # считаем ускорения
        separation = np.zeros(2)
        cohesion = np.zeros(2)
        alignment = np.zeros(2)

        if np.any(mask_cohesion):
            cohesion = compute_cohesion(boids_nearby, i_nearby, mask_cohesion)
        if np.any(mask_separation):
            separation = compute_separation(boids_nearby, i_nearby, mask_separation)
        if np.any(mask_alignment):
            alignment = compute_alignment(boids_nearby, i_nearby, mask_alignment)
        separation_from_walls = compute_separation_from_walls(i, indexes_in_grid, grid)
        noise = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])

        acceleration = coeff[0] * cohesion \
                       + coeff[1] * separation \
                       + coeff[2] * alignment \
                       + coeff[3] * separation_from_walls \
                       + coeff[4] * noise
        clip_array(acceleration, range=config.acceleration_range)
        boids[i, 4:6] = acceleration

        if i == 0:
            # заполняем боидов, которые попали в зону видимости боида с индексом 0
            count = 0
            neighbours = mask_grid[mask_alignment]
            for j in range(neighbours.shape[0]):
                neighbours_of_main_character[j] = neighbours[j]
                count += 1
            neighbours_of_main_character_size[0] = count

            # заполняем вектор скорости боида с индексом 0
            main_character_velocity[0] = boids_nearby[i_nearby][0:2]
            main_character_velocity[1] = boids_nearby[i_nearby][0:2] + boids_nearby[i_nearby][2:4]


def calculate_velocity(boids: np.ndarray, dt: float, velocity_range: np.array):
    """
    Пересчет скоростей боидсов за время dt
    """
    boids[:, 2:4] += boids[:, 4:6] * dt  # меняем скорости: v += dv, где dv — изменение скорости за dt
    clip_ndarray(boids[:, 2:4], velocity_range)  # обрезаем скорости, если они вышли за velocity_range


def calculate_position(boids: np.ndarray, dt: float):
    """
    Пересчет позиции боидсов за время dt
    """
    boids[:, 0:2] += boids[:, 2:4] * dt  # меняем кооординаты: r += v * dt


@njit(parallel=True)
def fill_arrow_color(boids, arrows_color):
    """
    Функция, которая перекрашивает боидсов в зависимости от их модуля скорости.

    Parameters
    ----------
    boids: матрица (N, 6), где boids[id] соответствует массиву [x, y, vx, vy, ax, ay],
        где v и a — скорость и ускорения соответственно
    arrows_color: вектор, где в i-ой строке находится RGBA цвет i-го боидса
    """
    v = boids[:, 2:4]
    velocity_norm = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    max_velocity, min_velocity = np.max(velocity_norm), np.min(velocity_norm)
    G_max = 0.8
    for i in prange(boids.shape[0]):
        G = (velocity_norm[i] - min_velocity) / (max_velocity - min_velocity) * G_max
        arrows_color[i] = np.array([1, G, 0, 0.9])
