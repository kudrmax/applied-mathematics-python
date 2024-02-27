import numpy as np

# настраиваемые

W, H = 640, 480  # размеры экрана
N = 2000  # кол-во птиц
fraction_of_perception_radius = 1 / 20

velocity_range = (0, .1)  # ограничения на скорости
acceleration_range = (0, .1)  # ограничения на ускорения

cohesion_range = (1, 50)
separation_range = (1, 50)
alignment_range = (1, 50)
sider_step = 0.01

max_speed_magnitude = 4
max_delta_velocity_magnitude = 10

coeffs = {
    'cohesion': .02,
    'separation': .01,
    'alignment': .02,
}

# ненастраиваемые

size = np.array([W / H, 1])
perception_radius = size[0] * fraction_of_perception_radius
