import numpy as np

# настраиваемые

W, H = 1920 * 2, 1080 * 2  # размеры экрана
N = 5000  # кол-во птиц
fraction_of_perception_radius = 1 / 30

velocity_range = (0, .1)  # ограничения на скорости
acceleration_range = (0, .1)  # ограничения на ускорения

slider_multiplier = 1000
cohesion_range = (0, 0.1)
separation_range = (0, 0.1)
alignment_range = (0, 0.1)

max_speed_magnitude = 4
max_delta_velocity_magnitude = 10

coeffs = {
    'cohesion': .02,
    'separation': .01,
    'alignment': .02,
}

# внутренние расчеты

size = np.array([W / H, 1])
perception_radius = size[0] * fraction_of_perception_radius

cohesion_range = (int(cohesion_range[0] * slider_multiplier), int(cohesion_range[1] * slider_multiplier))
separation_range = (int(separation_range[0] * slider_multiplier), int(separation_range[1] * slider_multiplier))
alignment_range = (int(alignment_range[0] * slider_multiplier), int(alignment_range[1] * slider_multiplier))
