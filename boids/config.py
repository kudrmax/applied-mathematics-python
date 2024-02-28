import numpy as np

# настраиваемые

W, H = 1920 * 2, 1080 * 2  # размеры экрана
N = 5000  # кол-во птиц
perception_radius = 1 / 30

velocity_range = (0, .1)  # ограничения на скорости
acceleration_range = (0, .1)  # ограничения на ускорения

slider_multiplier = 1000
cohesion_range = (0, .5)
separation_range = cohesion_range
alignment_range = cohesion_range
perception_radius_range = (1, 40)

max_speed_magnitude = 4
max_delta_velocity_magnitude = 10

coeffs = {
    'cohesion': .0,
    'separation': .0,
    'alignment': .0,
}


# внутренние расчеты

size = np.array([W / H, 1])

cohesion_range = (int(cohesion_range[0] * slider_multiplier), int(cohesion_range[1] * slider_multiplier))
separation_range = (int(separation_range[0] * slider_multiplier), int(separation_range[1] * slider_multiplier))
alignment_range = (int(alignment_range[0] * slider_multiplier), int(alignment_range[1] * slider_multiplier))
# perception_radius_range = (int(perception_radius_range[0] * slider_multiplier), int(perception_radius_range[1] * slider_multiplier))