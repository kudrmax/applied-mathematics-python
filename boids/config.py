import numpy as np

# настраиваемые

make_video_flag = False
W, H = 2560 / 2, 1600 / 4  # размеры экрана
N = 10000  # кол-во птиц
perception_radius = 1 / 30

velocity_range = (0.005, 0.05)  # ограничения на скорости
acceleration_range = (0.0, 0.8 * velocity_range[1])  # ограничения на скорости

angle = 45

slider_multiplier = 10000
cohesion_range = (0, 1.5)
separation_range = (0, 3.0)
alignment_range = (0, 3.0)
separation_from_walls_range = (0, 0.5)
perception_radius_range = (1, 80)
angle_range = (0, 360)

arrow_size = 5

max_speed_magnitude = 1  # 4

# max_speed_magnitude = 4 / (0.05 / 0.008)  # 4
# max_delta_velocity_magnitude =  10 / (0.05 / 0.008) # 10

coeffs = {
    'cohesion': 0.0,
    'separation': 0.0,
    'alignment': 0.0,
    'separation_from_walls': 0.0,
    'noise': 0.0  # 1.0
}

# внутренние расчеты

size = np.array([W / H, 1])

cohesion_range = (int(cohesion_range[0] * slider_multiplier), int(cohesion_range[1] * slider_multiplier))
separation_range = (int(separation_range[0] * slider_multiplier), int(separation_range[1] * slider_multiplier))
alignment_range = (int(alignment_range[0] * slider_multiplier), int(alignment_range[1] * slider_multiplier))
perception_radius_range = (
int(perception_radius_range[0] * slider_multiplier), int(perception_radius_range[1] * slider_multiplier))
separation_from_walls_range = (
int(separation_from_walls_range[0] * slider_multiplier), int(separation_from_walls_range[1] * slider_multiplier))
