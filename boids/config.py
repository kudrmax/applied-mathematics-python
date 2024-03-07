import numpy as np

# настраиваемые

W, H = 2560 / 2, 1600 / 4  # размеры экрана
N = 5000  # кол-во птиц
perception_radius = 1 / 30

# velocity_range = (0, 0.2 / (0.05 / 0.008))  # ограничения на скорости
velocity_range = (0, 0.2)  # ограничения на скорости

angle = 45

slider_multiplier = 10000
cohesion_range = (0, 0.5)
separation_range = cohesion_range
alignment_range = cohesion_range
separation_from_walls_range = (0, 0.5)
perception_radius_range = (1, 80)
angle_range = (0, 360)

arrow_size = 5

max_speed_magnitude = 4  # 4
max_acceleration_magnitude = 10  # 10

# max_speed_magnitude = 4 / (0.05 / 0.008)  # 4
# max_delta_velocity_magnitude =  10 / (0.05 / 0.008) # 10

coeffs = {
    'cohesion': .0,
    'separation': .0,
    'alignment': .0,
    'separation_from_walls': .0,
}

# внутренние расчеты

size = np.array([W / H, 1])

cohesion_range = (int(cohesion_range[0] * slider_multiplier), int(cohesion_range[1] * slider_multiplier))
separation_range = (int(separation_range[0] * slider_multiplier), int(separation_range[1] * slider_multiplier))
alignment_range = (int(alignment_range[0] * slider_multiplier), int(alignment_range[1] * slider_multiplier))
perception_radius_range = (int(perception_radius_range[0] * slider_multiplier), int(perception_radius_range[1] * slider_multiplier))
separation_from_walls_range = (int(separation_from_walls_range[0] * slider_multiplier), int(separation_from_walls_range[1] * slider_multiplier))
