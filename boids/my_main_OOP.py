import time

import numpy as np

from my_funcs import *
import config as config

from vispy import app, scene
from vispy.geometry import Rect

app.use_app('pyqt6')

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QWidget, QLabel, QCheckBox


class BoidsSimulation(QMainWindow):
    def __init__(self):
        super().__init__()

        # слайдеры

        # self.wall_bounce_checkbox = None

        # self.r_vecs = np.zeros((100, 2), dtype=float)
        # self.count = 0

        self.cohesion_slider = None
        self.separation_slider = None
        self.alignment_slider = None
        self.perception_radius_slider = None

        self.separation_label = None
        self.alignment_label = None
        self.cohesion_label = None
        self.perception_radius_label = None

        #
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # ресурсы
        self.delta_time = 0.0
        self.coeffs = config.coeffs
        self.W, self.H = config.W, config.H  # размеры экрана
        self.N = config.N  # кол-во птиц
        self.size = config.size
        self.perception_radius = config.perception_radius
        self.velocity_range = config.velocity_range
        # self.acceleration_range = config.acceleration_range
        self.max_speed_magnitude = config.max_speed_magnitude
        self.max_delta_velocity_magnitude = config.max_delta_velocity_magnitude

        # boids
        self.boids = np.zeros((self.N, 6), dtype=np.float64)  # boids[i] == [x, y, vx, vy, dvx, dvy]
        init_boids(self.boids, self.size, self.velocity_range)  # создаем птиц
        self.main_characters_boids = self.boids[0:1]

        self.grid = np.empty((
            int(self.size[0] // (2 * self.perception_radius)) + 2,
            int(self.size[1] // (2 * self.perception_radius)) + 2,
            self.N
        ), dtype=int)

        self.indexes_in_grid = (self.boids[:, 0:2] // (2 * self.perception_radius)).astype(int)

        self.grid_size = np.zeros((self.grid.shape[0], self.grid.shape[1]), dtype=int)

        for i in range(self.indexes_in_grid.shape[0]):
            row = self.indexes_in_grid[i][0]
            col = self.indexes_in_grid[i][1]
            index = self.grid_size[row, col]
            self.grid[row, col][index] = i
            self.grid_size[row, col] += 1

        # canvas
        self.canvas = scene.SceneCanvas(show=True, size=(self.W, self.H))  # создаем сцену
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=Rect(0, 0, self.size[0], self.size[1]))
        self.arrows = scene.Arrow(arrows=directions(self.boids, self.delta_time),
                                  arrow_color=(1, 1, 1, 0.9),
                                  arrow_size=5,
                                  connect='segments',
                                  parent=self.view.scene)
        self.red_arrows = scene.Arrow(arrows=directions(self.boids[0:1], self.delta_time),
                                      arrow_color=(1, 0, 0, 1),
                                      arrow_size=10,
                                      connect='segments',
                                      parent=self.view.scene)
        # self.blue_arrows = scene.Arrow(
        #     arrow_color=(0, 1, 0, 1),
        #     arrow_size=7.5,
        #     connect='segments',
        #     parent=self.view.scene)

        # слайдеры
        self.create_sliders(layout, self.W, self.H)
        self.setLayout(layout)

        # таймер
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def create_sliders(self, layout, width, height):
        # отобразить инфо на слайдерах

        self.cohesion_label = QLabel(self)
        self.cohesion_label.setText(f"Cohesion: {self.coeffs['cohesion']}")
        self.cohesion_slider = QSlider(Qt.Orientation.Horizontal)

        self.separation_label = QLabel(self)
        self.separation_label.setText(f"Separation: {self.coeffs['separation']}")
        self.separation_slider = QSlider(Qt.Orientation.Horizontal)

        self.alignment_label = QLabel(self)
        self.alignment_label.setText(f"Alignment: {self.coeffs['alignment']}")
        self.alignment_slider = QSlider(Qt.Orientation.Horizontal)

        self.perception_radius_label = QLabel(self)
        self.perception_radius_label.setText(f"Perception radius: 1 / {int(1 / self.perception_radius)}")
        self.perception_radius_slider = QSlider(Qt.Orientation.Horizontal)

        # self.wall_bounce_checkbox = QCheckBox("Wall bounce", self)
        # self.wall_bounce_checkbox.stateChanged.connect(self.wall_bounce_change)
        # self.wall_bounce_checkbox.setChecked(False)

        # изменить переменные, за которые отвечают слайдеры

        self.cohesion_slider.setRange(config.cohesion_range[0], config.cohesion_range[1])
        self.cohesion_slider.setValue(int(self.coeffs["cohesion"] * config.slider_multiplier))
        self.cohesion_slider.valueChanged.connect(self.cohesion_change)

        self.separation_slider.setRange(config.separation_range[0], config.separation_range[1])
        self.separation_slider.setValue(int(self.coeffs["separation"] * config.slider_multiplier))
        self.separation_slider.valueChanged.connect(self.separation_change)

        self.alignment_slider.setRange(config.alignment_range[0], config.alignment_range[1])
        self.alignment_slider.setValue(int(self.coeffs["alignment"] * config.slider_multiplier))
        self.alignment_slider.valueChanged.connect(self.alignment_change)

        self.perception_radius_slider.setRange(config.perception_radius_range[0], config.perception_radius_range[1])
        self.perception_radius_slider.setValue(int((1 / self.perception_radius) * config.slider_multiplier))
        self.perception_radius_slider.valueChanged.connect(self.perception_radius_change)

        # установка в layout

        layout.addWidget(self.canvas.native)

        # layout.addWidget(self.wall_bounce_checkbox)

        layout.addWidget(self.cohesion_label)
        layout.addWidget(self.separation_label)
        layout.addWidget(self.alignment_label)
        layout.addWidget(self.perception_radius_label)

        layout.addWidget(self.cohesion_slider)
        layout.addWidget(self.separation_slider)
        layout.addWidget(self.alignment_slider)
        layout.addWidget(self.perception_radius_slider)

    def cohesion_change(self, value):
        value = value / config.slider_multiplier
        self.coeffs["cohesion"] = float(value)
        self.cohesion_label.setText(f"Cohesion: {self.coeffs['cohesion']}")
        print(f"Cohesion changed to: {value}")

    def separation_change(self, value):
        value = value / config.slider_multiplier
        self.coeffs["separation"] = float(value)
        self.separation_label.setText(f"Separation: {value}")
        print(f"Separation changed to: {value}")

    def alignment_change(self, value):
        value = value / config.slider_multiplier
        self.coeffs["alignment"] = float(value)
        self.alignment_label.setText(f"Alignment: {value}")
        print(f"Alignment changed to: {value}")

    def perception_radius_change(self, value):
        value = int(value / config.slider_multiplier)
        self.perception_radius = 1 / value
        self.perception_radius_label.setText(f"Perception radius: 1 / {value}")
        print(f"Perception radius changed to: 1 / {value}")

    # def wall_bounce_change(self, state):
    #     if state == 2:
    #         self.wall_bounce = True
    #     else:
    #         self.wall_bounce = False
    #     print(f"Wall bounce changed to: {self.wall_bounce}")

    def update(self):
        start_time = time.time()  # начало отсчета времени
        coeffs_for_numba = np.array([self.coeffs["cohesion"], self.coeffs["separation"], self.coeffs["alignment"]])
        # neighbours =
        flocking(self.boids, self.perception_radius, coeffs_for_numba, config.size, self.indexes_in_grid, self.grid, self.grid_size)  # пересчет ускорений (взаимодействие между птицами)
        calculate_grid(self.boids, self.grid, self.grid_size, self.indexes_in_grid, 2 * self.perception_radius)
        propagate(self.boids, self.delta_time, config.velocity_range, self.grid, self.indexes_in_grid, self.perception_radius, self.grid_size)  # пересчет скоростей на основе ускорений
        self.arrows.set_data(arrows=directions(self.boids, self.delta_time))  # отрисовка стрелок
        self.red_arrows.set_data(arrows=directions(self.boids[0:1], self.delta_time))  # отрисовка стрелок
        # self.blue_arrows.set_data(arrows=directions(neighbours, self.delta_time))  # отрисовка стрелок
        self.canvas.update()  # отображение

        # self.r_vecs[self.count] = self.boids[0, 0:2]
        # self.count += 1
        # if (self.count == 20):
        #     arr = np.array(self.r_vecs)
        #     dr = arr[1:self.count] - arr[:self.count-1]
        #     print('MEAN = ', np.mean(np.linalg.norm(dr, axis=1)))

        # time.sleep(0.05) # строка для проверки того, что игра FPS независимая
        end_time = time.time()  # конец отсчета времени
        self.delta_time = end_time - start_time


if __name__ == '__main__':
    app.create()
    simulation_window = BoidsSimulation()
    simulation_window.show()
    simulation_window.canvas.measure_fps()
    app.run()
