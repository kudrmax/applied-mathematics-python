import time

import imageio

from my_funcs import *
import config as config

from vispy import app, scene
from vispy.geometry import Rect
from vispy.scene.visuals import Text

app.use_app('pyqt6')

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QWidget, QLabel, QCheckBox


class BoidsSimulation(QMainWindow):
    def __init__(self):
        super().__init__()

        # для записи видео
        self.writer = imageio.get_writer('video.mp4', fps=60)

        # слайдеры

        self.sector_checkbox = None
        self.zoom_camera_checkbox = None

        self.cohesion_slider = None
        self.separation_slider = None
        self.alignment_slider = None
        self.separation_from_walls_slider = None
        self.angle_slider = None

        self.separation_label = None
        self.alignment_label = None
        self.cohesion_label = None
        self.separation_from_walls_label = None
        self.angle_label = None

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # ресурсы
        self.delta_time = 0.05
        self.coeffs = config.coeffs
        self.W, self.H = config.W, config.H  # размеры экрана
        self.N = config.N  # кол-во птиц
        self.size = config.size
        self.perception_radius = config.perception_radius
        self.velocity_range = config.velocity_range
        self.max_speed_magnitude = config.max_speed_magnitude
        self.angle = config.angle
        self.sector_flag = False
        self.zoom_camera_flag = False
        self.frame_count = 0

        # boids
        self.boids = np.zeros((self.N, 6), dtype=np.float64)  # boids[i] == [x, y, vx, vy, dvx, dvy]
        init_boids(self.boids, self.size, self.velocity_range)  # создаем птиц
        self.main_character_boids = self.boids[0:1]
        self.neighbours_of_main_character = np.empty(self.N, dtype=int)
        self.neighbours_of_main_character_size = np.array([0], dtype=int)
        self.main_character_velocity = np.empty((2, 2), dtype=float)
        self.arrows_color = np.empty((self.N, 4), dtype=float)

        # grid
        self.cell_size = 2 * self.perception_radius
        self.grid = np.empty((
            int(self.size[0] // self.cell_size) + 1,
            int(self.size[1] // self.cell_size) + 1,
            self.N
        ), dtype=int)
        self.indexes_in_grid = np.empty(shape=(self.boids.shape[0], 2), dtype=int)
        self.grid_size = np.empty((self.grid.shape[0], self.grid.shape[1]), dtype=int)
        calculate_grid(self.boids, self.grid, self.grid_size, self.indexes_in_grid, self.cell_size)

        # canvas
        self.canvas = scene.SceneCanvas(show=True, size=(self.W, self.H), resizable=False)  # создаем сцену
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=Rect(0, 0, self.size[0], self.size[1]))

        # стрелки
        self.arrows = scene.Arrow(
            arrows=directions(self.boids, self.delta_time),
            arrow_color=(1, 1, 1, 0.7),
            arrow_size=10,
            connect='segments',
            parent=self.view.scene
        )
        self.main_character_arrows = scene.Arrow(
            arrows=directions(self.boids[0:1], self.delta_time),
            arrow_color=(0.471, 0.471, 1, 1),
            arrow_size=10,
            connect='segments',
            parent=self.view.scene
        )
        self.main_character_velocity_line = scene.Line(
            pos=np.array([[0, 0], [0, 0]]),
            color=(0.471, 0.471, 1, 1),
            width=1,
            connect='strip',
            parent=self.view.scene
        )
        self.neighbours_of_main_character_arrows = scene.Arrow(
            arrow_color=(0.431, 0.812, 0, 1),
            arrow_size=10,
            connect='segments',
            parent=self.view.scene
        )
        # self.main_character_visual_range = scene.Ellipse(
        #     center=self.main_character_boids[0][0:2],
        #     radius=self.perception_radius,
        #     color=(0, 0, 1, 0.3), border_width=0,
        #     num_segments=100,
        #     parent=self.view.scene
        # )

        # текст
        self.text = Text(parent=self.canvas.scene, color='white', face='Consolas')
        self.text.pos = self.canvas.size[0] // 2, self.canvas.size[1] // 10
        self.text.font_size = 20
        self.update_text()

        # layout
        self.create_sliders(layout)
        self.setLayout(layout)

        # таймер
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def create_sliders(self, layout):
        """
        Функция, для создания слайдеров и чекбоксов
        """

        # создание

        self.zoom_camera_checkbox = QCheckBox("Zoom", self)
        self.zoom_camera_checkbox.stateChanged.connect(self.zoom_camera)
        self.zoom_camera_checkbox.setChecked(False)

        self.cohesion_label = QLabel(self)
        self.cohesion_label.setText(f"Cohesion: {self.coeffs['cohesion']}")
        self.cohesion_slider = QSlider(Qt.Orientation.Horizontal)

        self.separation_label = QLabel(self)
        self.separation_label.setText(f"Separation: {self.coeffs['separation']}")
        self.separation_slider = QSlider(Qt.Orientation.Horizontal)

        self.alignment_label = QLabel(self)
        self.alignment_label.setText(f"Alignment: {self.coeffs['alignment']}")
        self.alignment_slider = QSlider(Qt.Orientation.Horizontal)

        self.separation_from_walls_label = QLabel(self)
        self.separation_from_walls_label.setText(f"Separation from walls: {self.coeffs['separation_from_walls']}")
        self.separation_from_walls_slider = QSlider(Qt.Orientation.Horizontal)

        self.angle_label = QLabel(self)
        self.angle_label.setText(f"Sector angle: {self.angle}")
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)

        self.sector_checkbox = QCheckBox("Sector:", self)
        self.sector_checkbox.stateChanged.connect(self.sector_change)
        self.sector_checkbox.setChecked(False)

        # connect

        self.cohesion_slider.setRange(config.cohesion_range[0], config.cohesion_range[1])
        self.cohesion_slider.setValue(int(self.coeffs["cohesion"] * config.slider_multiplier))
        self.cohesion_slider.valueChanged.connect(self.cohesion_change)

        self.separation_slider.setRange(config.separation_range[0], config.separation_range[1])
        self.separation_slider.setValue(int(self.coeffs["separation"] * config.slider_multiplier))
        self.separation_slider.valueChanged.connect(self.separation_change)

        self.alignment_slider.setRange(config.alignment_range[0], config.alignment_range[1])
        self.alignment_slider.setValue(int(self.coeffs["alignment"] * config.slider_multiplier))
        self.alignment_slider.valueChanged.connect(self.alignment_change)

        self.angle_slider.setRange(config.angle_range[0], config.angle_range[1])
        self.angle_slider.setValue(self.angle)
        self.angle_slider.valueChanged.connect(self.angle_change)

        self.separation_from_walls_slider.setRange(
            config.separation_from_walls_range[0],
            config.separation_from_walls_range[1]
        )
        self.separation_from_walls_slider.setValue(int(self.coeffs["separation_from_walls"] * config.slider_multiplier))
        self.separation_from_walls_slider.valueChanged.connect(self.separation_from_walls_change)

        # установка в layout

        layout.addWidget(self.canvas.native)

        layout.addWidget(self.zoom_camera_checkbox)

        layout.addWidget(self.cohesion_label)
        layout.addWidget(self.cohesion_slider)

        layout.addWidget(self.separation_label)
        layout.addWidget(self.separation_slider)

        layout.addWidget(self.alignment_label)
        layout.addWidget(self.alignment_slider)

        layout.addWidget(self.separation_from_walls_label)
        layout.addWidget(self.separation_from_walls_slider)

        layout.addWidget(self.sector_checkbox)

        layout.addWidget(self.angle_label)
        layout.addWidget(self.angle_slider)

    def cohesion_change(self, value):
        """
        Функция, для изменения слайдера, отвечающего за cohesion
        """
        value = value / config.slider_multiplier
        self.coeffs["cohesion"] = float(value)
        self.cohesion_label.setText(f"Cohesion: {self.coeffs['cohesion']}")
        self.update_text()

    def separation_change(self, value):
        """
        Функция, для изменения слайдера, отвечающего за separation
        """
        value = value / config.slider_multiplier
        self.coeffs["separation"] = float(value)
        self.separation_label.setText(f"Separation: {value}")
        self.update_text()

    def alignment_change(self, value):
        """
        Функция, для изменения слайдера, отвечающего за alignment
        """
        value = value / config.slider_multiplier
        self.coeffs["alignment"] = float(value)
        self.alignment_label.setText(f"Alignment: {value}")
        self.update_text()

    def separation_from_walls_change(self, value):
        """
        Функция, для изменения слайдера, отвечающего за separation_from_walls
        """
        value = value / config.slider_multiplier
        self.coeffs["separation_from_walls"] = float(value)
        self.separation_from_walls_label.setText(f"Separation from walls: {value}")
        self.update_text()

    def angle_change(self, value):
        """
        Функция, для изменения слайдера, отвечающего за угол сектора
        """
        self.angle = value
        self.angle_label.setText(f"Sector angle: {value}")
        self.update_text()

    def sector_change(self, state):
        """
        Функция, для изменения чекбокса, отвечающего за форму области, в которой боидсы видят друг друга — сектор или круг.
        """
        if state == 2:
            self.sector_flag = True
        else:
            self.sector_flag = False
        self.update_text()

    def zoom_camera(self, state):
        """
        Функция, для изменения чекбокса, отвечающего за то, куда прикреплена камера —
            следует за боидсом с индексом 0 или никак не закреплена
        """
        if state == 2:
            self.zoom_camera_flag = True
            self.view.camera.center = tuple(self.boids[0, 0:2])
            self.view.camera.zoom(1 / 6)
        else:
            self.zoom_camera_flag = False
            self.view.camera.center = (0.5, 0.5)
            self.view.camera.zoom(6)

    def update_text(self):
        pass
        # self.text.text = f"N = {self.N}, coh = {self.coeffs['cohesion']}, sep = {self.coeffs['separation']}, alg = {self.coeffs['alignment']},\nwalls = {self.coeffs['separation_from_walls']}, angle = {self.angle}, FPS = {round(self.canvas.fps, 2)}"

    def update_graphics(self):
        """
        Функция, для отображения боидсов на экране
        """
        fill_arrow_color(self.boids, self.arrows_color)
        # self.main_character_visual_range.center = self.main_character_boids[0][0:2] # отображение visual_range
        self.main_character_velocity_line.set_data(pos=self.main_character_velocity)
        self.arrows.set_data(color=(1, 0, 0, 1), arrows=directions(self.boids, self.delta_time))
        self.arrows.arrow_color = self.arrows_color
        self.neighbours_of_main_character_arrows.set_data(
            arrows=directions(
                self.boids[self.neighbours_of_main_character[:self.neighbours_of_main_character_size[0]]],
                self.delta_time
            )
        )
        self.main_character_arrows.set_data(arrows=directions(self.boids[0:1], self.delta_time))
        if self.zoom_camera_flag:
            delta_distance = self.boids[0, 0:2] - self.view.camera.center[0:2]
            self.view.camera.pan(delta_distance)

        self.canvas.measure_fps()
        self.setWindowTitle(f"N = {self.N}; FPS = {np.round(self.canvas.fps, 2)};")
        self.update_text()
        self.canvas.update()

    def update(self):
        """
        Функция, которая вызывается в каждом кадре
        """

        # отрисовка
        self.update_graphics()

        # начало отсчета времени
        # start_time = time.time()

        # алгоритм boids (взаимодействие птиц между друг другом)
        calculate_acceleration(
            self.boids,
            self.perception_radius,
            np.array([
                self.coeffs["cohesion"],
                self.coeffs["separation"],
                self.coeffs["alignment"],
                self.coeffs["separation_from_walls"],
                self.coeffs["noise"]
            ]),
            self.grid,
            self.grid_size,
            self.indexes_in_grid,
            self.neighbours_of_main_character,
            self.neighbours_of_main_character_size,
            self.main_character_velocity,
            self.sector_flag,
            self.angle
        )

        # коллизия со стенами
        compute_walls_collition(self.boids, self.size)

        # пересчет сетки для подсчета расстояния
        calculate_grid(self.boids,
                       self.grid,
                       self.grid_size,
                       self.indexes_in_grid,
                       2 * self.perception_radius)

        # пересчет скоростей
        calculate_velocity(self.boids, self.delta_time, config.velocity_range)

        # пересчет позиции
        calculate_position(self.boids, self.delta_time)

        # конец отсчета времени
        # end_time = time.time()
        # self.delta_time = end_time - start_time

        if config.make_video_flag:
            self.frame_count += 1
            if self.frame_count <= 2000:
                frame = self.canvas.render(alpha=False)
                self.writer.append_data(frame)
            # if self.frame_count % 100 == 0:
            #     print(self.frame_count)


if __name__ == '__main__':
    app.create()
    simulation_window = BoidsSimulation()
    simulation_window.show()
    simulation_window.canvas.measure_fps()
    app.run()
