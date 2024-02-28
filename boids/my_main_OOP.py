import time
from my_and_nikitas_funcs import *
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
        self.wall_bounce_checkbox = None
        self.alignment_label = None
        self.separation_label = None
        self.cohesion_slider = None
        self.separation_slider = None
        self.alignment_slider = None
        self.cohesion_label = None

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
        self.fraction_of_perception_radius = config.fraction_of_perception_radius
        self.velocity_range = config.velocity_range
        self.acceleration_range = config.acceleration_range
        self.max_speed_magnitude = config.max_speed_magnitude
        self.max_delta_velocity_magnitude = config.max_delta_velocity_magnitude

        # boids
        self.boids = np.zeros((self.N, 6),
                              dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, dvx, dvy]
        init_boids(self.boids, self.size, self.velocity_range)  # создаем птиц

        # canvas
        self.canvas = scene.SceneCanvas(show=True, size=(self.W, self.H))  # создаем сцену
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=Rect(0, 0, self.size[0], self.size[1]))
        self.arrows = scene.Arrow(arrows=directions(self.boids, self.delta_time),
                                  arrow_color=(1, 1, 1, 1),
                                  arrow_size=5,
                                  connect='segments',
                                  parent=self.view.scene)

        # слайдеры
        self.create_sliders(layout, self.W, self.H)
        self.setLayout(layout)

        # таймер
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def create_sliders(self, layout, width, height):
        self.cohesion_label = QLabel(self)
        self.cohesion_label.setText(f"Cohesion: {self.coeffs['cohesion']}")
        self.cohesion_slider = QSlider(Qt.Orientation.Horizontal)

        self.separation_label = QLabel(self)
        self.separation_label.setText(f"Separation: {self.coeffs['separation']}")
        self.separation_slider = QSlider(Qt.Orientation.Horizontal)

        self.alignment_label = QLabel(self)
        self.alignment_label.setText(f"Alignment: {self.coeffs['alignment']}")
        self.alignment_slider = QSlider(Qt.Orientation.Horizontal)

        self.cohesion_slider.setGeometry(50, 0, int(width * 0.7), 30)
        self.separation_slider.setGeometry(50, 40, int(width * 0.7), 30)
        self.alignment_slider.setGeometry(50, 80, int(width * 0.7), 30)

        self.wall_bounce_checkbox = QCheckBox("Wall bounce", self)
        self.wall_bounce_checkbox.stateChanged.connect(self.wall_bounce_change)
        self.wall_bounce_checkbox.setChecked(False)

        self.cohesion_slider.setRange(config.cohesion_range[0], config.cohesion_range[1])
        self.cohesion_slider.setValue(int(self.coeffs["cohesion"] * config.slider_multiplier))
        self.cohesion_slider.valueChanged.connect(self.cohesion_change)

        self.separation_slider.setRange(config.separation_range[0], config.separation_range[1])
        self.separation_slider.setValue(int(self.coeffs["separation"] * config.slider_multiplier))
        self.separation_slider.valueChanged.connect(self.separation_change)

        self.alignment_slider.setRange(config.alignment_range[0], config.alignment_range[1])
        self.alignment_slider.setValue(int(self.coeffs["alignment"] * config.slider_multiplier))
        self.alignment_slider.valueChanged.connect(self.alignment_change)

        layout.addWidget(self.canvas.native)
        layout.addWidget(self.wall_bounce_checkbox)
        layout.addWidget(self.cohesion_label)
        layout.addWidget(self.cohesion_slider)
        layout.addWidget(self.separation_label)
        layout.addWidget(self.separation_slider)
        layout.addWidget(self.alignment_label)
        layout.addWidget(self.alignment_slider)

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

    def wall_bounce_change(self, state):
        if state == 2:
            self.wall_bounce = True
        else:
            self.wall_bounce = False
        print(f"Wall bounce changed to: {self.wall_bounce}")

    def update(self):
        start_time = time.time()  # начало отсчета времени
        coeffs_for_numba = np.array([self.coeffs["cohesion"], self.coeffs["separation"], self.coeffs["alignment"]])
        flocking(self.boids, config.perception_radius, coeffs_for_numba,
                 config.size)  # пересчет ускорений (взаимодействие между птицами)
        propagate(self.boids, self.delta_time, config.velocity_range)  # пересчет скоростей на основе ускорений
        paint_arrows(self.arrows, self.boids, self.delta_time)  # отрисовка стрелок
        self.canvas.update()  # отображение

        # time.sleep(0.05) # строка для проверки того, что игра FPS независимая
        end_time = time.time()  # конец отсчета времени
        self.delta_time = end_time - start_time


if __name__ == '__main__':
    app.create()
    simulation_window = BoidsSimulation()
    simulation_window.show()
    simulation_window.canvas.measure_fps()
    app.run()
