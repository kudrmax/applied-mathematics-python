import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
import time
from my_funcs import *
import config as config

app.use_app('pyqt5')

global delta_time
delta_time = 0.0
coeffs = np.array([config.coeffs["cohesion"], config.coeffs["separation"], config.coeffs["alignment"]])

boids = np.zeros((config.N, 6),
                 dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, dvx, dvy]
init_boids(boids, config.size, velocity_range=config.velocity_range)  # создаем птиц

canvas = scene.SceneCanvas(show=True, size=(config.W, config.H))  # создаем сцену
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, config.size[0], config.size[1]))
arrows = scene.Arrow(arrows=directions(boids, delta_time),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)


def update(event):
    global delta_time
    start_time = time.time()  # начало отсчета времени

    flocking(boids, config.perception_radius, coeffs, config.size)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, delta_time, config.velocity_range)  # пересчет скоростей на основе ускорений
    paint_arrows(arrows, boids, delta_time)  # отрисовка стрелок
    canvas.update()  # отображение

    # time.sleep(0.05) # строка для проверки того, что игра FPS независимая
    end_time = time.time()  # конец отсчета времени
    delta_time = end_time - start_time
    # delta_time = 0.01


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
