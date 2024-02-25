import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
import time

from my_funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

W, H = 640, 480  # размеры экрана
N = 100  # кол-во птиц
ratio = W / H
w, h = ratio, 1
field_size = np.array([w, h])
global delta_time
delta_time = 0.0

radius = field_size[0] / 20  #
vrange = (0, 1000.1)  # ограничения на скорости

#                    c      a    s      w
# coeffs = np.array([0.2, 5.5, .3, 0.0])  # коэффициенты взаисодейлствя
# coeffs = np.array([0.01, .41, .2, 0.0])  # коэффициенты взаисодейлствя # хорошие
# coeffs = np.array([0.05, .3, .2, 0.0])  # коэффициенты взаисодейлствя # хорошие 2
coeffs = np.array([0.005, .05, .02, 0.0])  # коэффициенты взаисодейлствя # до numba
# coeffs = np.array([0.5, .05, .02, 0.0])  # коэффициенты взаисодейлствя
coeffs = np.array([.0, 100.0, 1.0, 0.0])  # коэффициенты взаисодейлствя
coeffs = np.array([1.0, 1.0, 1.0, 0.0])  # коэффициенты взаисодейлствя

boids = np.zeros((N, 6), dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, dvx, dvy]
init_boids(boids, field_size, vrange=vrange)  # создаем птиц

canvas = scene.SceneCanvas(show=True, size=(W, H))  # создаем сцену
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, w, h))
arrows = scene.Arrow(arrows=directions(boids, delta_time),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)


def update(event):
    global delta_time
    start_time = time.time()  # начало отсчета времени

    flocking(boids, radius, coeffs, field_size, vrange, delta_time)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, delta_time, vrange)  # пересчет скоростей на основе ускорений
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
