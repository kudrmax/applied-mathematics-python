import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
import time

from my_funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

W, H = 640, 480  # размеры экрана
N = 500  # кол-во птиц
ratio = W / H
w, h = ratio, 1
field_size = (w, h)
global delta_time
delta_time = 0.0

perseption = 1 / 20  #
vrange = (0, 0.2)  # ограничения на скорости

#                    c      a    s      w
coeffs = np.array([300.0, 100.0, 100, 0.03])  # коэффициенты взаисодейлствя

boids = np.zeros((N, 6), dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, ax, ay]
init_boids(boids, field_size, vrange=vrange)  # создаем птиц
# boids[:, 4:6] = 0.0  # задаем птицам ускорения

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

    flocking(boids, perseption, coeffs, field_size, vrange)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, delta_time, vrange)  # пересчет скоростей на основе ускорений
    paint_arrows(arrows, boids, delta_time)  # отрисовка стрелок
    canvas.update()  # отображение

    # time.sleep(0.05) # строка для проверки того, что игра FPS независимая
    end_time = time.time()  # конец отсчета времени
    delta_time = end_time - start_time


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()