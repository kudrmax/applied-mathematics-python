import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
import time

from my_funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

W, H = 640, 480  # размеры экрана
N = 500  # кол-во птиц
dt = 0.001
ratio = W / H
w, h = ratio, 1
field_size = (w, h)
global m_delta_time
m_delta_time = 0.0

perseption = 1 / 20  #
vrange = (0, 0.1)  # ограничения на скорости

coeffs = np.array([3.0, 0.02, 4, 0.03])  # коэффициенты взаисодейлствя

boids = np.zeros((N, 6), dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, ax, ay]
init_boids(boids, field_size, vrange=vrange)  # создаем птиц
boids[:, 4:6] = 10.0  # задаем птицам ускорения

canvas = scene.SceneCanvas(show=True, size=(W, H))  # создаем сцену
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, w, h))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)


def paint_arrows(arrows, m_delta_time):
    arrows.set_data(arrows=directions(boids, m_delta_time))  # отрисовка стрелок


def update(event):
    global m_delta_time
    start_time = time.time()

    flocking(boids, perseption, coeffs, ratio, vrange)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, m_delta_time, vrange)  # пересчет скоростей на основе ускорений
    paint_arrows(arrows, m_delta_time)  # отрисовка стрелок
    canvas.update()  # отображение

    time.sleep(0.05)
    end_time = time.time()

    m_delta_time = end_time - start_time


if __name__ == '__main__':  # @todo это
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
