import numpy as np
from vispy import app, scene
from vispy.geometry import Rect

from my_funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

W, H = 640, 480  # размеры экрана
N = 500  # кол-во птиц
dt = 0.001
ratio = W / H
w, h = ratio, 1
field_size = (w, h)

perseption = 1 / 20  #
vrange = (0, 1.0)  # ограничения на скорости

coeffs = np.array([1.0, 0.02, 4, 0.03])  # коэффициенты взаисодейлствя

boids = np.zeros((N, 6), dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, ax, ay]
init_boids(boids, field_size, vrange=vrange)  # создаем птиц
boids[:, 4:6] = 0.0  # задаем птицам ускорения

canvas = scene.SceneCanvas(show=True, size=(W, H))  # создаем сцену
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, w, h))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)


def paint_arrows(arrows):
    arrows.set_data(arrows=directions(boids, dt))  # отрисовка стрелок


def update(event):
    flocking(boids, perseption, coeffs, ratio, vrange)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, dt, vrange)  # пересчет скоростей на основе ускорений
    paint_arrows(arrows)  # отрисовка стрелок
    canvas.update()  # отображение


if __name__ == '__main__':  # @todo это
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
