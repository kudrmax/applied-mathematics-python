import numpy as np
from vispy import app, scene
from vispy.geometry import Rect

# from funcs import init_boids
# from funcs import directions
from funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

w, h = 640, 480  # размеры экрана
N = 500  # кол-во птиц
dt = 0.001
asp = w / h  # мультиплаер
perseption = 1 / 20  #
vrange = (0, 0.1)  # ограничения на скорости

coeffs = np.array([0.05, 0.02, 4, 0.03])  # коэффициенты взаисодейлствя

boids = np.zeros((N, 6), dtype=np.float64)  # одна строка матрица <-> одна птица с параметрами [x, y, vx, vy, ax, ay]
init_boids(boids, asp, vrange=vrange)  # создаем птиц
boids[:, 4:6] = 0.1  # задаем птицам ускорения

canvas = scene.SceneCanvas(show=True, size=(w, h))  # создаем сцену
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)


def paint_arrows(arrows):
    arrows.set_data(arrows=directions(boids, dt))  # отрисовка стрелок


def update(event):
    flocking(boids, perseption, coeffs, asp, vrange)  # пересчет ускорений (взаимодействие между птицами)
    propagate(boids, dt, vrange)  # пересчет скоростей на основе ускорений
    paint_arrows(arrows)  # отрисовка стрелок
    canvas.update()  # отображение


if __name__ == '__main__':  # @todo это
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
