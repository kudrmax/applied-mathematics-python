import numpy as np
from vispy import app, scene
from vispy.geometry import Rect

# from funcs import init_boids
# from funcs import directions
from funcs import *

app.use_app('pyqt5')
# app.use_app('pyglet')

w, h = 640, 480
N = 500
dt = 0.001
asp = w / h
perseption = 1 / 20
vrange = (0, 0.1)

coeffs = np.array([0.05, 0.02, 4, 0.03])

boids = np.zeros((N, 6), dtype=np.float64)  # x, y, vx, vy, ax, ay
init_boids(boids, asp, vrange=vrange) # тут ускорения
boids[:, 4:6] = 0.1

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)


def update(event):
    propagate(boids, dt, vrange)
    arrows.set_data(arrows=directions(boids, dt))
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
