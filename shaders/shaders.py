import taichi as ti
import taichi_glsl as ts
import time

ti.init(arch=ti.gpu)

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
resf = ts.vec2(float(w), float(h))
# layers = 5

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


@ti.kernel
def render(t: ti.f32):
    col0 = ts.vec3(255., 131., 137.) / 255.0

    for fragCoord in ti.grouped(pixels):
        uv = (fragCoord - 0.5 * resf) / resf[1]

        col = ts.vec3(0.)
        col.gb = uv
        # col.gb = uv + 0.5

        pixels[fragCoord] = ts.clamp(col, 0., 1.)


if __name__ == '__main__':

    gui = ti.GUI('Titile', res=res, fast_gui=True)
    start = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        t = time.time() - start
        render(t)
        gui.set_image(pixels)
        gui.show()

    gui.close()
