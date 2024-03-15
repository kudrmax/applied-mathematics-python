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

@ti.func
def rotate(angle):
    c = ti.cos(angle)
    s = ti.sin(angle)
    return ts.mat([c, -s], [s, c])

@ti.kernel
def render(t: ti.f32):
    col0 = ts.vec3(255., 131., 137.) / 255.0

    for fragCoord in ti.grouped(pixels):
        uv = (fragCoord - 0.5 * resf) / resf[1]
        uv *= 10 * ti.sin(t * 10)
        uv = rotate(0.1 * t * 50) @ uv
        frac_uv = ts.fract(uv) - 0.5

        grid = ts.smoothstep(ti.abs(frac_uv).max(), 0.4, 0.5)
        # grid_x = ti.abs(frac_uv.x)
        col = ts.vec3(1.0, 0., 0.)
        col = ts.mix(
            col,
            ts.vec3(0., 1., 0.),
            grid
        )
        # col += ts.vec3(1., 0., 1.) * grid
        # col.gb = uv
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
