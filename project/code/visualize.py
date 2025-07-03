# type: ignore
import taichi as ti
import taichi.math as tm 
import numpy as np
import matplotlib.pyplot as plt
import json

ti.init(arch=ti.gpu, debug=True)

# SETTINGS
RES : int = 300     # pixels per unit length
H : float = 0.1     # particle spacing
AROUND : int = 100

# LOAD FILE
with open("minAction_approx.save") as f:
    data = json.load(f)

# COMPUTE CONSTANTS
STEPS : int = len(data["xs"])
N : int = len(data["xs"][0])
NX : int = int(0.3*RES) + 2*AROUND
NY : int = int(5.0*RES) + 2*AROUND
MAX_P = max([max(data["ps"][i]) for i in range(STEPS)])
print(NX,NY,MAX_P)

# DEFINE TAICHI FIELDS
x = ti.Vector.field(n=2, dtype=ti.f32, shape=(N,), needs_grad=True) # positions
p = ti.field(dtype=ti.f32, shape=(N,))                              # pressures
pixels = ti.Vector.field(n=4, dtype=ti.f32, shape=(NX,NY))          # output pixels
max_p = ti.field(dtype=ti.f32, shape=())                            # normalize pressure

# MAP SCALARS TO COLOURS
@ti.func 
def colour_map(col:ti.float32) -> ti.types.vector(3, ti.float32): 
    x = 1.0-col
    # https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/IDL_CB-Spectral.frag
    r:ti.float32 = 0.0 # type: ignore
    g:ti.float32 = 0.0 # type: ignore
    b:ti.float32 = 0.0 # type: ignore
    # RED
    if (x < 0.09752005946586478):
        r = 5.63203907203907E+02 * x + 1.57952380952381E+02
    elif (x < 0.2005235116443438):
        r = 3.02650769230760E+02 * x + 1.83361538461540E+02
    elif (x < 0.2974133397506856):
        r = 9.21045429665647E+01 * x + 2.25581007115501E+02
    elif (x < 0.5003919130598823):
        r = 9.84288115246108E+00 * x + 2.50046722689075E+02
    elif (x < 0.5989021956920624):
        r = -2.48619704433547E+02 * x + 3.79379310344861E+02
    elif (x < 0.902860552072525):
        r = ((2.76764884219295E+03 * x - 6.08393126459837E+03) * x + 3.80008072407485E+03) * x - 4.57725185424742E+02
    else:
        r = 4.27603478260530E+02 * x - 3.35293188405479E+02
    # GREEN
    if (x < 0.09785836420571035):
        g = 6.23754529914529E+02 * x + 7.26495726495790E-01
    elif (x < 0.2034012006283468):
        g = 4.60453201970444E+02 * x + 1.67068965517242E+01
    elif (x < 0.302409765476316):
        g = 6.61789401709441E+02 * x - 2.42451282051364E+01
    elif (x < 0.4005965758690823):
        g = 4.82379130434784E+02 * x + 3.00102898550747E+01
    elif (x < 0.4981907026473237):
        g = 3.24710622710631E+02 * x + 9.31717541717582E+01
    elif (x < 0.6064345916502067):
        g = -9.64699507389807E+01 * x + 3.03000000000023E+02
    elif (x < 0.7987472620841592):
        g = -2.54022986425337E+02 * x + 3.98545610859729E+02
    else:
        g = -5.71281628959223E+02 * x + 6.51955082956207E+02
    # BLUE
    if (x < 0.0997359608740309):
        b = 1.26522393162393E+02 * x + 6.65042735042735E+01;
    elif (x < 0.1983790695667267):
        b = -1.22037851037851E+02 * x + 9.12946682946686E+01;
    elif (x < 0.4997643530368805):
        b = (5.39336225400169E+02 * x + 3.55461986381562E+01) * x + 3.88081126069087E+01;
    elif (x < 0.6025972254407099):
        b = -3.79294261294313E+02 * x + 3.80837606837633E+02;
    elif (x < 0.6990141388105746):
        b = 1.15990231990252E+02 * x + 8.23805453805459E+01;
    elif (x < 0.8032653181119567):
        b = 1.68464957265204E+01 * x + 1.51683418803401E+02;
    elif (x < 0.9035796343050095):
        b = 2.40199023199020E+02 * x - 2.77279202279061E+01;
    else:
        b = -2.78813846153774E+02 * x + 4.41241538461485E+02;
    return tm.clamp(ti.Vector([r, g, b]) / 255.0, 0.0, 1.0)

@ti.func
def W(r:ti.float32):
    '''Cubic Spline Kernel Function'''
    alpha = ti.static (5.0 / (14.0 * 3.1415926535 * H * H))
    q = r / H;
    t1 = max(0., (2-q))
    t2 = max(0., (1-q))
    return alpha * (t1**3 - 4*t2**3)

@ti.kernel
def display(f:ti.template()):
    for xi, yi in ti.ndrange(NX, NY):
        x_coord = (xi-AROUND)/RES
        y_coord = (yi-AROUND)/RES

        mag = 0.
        for i in range(N):
            dx = x[i].x-x_coord
            dy = x[i].y-y_coord
            # compute SPH-interpolation of the pressure at x,y
            # here: assume rest density
            mag += (H**2) * f[i] * W(tm.sqrt(dx*dx + dy*dy))
            # otherwise:
            # magnitude += (M/rho[i]) * p[i] * W(tm.sqrt(dx*dx+dy+dy))
        # normalize to [0;1] for colour mapping
        mag /= MAX_P

        # map the result to a colour
        c = colour_map(mag)
        pixels[xi,yi] = tm.mix(
            ti.Vector([0,0,0,1]), 
            ti.Vector([c.x,c.y,c.z,mag]),
            mag**0.2
        )

# MAIN LOOP
gui = ti.GUI("Optimal SPH", res=(NX, NY), fast_gui = True)
for i in range(STEPS):
    max_p[None] = max(data["ps"][i])
    p_np = np.array(data["ps"][i]).reshape(N)
    x.from_numpy(np.array(data["xs"][i]))
    p.from_numpy(p_np)
    display(p)
    gui.set_image(pixels)
    gui.show()

