from typing import Any
import numpy as np
from vispy import scene, app
from vispy.scene import visuals
from vispy.app import run
from scipy.spatial import cKDTree

# PARAMETERS
h: float = 0.1           # smoothing length
rho0: float = 1.0        # rest density
k: float = 500.          # gas stiffness constant
nu: float = 0.001        # viscosity coefficient
Dt: float = 1e-3         # time step

# SETTINGS
FPS: float = 160
box_x: float = 1.0
box_y: float = 0.5

# CONSTANTS
g: float = -9.81         # gravity acceleration
dt: float = Dt
r_c: float = 2.0 * h      # kernel cutoff radius
m: float = rho0 * (h * h) # particle mass (2D) 
EPS: float = 1e-6         # small epsilon to avoid divide-by-zero

# KERNEL FUNCTIONS
def W(r: float) -> float:
    """Cubic spline smoothing kernel"""
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h
    if q < 1.0:
        return alpha * ( (2 - q)**3 - 4 * (1 - q)**3 )
    elif q < 2.0:
        return alpha * (2 - q)**3
    else:
        return 0.0


def gradW(r_vec: np.ndarray) -> np.ndarray:
    """Gradient of cubic spline kernel"""
    r = np.linalg.norm(r_vec)
    if r < EPS:
        return np.zeros_like(r_vec)
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h
    if q < 1.0:
        val = alpha / (h * r) * ( -3 * (2 - q)**2 + 12 * (1 - q)**2 )
    elif q < 2.0:
        val = -3 * alpha * (2 - q)**2 / (h * r)
    else:
        return np.zeros_like(r_vec)
    return val * r_vec

# INITIALIZE PARTICLES
# uniform grid in box [0,box_x] x [0,box_y]
def make_box(xl, xh, yl, yh, spacing):
    xs, ys = np.meshgrid(np.arange(xl, xh, spacing), np.arange(yl, yh, spacing))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    # pts += np.random.normal(scale=1e-4, size=pts.shape)
    return pts

x = make_box(0, box_x + h, 0, box_y, h)
N = x.shape[0]
v = np.zeros_like(x)
a = np.zeros_like(x)

rho = np.zeros(N)
p = np.zeros(N)

# GUI SETUP
canvas = scene.SceneCanvas(title="SPH-Fluid", keys='interactive', show=True,
                           bgcolor='white', size=(400, 400 * box_y/box_x))
view = canvas.central_widget.add_view()
scatter = visuals.Markers() # type: ignore
scatter.set_data(x, face_color='red', size=5)
view.add(scatter)
view.camera = 'panzoom'
view.camera.set_range(x=(-1, box_x+1), y=(-1, box_y+2)) # type: ignore

# MAIN UPDATE LOOP
def main(event=None):
    global x, v, a, rho, p
    # neighbor search
    tree = cKDTree(x)
    nbrs = tree.query_ball_point(x, r_c)

    # compute density and pressure
    for i in range(N):
        rho_i = 0.0
        for j in nbrs[i]:
            r = np.linalg.norm(x[i] - x[j])
            rho_i += m * W(r)
        rho[i] = rho_i
        p[i] = k * max(rho_i - rho0, 0)

    # compute forces
    a.fill(0.0)
    for i in range(N):
        f_pressure = np.zeros(2)
        f_visc = np.zeros(2)
        for j in nbrs[i]:
            if i == j: continue
            rij = x[i] - x[j]
            # pressure term
            term = (p[i]/(rho[i]**2 + EPS) + p[j]/(rho[j]**2 + EPS))
            f_pressure += -m * term * gradW(rij)
            # viscosity term (Monaghan)
            vij = v[j] - v[i]
            r = np.linalg.norm(rij)
            if r > EPS:
                f_visc += nu * m * (vij.dot(rij) / (r**2 + 0.01*h**2)) * gradW(rij)
        # gravity
        f_gravity = np.array([0.0, g]) * rho[i]
        # total acceleration a = (f_pressure + f_visc + f_gravity) / rho
        a[i] = (f_pressure + f_visc + f_gravity) / rho[i]

    # integrate (symplectic Euler)
    v += dt * a
    x += dt * v

    # simple boundary conditions: reflect at walls
    for i in range(N):
        if x[i,0] < 0:
            x[i,0] = 0; v[i,0] *= -0.5
        if x[i,0] > box_x:
            x[i,0] = box_x; v[i,0] *= -0.5
        if x[i,1] < 0:
            x[i,1] = 0; v[i,1] *= -0.5
        if x[i,1] > box_y:
            x[i,1] = box_y; v[i,1] *= -0.5

    # update visualization
    scatter.set_data(x, face_color='red', size=5)

# RUN
if __name__ == '__main__':
    timer = app.Timer(1.0 / FPS, connect=main, start=True)
    canvas.app.run()
