

# IMPORTS
import numpy as np
from vispy import scene, app # type: ignore
from vispy.scene import visuals # type: ignore
from vispy.app import run # type: ignore
from scipy.spatial import cKDTree
import casadi as ca

# PARAMETERS
h : float = 0.1         # kernel smoothing length
ρ_0 : float = 1.0       # rest density
k_eos : float = 750.    # stiffness constant for equation of state
nu : float = 0.001      # viscosity 
dt : float = 1e-3       # time step size

# SETTINGS
FPS : float = 165       # target GUI FPS
box_x : float = 0.5     # domain extent in x-direction
box_y : float = 2       # domain extent in y-direction

# CONSTANTS
g : float = -9.81       # gravitational acceleration (y-direction)
r_c : float = 2.0*h     # kernel funciton cutoff radius
m : float = ρ_0*(h**2)  # particle rest mass
EPS : float = 1e-6      # epsilon to avoid division by zero


# UTILITY FUNCTIONS
def box(xl, xh, yl, yh,s):
    """Helper function for creating a uniform grid of points in a box with given spacing"""
    # create uniform box filled with positions
    xs,ys = np.meshgrid(np.arange(xl, xh, s), np.arange(yl, yh, s))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)

def mag(x_ij:np.ndarray):
    return np.sqrt(np.dot(x_ij, x_ij))

# KERNEL FUNCTIONS
def W(r:float) -> float:
    '''Cubic Spline Kernel Function'''
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    if q<1:
        return alpha * ((2-q)**3 - 4*(1-q)**3)
    elif q<2:
        return alpha *  (2-q)**3
    else:
        return 0.

def dW(x_ij:np.ndarray)->np.ndarray:
    '''Gradient of the Cubic Spline Kernel Function'''
    r = mag(x_ij)
    if r < EPS:
        return np.zeros_like(x_ij)
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    if q<1:
        return alpha / (h * r) * (-3*(2-q)**2 + 12*(1-q)**2) * x_ij
    elif q<2:
        return alpha / (h * r) * (-3*(2-q)**2)               * x_ij
    else:
        return np.zeros_like(x_ij)


# INITIALIZE FIELD QUANTITIES
x = box(0, box_x+h, 0, box_y,h)                     # positions x
x += np.random.normal(scale=1e-5, size=x.shape)     # add miniscule jitter to x_0
b = np.vstack([                                     # boundary particles
    box(-3*h,       -h,        0,       box_y+1,   h),   # - left
    box(box_x+h,    box_x+4*h, 0,       box_y+1,   h),   # - right
    box(-3*h,       box_x+4*h, -3*h,    -h,        h),   # - bottom
])
tree_b = cKDTree(b)     # KDTree for boundary particles is static
N = x.shape[0]          # total number of particles
v = np.zeros_like(x)    # velocities
a = np.zeros_like(x)    # accelerations
rho = np.zeros(N)       # densities
p = np.zeros(N)         # pressures



# GUI SETUP
canvas = scene.SceneCanvas(title="OPTI-SPH", keys='interactive', show=True, bgcolor='white', size=(300, 300*(box_y+1)/box_x))
view = canvas.central_widget.add_view()
scatter_x = visuals.Markers() # type: ignore
scatter_x.set_data(x, face_color='red', size=5)
view.add(scatter_x)
scatter_bdy = visuals.Markers() # type: ignore
scatter_bdy.set_data(b, face_color='blue', size=5)
view.add(scatter_bdy)
view.camera = 'panzoom'
view.camera.set_range(x=(-1, box_x+1), y=(-1, box_y+2)) # type: ignore

# MAIN FUNCTION
def main(event=None):
    global x,v,a,rho,p

    # NEIGHBOUR SEARCH
    tree_x = cKDTree(x) # KDTree for fixed radius neighbour query
    nbrs = tree_x.query_ball_point(x, r_c)
    nbrs_b = tree_b.query_ball_point(x, r_c)

    # UPDATE DENSITY
    for i in range(N):
        # ρ_i = Σ_j W_ij m_j
        rho[i] = (
            sum([W(mag(x[i]-x[j])) * m for j in nbrs[i]]) 
            + sum([W(mag(x[i]-b[k])) * m for k in nbrs_b[i]])
        )
    print("avg density", np.average(rho))
    
    # COMPUTE NON-PRESSURE ACCELERATIONS
    for i in range(N):
        a[i] = np.array([0., g]) + sum([
            nu * 8. * m * ((v[i]-v[j]).dot((x_ij := x[i]-x[j])) / (x_ij.dot(x_ij) + 0.01*h**2)) * dW(x_ij) / rho[i]
            for j in nbrs[i]
        ])

    # INTEGRATE A -> V
    v += dt*a

    # UPDATE TO PREDICTED DENSITY
    for i in range(N):
        rho[i] += sum([
            dt * m * (dW(x[i]-x[j])).dot(v[i]-v[j])
            for j in nbrs[i]
        ]) + sum([
            dt * m * (dW(x[i]-b[k])).dot(v[i])
            for k in nbrs_b[i]
        ])

    # COMPUTE PRESSURE
    for i in range(N):
        # p_i = k (ρ_i - ρ_0)_+
        p[i] = k_eos * max(0, rho[i]-ρ_0)
    
    # compute pressures with casadi
    opti = ca.Opti()
    opti.solver('ipopt', {
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'ipopt.print_timing_statistics': 'no',
        'ipopt.file_print_level': 0
    })
    p_ca = opti.variable(N)
    
    err_pred = []
    v_pred = []
    for i in range(N):
        # compute pressure acceleration from p_ca
        a_x = 0.
        a_y = 0.
        for j in nbrs[i]:
            dw = dW(x[i]-x[j])
            a_x += -m * (p_ca[i]/(rho[i]**2) + p_ca[j]/(rho[j]**2)) * dw[0]
            a_y += -m * (p_ca[i]/(rho[i]**2) + p_ca[j]/(rho[j]**2)) * dw[1]
        for k in nbrs_b[i]:
            dw = dW(x[i]-b[k])
            a_x += -m * (p_ca[i]/(rho[i]**2)) * dw[0]
            a_y += -m * (p_ca[i]/(rho[i]**2)) * dw[1]
        # compute predicted velocities from those pressure accelerations
        v_pred += [[v[i][0] + a_x * dt, v[i][1] + a_y * dt]]
    max_compr = 0.
    for i in range(N):
        # compute predicted density from predicted velocities
        rho_i = rho[i] 
        for j in nbrs[i]:
            dw = dW(x[i]-x[j])
            v_ij_x = v_pred[i][0]-v_pred[j][0]
            v_ij_y = v_pred[i][1]-v_pred[j][1]
            rho_i += dt * m * (dw[0]*v_ij_x + dw[1]*v_ij_y)
        for k in nbrs_b[i]:
            dw = dW(x[i]-b[k])
            rho_i += dt * m * (dw[0]*v_pred[i][0] + dw[1]*v_pred[i][1])
        # compute error from predicted densities
        err_pred += [(rho_i-ρ_0) / N]

    # err represents the per-particle predicted density error
    # where positive values correspond to compressions.
    err = ca.vertcat(*err_pred) 

    
    # optimization problem formulation
    # opti.minimize(ca.sumsqr(err) + ca.sumsqr(p_ca*(1/k_eos)))
    # opti.subject_to( err <= 0.01 )
    # opti.subject_to( p_ca >= 0 )
    # opti.set_initial(p_ca, p)

    opti.minimize(ca.sumsqr(p_ca*(1/k_eos)))
    opti.subject_to( err <= 0.01 )
    opti.subject_to( p_ca >= 0 )
    opti.set_initial(p_ca, p)

    # opti.minimize( ca.sumsqr(err) + 0.0001*ca.sumsqr(p_ca) )
    # opti.subject_to( p_ca >= 0 )
    # opti.set_initial(p_ca, p)
    # sol = opti.solve()
    # p = sol.value(p_ca)
    print("avg pressure", np.average(p))

    # COMPUTE PRESSURE ACCELERATIONS
    for i in range(N):
        # ρ_i = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij  -Σ_k m_k (p_i/ρ_i²) ∇W_ik
        a[i] = sum([
            -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dW(x[i]-x[j])
            for j in nbrs[i]
        ]) + sum([
            -m * (p[i]/(rho[i]**2)) * dW(x[i]-b[k])
            for k in nbrs_b[i]
        ])

    # SEMI-IMPLICIT EULER TIME INTEGRATION
    v += dt*a
    x += dt*v 

    # ENFORCE BOUNDARY CONDITIONS
    for i in range(N):
        gamma = 0               # stop the particle
        border = 5              # how far outside before these conditions apply
        if x[i,0] < -border:        # left wall
            x[i,0] = 0; v[i,0] *= gamma
        if x[i,0] > box_x+border:   # right wall
            x[i,0] = box_x; v[i,0] *= gamma
        if x[i,1] < -border:        # bottom wall
            x[i,1] = 0; v[i,1] *= gamma
        if x[i,1] > box_y+border:   # top wall
            x[i,1] = box_y; v[i,1] *= gamma

    # update gui 
    scatter_x.set_data(x, face_color='red', size=5)


# RUN GUI
timer = app.Timer()
timer.connect(main)
timer.start(1./FPS)
if __name__ == '__main__':
    canvas.app.run()
