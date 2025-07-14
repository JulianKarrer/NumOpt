

# IMPORTS
import numpy as np
from vispy import scene, app # type: ignore
from vispy.scene import visuals # type: ignore
from vispy.app import run # type: ignore
from scipy.spatial import cKDTree
import casadi as ca
import json

# PARAMETERS
h : float = 0.1         # kernel smoothing length
ρ_0 : float = 1e3       # rest density
k_eos : float = 750.    # stiffness constant for equation of state
nu : float = 0.001      # viscosity 
dt : float = 1e-2       # time step size
ε_thresh:float=1e-3*ρ_0 # error threshold of density invariance (0.1% rest density)

# SETTINGS
FPS : float = 165       # target GUI FPS
box_x : float = 4     # domain extent in x-direction
box_y : float = 4       # domain extent in y-direction
USE_PCISPH : bool = True
RUN_GUI : bool = False
FILENAME = "pcisphdynv2.save" if USE_PCISPH else "optidynv2.save"

# CONSTANTS
r_nb : float = 2*h      # neighbourhood search radius
g : float = -9.81       # gravitational acceleration (y-direction)
r_c : float = 2.0*h     # kernel funciton cutoff radius
m : float = ρ_0*(h**2)  # particle rest mass
EPS : float = 1e-6      # epsilon to avoid division by zero
W_corr:float = 1        # correction factor to ensure normalization 
                        # of kernel, i.e.: ∫∫ W(x) dx = 1 (this is automatically set later)
T : float = 5.0         # total time simulated


# UTILITY FUNCTIONS
def box(xl, xh, yl, yh,s):
    """Helper function for creating a uniform grid of points in a box with given spacing"""
    # create uniform box filled with positions
    xs,ys = np.meshgrid(np.arange(xl, xh, s), np.arange(yl, yh, s))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)

def mag(x_ij:np.ndarray):
    return np.sqrt(np.dot(x_ij, x_ij))

# KERNEL FUNCTIONS
def W(r):
    '''Cubic Spline Kernel Function'''
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    t1 = max(0., (2-q))
    t2 = max(0., (1-q))
    return alpha * (t1**3 - 4*t2**3) * W_corr

def W_ca(r):
    '''Cubic Spline Kernel Function (CasADi firendly version)'''
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    t1 = ca.fmax(0., (2-q))
    t2 = ca.fmax(0., (1-q))
    return alpha * (t1**3 - 4*t2**3) * W_corr

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

def dW_ca(dx, dy):
    '''Gradient of the Cubic Spline Kernel Function (CasADi firendly version)'''
    r = ca.sqrt(dx*dx + dy*dy)
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    t1 = ca.fmax(2-q, 0)
    t2 = ca.fmax(1-q, 0)
    factor = alpha / (h * r) * (-3*t1**2 + 12*t2**2)
    return (factor * dx, factor * dy)


# COMPUTE APPROXIMATE "OPTIMAL" STIFFNESS COEFFICIENT
# For the ideal gas equation as an equation of state, this corresponds
# to δ in the paper:
# Predictive-Corrective Incompressible SPH [Solenthaler and Pajarola]
# https://www.ifi.uzh.ch/dam/jcr:ffffffff-daa5-74d6-0000-00005a4f5c99/pcisph.pdf
# This can be used as a guess for relating pressure and density errors as
# p = δ * Δρ
xs,ys = np.meshgrid(np.linspace(-2*h, 2*h, 5), np.linspace(-2*h, 2*h, 5))
xs_perfect = np.stack([xs.ravel(), ys.ravel()], axis=1)
β = dt**2 * m**2 * 2/(ρ_0**2)
squared_grad_sum = (ΣdW := np.sum([dW(xs_perfect[j]) for j in range(len(xs_perfect))], axis=0)).dot(ΣdW)
grad_squared_sum = sum([(grad:= dW(xs_perfect[j])).dot(grad) for j in range(len(xs_perfect))])
δ_pcisph = -1/(β * (-squared_grad_sum -grad_squared_sum))

# compute kernel correction factor to ensure ∫∫ W(x) dx = 1
# and that the density is rest density if sampled on a uniform grid
ρ_measured = 0.
for j in range(25):
    ρ_measured += W(mag(xs_perfect[j])) * m
W_corr = ρ_0/ρ_measured


# INITIALIZE FIELD QUANTITIES
x = box(0, box_x/2+h, 0, box_y/2,h)                     # positions x
x += np.random.normal(scale=1e-5, size=x.shape)     # add miniscule jitter to x_0
b = np.vstack([                                     # boundary particles
    box(-3*h,       -h,        0,       box_y+1,   h),   # - left
    box(box_x+h,    box_x+4*h, 0,       box_y+1,   h),   # - right
    box(-3*h,       box_x+4*h, -3*h,    -h,        h),   # - bottom
])
tree_b = cKDTree(b)     # KDTree for boundary particles is static
N = x.shape[0]          # total number of particles
v = np.zeros_like(x)    # velocities
v_star = np.zeros_like(x) # intermediate velocities for PCISPH
a = np.zeros_like(x)    # accelerations
rho = np.zeros(N)       # densities
p = np.zeros(N)         # pressures
p_acc = np.zeros(N)     # accumulated PCISPH pressures

print(f"Running with {N} particles")

# ERROR MEASURES
def predicted_compr_error(p,x,v,rho,nbrs,nbrs_b):
    """Computes the per-particle predicted density error given a set of pressures,
    and predicted densities. Returned positive values correspond to compressions."""
    v_pred = []
    for i in range(N):
        # compute pressure acceleration from p_ca
        a_x = 0.
        a_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dw = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            a_x += -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dw[0]
            a_y += -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dw[1]
        for k in nbrs_b[i]:
            dw = dW_ca(x[i][0]-b[k][0], x[i][1]-b[k][1])
            a_x += -m * (p[i]/(rho[i]**2)) * dw[0]
            a_y += -m * (p[i]/(rho[i]**2)) * dw[1]
        # compute predicted velocities from those pressure accelerations
        v_pred += [[v[i][0] + a_x * dt,v[i][1] + a_y * dt]]
    err_pred = []
    for i in range(N):
        # compute predicted density from predicted positions
        rho_i = rho[i] 
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            v_ij_x = v_pred[i][0]-v_pred[j][0]
            v_ij_y = v_pred[i][1]-v_pred[j][1]
            rho_i += dt * m * (dwx*v_ij_x + dwy*v_ij_y)
        for k in nbrs_b[i]:
            dw = dW_ca(x[i][0]-b[k][0], x[i][1]-b[k][1])
            rho_i += dt * m * (dw[0]*v_pred[i][0] + dw[1]*v_pred[i][1])
        # compute error from predicted densities
        err_pred += [rho_i - ρ_0]
    return ca.vertcat(*err_pred) 

def predicted_compr_error_exact(p,x,v,rho,nbrs,nbrs_b):
    """Computes the per-particle predicted density error given a set of pressures,
    and predicted densities. Returned positive values correspond to compressions."""
    x_pred = []
    for i in range(N):
        # compute pressure acceleration from pressures
        # a_prs_i = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij  -Σ_k m_k (p_i/ρ_i²) ∇W_ik
        a_x = 0.
        a_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            a_x += -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwx
            a_y += -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwy
        for k in nbrs_b[i]:
            dw = dW(x[i]-b[k])
            a_x += -m * (p[i]/(rho[i]**2)) * dw[0]
            a_y += -m * (p[i]/(rho[i]**2)) * dw[1]
        # compute predicted velocities from those pressure accelerations
        v_x = (v[i][0] + a_x * dt)
        v_y = (v[i][1] + a_y * dt)
        x_pred += [[x[i][0] + v_x*dt, x[i][1] + v_y*dt]]
    err_pred = []
    for i in range(N):
        # compute predicted density from predicted positions
        rho_i = 0.
        for j in nbrs[i]:
            rho_i += m * W_ca(ca.sqrt((x_pred[i][0] - x_pred[j][0])**2 + (x_pred[i][1] - x_pred[j][1])**2))
        for k in nbrs_b[i]:
            rho_i += m * W_ca(ca.sqrt((x_pred[i][0] - b[k][0])**2 + (x_pred[i][1] - b[k][1])**2))
        # compute error from predicted densities
        err_pred += [rho_i - ρ_0]
    return ca.vertcat(*err_pred) 


# QUALITY MEASURES
def int_nabla_p_squared(p, x, nbrs, nbrs_b):
    """Compute the integral of the squared norm of pressure gradients.

    ∫ |∇p|² ≅ | -m_i ρ_i a_prs_i |² 
    where a_prs_i is the SPH discretization of the pressure acceleration.
    """
    res = 0.
    for i in range(N):
        nabla_p_x = 0.
        nabla_p_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwy
            # non-symmetric SPH gradient discretization would be:
            # nabla_p_x += p[j] * m/rho[j] * dw[0]
            # nabla_p_y += p[j] * m/rho[j] * dw[1]
        for k in nbrs_b[i]:
            dwx, dwy = dW_ca(x[i][0]-b[k][0], x[i][1]-b[k][1])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2)) * dwy
        res += (nabla_p_x*nabla_p_x + nabla_p_y * nabla_p_y)
    return res


def compute_action(p, x, nbrs, nbrs_b):
    """Compute half the sum of the squared norms of pressure gradients:
    - 1/2  Σ_i  |∇p|² where ∇p is the SPH discretization of the pressure gradient.

    See equation 2 of `The principle of minimum pressure gradient:
    An alternative basis for physics-informed
    learning of incompressible fluid mechanics` by Alhussein and Daqaq.

    - Note that external forces are already subtracted by only considering pressure accelerations.
    - Further Note that here, m_j = m_i = m
    """
    res = 0.
    for i in range(N):
        # compute pressure accelerations
        nabla_p_x = 0.
        nabla_p_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwy
        for k in nbrs_b[i]:
            dw = dW(x[i]-b[k])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2)) * dw[0]
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2)) * dw[1]
        # sum over 1/2 the squared norm of the pressure gradient
        res += 0.5 *(nabla_p_x*nabla_p_x + nabla_p_y * nabla_p_y)
    return res 

def dirichlet_energy(p, x, err, nbrs, nbrs_b, compression_only=False):
    """Compute the Dirichlet energy of the pressure field.
    The Dirichlet energy in this case reads: 
    - E = ∫ 1/2 · <∇p>² - p (ρ* - ρ_0)/Δt²
    The minimizer of which for constant pressures at the boundaries solves the pressure Poisson equation:
    - ∇²p + (ρ* - ρ_0)/Δt² = 0
    Note that (ρ* - ρ_0) is the density error term.
    """
    res = 0.
    for i in range(N):
        nabla_p_x = 0.
        nabla_p_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW_ca(x[i][0]-x[j][0], x[i][1]-x[j][1])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwy
        for k in nbrs_b[i]:
            dwx, dwy = dW_ca(x[i][0]-b[k][0], x[i][1]-b[k][1])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2)) * dwy
        if compression_only:
            # only penzlizing compressive errors makes this objective non-smooth due to the use of fmax
            # this results in diverging iterates for inexact err and nonsense results for exact err
            res += 0.5*(nabla_p_x*nabla_p_x + nabla_p_y * nabla_p_y) - p[i]*(ca.fmax(err[i], 0.)/dt**2)
        else:
            res += 0.5*(nabla_p_x*nabla_p_x + nabla_p_y * nabla_p_y) - p[i]*(err[i]/dt**2)
    return res

# SPH FUNCTIONS
def update_neighbour_sets(x):
    """Return an updated index set of fluid and boundary neighbour indices,
    given a vector of fluid particle positions `x`"""
    # KDTree for fixed radius neighbour query
    tree_x = cKDTree(x) 
    # find indices of neighbouring fluid and boundary particles
    nbrs = tree_x.query_ball_point(x, r_nb)
    nbrs_b = tree_b.query_ball_point(x, r_nb)
    return nbrs, nbrs_b


# GUI SETUP
if RUN_GUI:
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


# BUFFERS FOR RESULTS AND WARMSTART
history = {
    "xs" : [],
    "er" : [],
    "ps" : [],
    "ts" : [],
    "ay" : [],
    "rhos" : [],
    "dt": dt,
    "box_x": box_x,
    "box_y": box_y,
    "h": h,
    "rho_0": ρ_0,
}
lam_g = None # store lambda g values for warm starts


# PRESSURE SOLVE
def pressure_solve_opti(x, v, rho, nbrs, nbrs_b, p, two_stage=False):
    global lam_g
    opti = ca.Opti()
    options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver("ipopt", options)
    p_ca = opti.variable(N)
    ε = opti.variable()

    # err represents the per-particle predicted density error
    # where positive values correspond to compressions.
    err = predicted_compr_error(p_ca, x, v, rho, nbrs, nbrs_b)
    # err = predicted_compr_error_exact(p_ca, x, v, rho, nbrs, nbrs_b)

    if two_stage:
        # find smallest feasible error threshold
        opti.minimize(ε)
        opti.subject_to( err <= ε )
        opti.subject_to( ε >= 0 )
        opti.subject_to( p_ca >= 0 )
        opti.set_initial(ε, 1e-3)
        sol1 = opti.solve()
        ε_star = sol1.value(ε)
        p = sol1.value(p_ca)
    else:
        ε_star = ε_thresh
    print("min feasible error:", ε_star)

    # # optimization problem formulation
    # opti.minimize(ca.sumsqr(err) + 1e-2*ca.sumsqr(p_ca*(1/δ_pcisph)))
    # opti.subject_to( p_ca >= 0 )

    # # minimize sum of squares of pressure
    # opti.minimize(ca.sumsqr(p_ca*(1/δ_pcisph)))
    # opti.subject_to( err <= 0.1e-2 )
    # opti.subject_to( p_ca >= 0 )

    # minimize pressure gradients # for y=4,x=0.4 infeasible with exact error
    opti.minimize(compute_action(p_ca, x, nbrs, nbrs_b))
    opti.subject_to( err <= ε_star )
    opti.subject_to( p_ca >= 0 )

    # # no errors allowed
    # opti.minimize(int_nabla_p_squared(p_ca, x, nbrs, nbrs_b))
    # opti.subject_to( err <= 0 )
    # opti.subject_to( p_ca >= 0 )

    # # true dirichlet energy minimization
    # opti.minimize(dirichlet_energy(p_ca, x, err, nbrs, nbrs_b))
    # opti.subject_to( p_ca >= 0 )

    # warm-start primal and dual values from previous solve
    opti.set_initial(p_ca, p)
    if not (lam_g is None):
        opti.set_initial(opti.lam_g, lam_g)

    # solve the problem and grab the resulting pressures!
    sol = opti.solve()
    p = sol.value(p_ca)
    lam_g = sol.value(opti.lam_g)
    err_vals = sol.value(err)

    print("max error:", np.max(sol.value(err)))
    return p, err_vals

t = 0
# MAIN FUNCTION
def main(event=None):
    global x,v,a,rho,p,history,lam_g,t
    # NEIGHBOUR SEARCH
    nbrs, nbrs_b = update_neighbour_sets(x)

    # UPDATE DENSITY
    rho = np.fromiter(( 
        # ρ_i = Σ_j W_ij m_j
          sum([W(mag(x[i]-x[j])) * m for j in nbrs[i]]) 
        + sum([W(mag(x[i]-b[k])) * m for k in nbrs_b[i]])
        for i in range(N)), float, N)
    
    print("avg density", np.average(rho), "max density", np.max(rho))
    
    # COMPUTE NON-PRESSURE ACCELERATIONS
    for i in range(N):
        a[i] = np.array([0., g]) + sum([
            nu * 8. * m * ((v[i]-v[j]).dot((x_ij := x[i]-x[j])) / (x_ij.dot(x_ij) + 0.01*h**2)) * dW(x_ij) / rho[i]
            for j in nbrs[i]
        ])
    
    # INTEGRATE A -> V
    v += dt*a
    
    if not USE_PCISPH:
        # PRESSURE SOLVE USING CASADI
        if lam_g is None:
            # without warm-start values to use, use equation of state as initial guess
            p = np.fromiter(( δ_pcisph * max(0, rho[i]-ρ_0) for i in range(N)), float)

        p, err_vals = pressure_solve_opti(x,v,rho,nbrs,nbrs_b,p)
        print("avg pressure", np.average(p))
    else:
        # COMPUTE PRESSURE GUESS from equation of state (ideal gas, δ_pcisph)
        v_star = np.copy(v)
        update_rho_star = lambda: np.fromiter(( 
            rho[i] + 
            sum([
                dt * m * (dW(x[i]-x[j])).dot(v_star[i]-v_star[j])
                for j in nbrs[i]
            ]) + sum([
                dt * m * (dW(x[i]-b[k])).dot(v_star[i])
                for k in nbrs_b[i]
            ])
            for i in range(N)), float)
        rho_star = update_rho_star()
        p_acc = np.zeros_like(p)

        while True:
            err_vals = (rho_star - ρ_0)
            if np.max(err_vals) <= ε_thresh:
                break;
            # compute predicted density given current predicted velocities
            rho_star = update_rho_star()
            
            # compute pressures using PCISPH state equation
            p = np.fromiter(( δ_pcisph * max(0, rho_star[i]-ρ_0) for i in range(N)), float)
            # accumulate total pressure values for each particle
            p_acc += p
            
            # compute pressure accelerations and integrate to update predicted velocities
            for i in range(N):
                a[i] = sum([
                    -m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dW(x[i]-x[j])
                    for j in nbrs[i]
                ]) + sum([
                    -m * (p[i]/(rho[i]**2)) * dW(x[i]-b[k])
                    for k in nbrs_b[i]
                ])
            v_star += dt*a
        # finally, use the accumulated pressure values as output
        p = p_acc

    # APPLY PRESSURE FORCES
    for i in range(N):
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
        border = 5                # how far outside before these conditions apply
        if x[i,0] < -border:        # left wall
            x[i,0] = -border; v[i,0] *= gamma
        if x[i,0] > box_x+border:   # right wall
            x[i,0] = box_x+border; v[i,0] *= gamma
        if x[i,1] < -border:        # bottom wall
            x[i,1] = -border; v[i,1] *= gamma
        if x[i,1] > box_y+border:   # top wall
            x[i,1] = box_y+border; v[i,1] *= gamma

    # save time step to history
    t+=1
    history["xs"] += [x.tolist()]
    history["ps"] += [p.tolist()]
    history["rhos"] += [rho.tolist()]
    history["er"] += [err_vals.tolist()] # type: ignore
    history["ts"] += [(t)*dt]
    history["ay"] += [[a_i[1] for a_i in a]]

    # update gui 
    if RUN_GUI:
        scatter_x.set_data(x, face_color='red', size=5)


if RUN_GUI:
    # RUN GUI
    timer = app.Timer()
    timer.connect(main)
    timer.start(1./FPS)
    if __name__ == '__main__':
        canvas.app.run() # type: ignore
else:
    while t*dt <= T:
        print(f"STEP: t={t}/{int(T/dt)}")
        main()
    with open(FILENAME, "w") as f:
        f.write(json.dumps(history))
