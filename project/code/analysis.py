import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
import numpy as np




# load all data for conventional and optimal solver
with open("pcisphv2.save") as f:
    data_c = json.load(f)
with open("optiv2.save") as f:
    data_o = json.load(f)

N = len(data_c["xs"][0])
STEPS = len(data_c["xs"])
h = data_c["h"]
ρ_0 = data_c["rho_0"]
m : float = ρ_0*(h**2)
r_nb : float = 2*h
box_x = data_c["box_x"]
box_y = data_c["box_y"]
EPS : float = 1e-6
print(N, h, ρ_0, m, r_nb)


# reconstruct boundaries
def box(xl, xh, yl, yh,s):
    xs,ys = np.meshgrid(np.arange(xl, xh, s), np.arange(yl, yh, s))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)
b = np.vstack([ 
    box(-3*h,       -h,        0,       box_y+1,   h),
    box(box_x+h,    box_x+4*h, 0,       box_y+1,   h),
    box(-3*h,       box_x+4*h, -3*h,    -h,        h),
])
tree_b = cKDTree(b) 


# define kernel functions
W_corr = 1
def W(r):
    '''Cubic Spline Kernel Function'''
    alpha = 5.0 / (14.0 * np.pi * h * h)
    q = r / h;
    t1 = max(0., (2-q))
    t2 = max(0., (1-q))
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

# compute kernel correction factor
def mag(x_ij:np.ndarray):
    return np.sqrt(np.dot(x_ij, x_ij))
xs,ys = np.meshgrid(np.linspace(-2*h, 2*h, 5), np.linspace(-2*h, 2*h, 5))
xs_perfect = np.stack([xs.ravel(), ys.ravel()], axis=1)
ρ_measured = 0.
for j in range(25):
    ρ_measured += W(mag(xs_perfect[j])) * m
W_corr = ρ_0/ρ_measured
def update_neighbour_sets(x):
    """Return an updated index set of fluid and boundary neighbour indices,
    given a vector of fluid particle positions `x`"""
    # KDTree for fixed radius neighbour query
    tree_x = cKDTree(x) 
    # find indices of neighbouring fluid and boundary particles
    nbrs = tree_x.query_ball_point(x, r_nb)
    nbrs_b = tree_b.query_ball_point(x, r_nb)
    return nbrs, nbrs_b

def compute_action(p, x, nbrs, nbrs_b, rho):
    res = 0.
    for i in range(N):
        # compute pressure accelerations
        nabla_p_x = 0.
        nabla_p_y = 0.
        for j in nbrs[i]:
            if i==j: continue
            dwx, dwy = dW(x[i]-x[j])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwx
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2) + p[j]/(rho[j]**2)) * dwy
        for k in nbrs_b[i]:
            dw = dW(x[i]-b[k])
            nabla_p_x += rho[i] * m * (p[i]/(rho[i]**2)) * dw[0]
            nabla_p_y += rho[i] * m * (p[i]/(rho[i]**2)) * dw[1]
        # sum over 1/2 the squared norm of the pressure gradient
        res += 0.5 *(nabla_p_x*nabla_p_x + nabla_p_y * nabla_p_y)
    return res 



get_max = lambda data,key: [max(data[key][i]) for i in range(len(data[key]))]
get_avg = lambda data,key: [sum(data[key][i])/len(data[key][i]) for i in range(len(data[key]))]

def get_action(data,i):
    x = np.array(data["xs"][i])
    p = np.array(data["ps"][i])
    rho = np.array(data["rhos"][i])
    nbrs,nbrs_b = update_neighbour_sets(x)
    return compute_action(p,x,nbrs,nbrs_b,rho)


# plot action
plt.plot(data_c["ts"], [get_action(data_c, i) for i in range(STEPS)], label="PCISPH Action")
plt.plot(data_o["ts"], [get_action(data_o, i) for i in range(STEPS)], label="QP Action")
plt.legend()
plt.show()

# plot densities
plt.plot(data_c["ts"], get_avg(data_c, "rhos"), label="PCISPH ρ avg")
plt.plot(data_c["ts"], get_max(data_c, "rhos"), label="PCISPH ρ max")
plt.plot(data_o["ts"], get_avg(data_o, "rhos"), label="QP ρ avg")
plt.plot(data_o["ts"], get_max(data_o, "rhos"), label="QP ρ max")
plt.legend()
plt.show()

# plot pressures
plt.plot(data_c["ts"], get_avg(data_c, "ps"), label="PCISPH p avg")
plt.plot(data_c["ts"], get_max(data_c, "ps"), label="PCISPH p max")
plt.plot(data_o["ts"], get_avg(data_o, "ps"), label="QP p avg")
plt.plot(data_o["ts"], get_max(data_o, "ps"), label="QP p max")
plt.legend()
plt.show()


print("ts\n",  data_c["ts"] )
print("rhoavgc\n", get_avg(data_c, "rhos"))
print("rhoavgo\n", get_avg(data_o, "rhos"))
print("rhomaxc\n", get_max(data_c, "rhos"))
print("rhomaxo\n", get_max(data_o, "rhos"))

print("pavgc\n", get_avg(data_c, "ps"))
print("pavgo\n", get_avg(data_o, "ps"))
print("pmaxc\n", get_max(data_c, "ps"))
print("pmaxo\n", get_max(data_o, "ps"))


# xs = lambda i,data: [x[1] for x in data["xs"][i]]
# ay = lambda i,data: data["ay"][i]
# N = len(data["ay"])
# fig, ax = plt.subplots()
# scat = ax.scatter([], [])
# ax.set_xlim(min(min(xs(i)) for i in range(N)), max(max(xs(i)) for i in range(N)))
# ax.set_ylim(min(min(ay(i)) for i in range(N)), max(max(ay(i)) for i in range(N)))
# ax.set_xlabel("x")
# ax.set_ylabel("ay")
# def update(i):
#     x = xs(i)
#     y = ay(i)
#     scat.set_offsets(list(zip(x, y)))
#     ax.set_title(f"i = {i}")
#     return scat,
# is_indices = range(N)
# ani = FuncAnimation(fig, update, frames=is_indices, blit=True, repeat=False)
# plt.show()