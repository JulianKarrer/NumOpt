import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')

# create empty optimization problem
opti = ca.Opti()

# define variables
x = opti.variable()
y = opti.variable()

# define objective
f = x**2 - 2*x + y**2 + y

# hand objective to casadi, no minimization done yet
opti.minimize(f)

# define constraints. To include several constraints, just call this
# function several times
opti.subject_to( x >= 1.5)
opti.subject_to( x + y >= 0 )

# define solver
opti.solver('ipopt')                    # Use IPOPT as solver
# opti.solver('sqpmethod')              # Use sqpmethod as solver

# solve optimization problem
sol = opti.solve()

# read and print solution
xopt = sol.value(x)
yopt = sol.value(y)

def plot(xmin,xmax,ymin,ymax,func,feasible,xopt,yopt,res=1000, colour_res=100):
    # create mesh to draw, calculate function values on grid
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys)
    Z = func(X,Y)
    levels = np.linspace(Z.min(), Z.max(), colour_res)
    fig, ax = plt.subplots()
    # create contour
    cs = ax.contourf(X, Y, Z, levels, zorder=1)
    _cbar = fig.colorbar(cs)
    # shade feasible set
    mask = ~feasible(X, Y)
    overlay = np.zeros((res, res, 4)) # create RGBA field
    overlay[mask] = [1, 1, 1, 0.2] # base colour of feasible area shading
    ax.imshow(overlay, extent=(xmin, xmax, ymin, ymax), origin='lower', aspect='auto', zorder=2)
    # draw dot at optimum
    ax.scatter(x=xopt,y=yopt)
    # show plot
    plt.tight_layout()
    plt.show()


print()
if opti.stats()['return_status'] == 'Solve_Succeeded':  # if casadi returns successfully
    print(f"Optimal solution found: x = {xopt}, y = {yopt}")
    plot(-3,3,-3,3, lambda x,y: x**2-2*x+y**2+y, lambda x,y: (x>=1.5) & (x+y>=0), xopt, yopt)
else:
    print("Problem failed")
