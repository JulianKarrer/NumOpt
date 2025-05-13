# %%
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


# problem settings
GROUND_CONSTRAINTS = 3
"""
    0: only constraints on two ends of chain
    1: linear ground constraints
    2: convex quadratic constraint
    3: nonconvex quadratic constraint
"""
assert GROUND_CONSTRAINTS in [0, 1, 2, 3], \
    "GROUND_CONSTRAINTS needs to be equal to 0, 1, 2, or 3"


CUSTOM_INITIALIZATION = 1
"""if 0: no custom initialization, 1: one possible initialzation, 2: another"""


# create empty optimization problem
opti = ca.Opti()

# problem parameters
N = 40
mi = 4/N
Di = (70/40)*N
g0 = 9.81
L  = 1

# definition of variables
y = opti.variable(N)
z = opti.variable(N)

# objective function
Vchain = 0
for i in range(N):
   Vchain += g0*mi*z[i]
for i in range(N-1):
    Vchain += 1/2*Di*((y[i]-y[i+1])**2 + (z[i]-z[i+1])**2)


# pass objective to solver
opti.minimize(Vchain)

# constraints
opti.subject_to( y[0] == -2 )
opti.subject_to( z[0] ==  1 )
opti.subject_to( y[N-1] ==  2 )
opti.subject_to( z[N-1] ==  1 )

if GROUND_CONSTRAINTS == 1:
    # linear constraints
    opti.subject_to(       z >= 0.5)
    opti.subject_to( z-0.1*y >= 0.5) 
elif GROUND_CONSTRAINTS == 2:
    # convex quadratic constraints
    opti.subject_to( z >= -0.2 + 0.1*y**2) 
elif GROUND_CONSTRAINTS == 3:
    # non-convex quadratic constraints
    opti.subject_to( z >= -y**2) 


# Setting the solver
opti.solver('ipopt')
    
if CUSTOM_INITIALIZATION == 1:
    opti.set_initial(y, -1)
    opti.set_initial(z, -1)
elif CUSTOM_INITIALIZATION == 2:
    opti.set_initial(y, 1)
    opti.set_initial(z, -1)

sol = opti.solve()

# to see some more info
# disp(sol.stats)

if sol.stats()['return_status'] == 'Solve_Succeeded':
    
    Y = sol.value(y)
    Z = sol.value(z)
    
    fig, ax = plt.subplots()
    ax.plot(Y, Z, '--or')
    ax.plot(-2, 1, 'xg', markersize=10)
    ax.plot(2, 1, 'xg', markersize=10)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.set_title(r"Optimal solution hanging chain")
    
    if GROUND_CONSTRAINTS == 1:
        ax.plot([-2, 2], [0.5, 0.5], ':b')
        ax.plot([-2, 2], [0.3, 0.7], ':b')
    elif GROUND_CONSTRAINTS == 2:
        yi = np.linspace(-2, 2, 400)
        zi = -0.2 + 0.1*yi**2
        #zi = -1.3 + 0.5*yi**2
        ax.plot(yi,zi,':b')
    elif GROUND_CONSTRAINTS == 3:
        yi = np.linspace(-2, 2, 400)
        zi = -yi**2
        ax.plot(yi,zi,':b')
        ax.set_ylim([-1, 2])
    ax.set_xlim([-2, 2])
plt.show()

# %%
