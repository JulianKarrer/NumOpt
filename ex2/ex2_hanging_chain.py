import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

# PARAMETERS
N = 40            # number of masses
m = 4/N           # mass
Di = (70/40)*N    # spring constant
g0 = 9.81         # gravity

def create_opti():
    # create empty optimization problem
    opti = ca.Opti()
    y = opti.variable(N)
    z = opti.variable(N)
    # define the objective
    Vchain = 0
    for i in range(N):
        if i<N-1:
            Vchain += 1/2 * Di * ((y[i] - y[i+1])**2 + (z[i] - z[i+1])**2)
        Vchain += m*g0*z[i]
    # pass the objective to opti
    opti.minimize(Vchain)
    # add equality constraints
    opti.subject_to( y[0] == -2 )
    opti.subject_to( z[0] == 1 )
    opti.subject_to( y[N-1] == 2 )
    opti.subject_to( z[N-1] == 1 )
    return opti, y, z

def plot_solution(title, opti, y, z, init=0, save=""):
    # set solver to ipopt and solve the problem:
    opti.solver('ipopt')
    opti.set_initial(y,init)
    opti.set_initial(z,init)
    # opti.solver('sqpmethod')
    sol = opti.solve()
    # get solution and plot results
    Y = sol.value(y)
    Z = sol.value(z)
    fig, ax = plt.subplots()
    ax.plot(Y, Z, '--or')
    ax.plot(-2, 1, 'xg', markersize=10)
    ax.plot(2, 1, 'xg', markersize=10)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.set_title(title)
    ## show plots
    plt.show()
    if save != "":
        fig.savefig(save+".pdf", dpi=fig.dpi)

# no ground constraints:
opti, y, z = create_opti()
plot_solution(r"Optimal solution hanging chain (without extra constraints)",opti, y, z, save="unconstrained")

# linear ground constraints:
opti, y, z = create_opti()
opti.subject_to( z >= 0.5 )
opti.subject_to (z - 0.1*y >= 0.5)
plot_solution(r"Optimal solution hanging chain (linear ground constraints)", opti, y, z, save="linear")

# convex quadratic ground constraints:
opti, y, z = create_opti()
opti.subject_to( z >= -0.2 + 0.1*y**2 )
plot_solution(r"Optimal solution hanging chain (convex ground constraints)", opti, y, z, save="convex")

# concave quadratic ground constraints:
opti, y, z = create_opti()
opti.subject_to( z >= -y**2 )
plot_solution(r"Hanging chain (concave ground constraints, $y_i^{init},z_i^{init}=0$)", opti, y, z, init=0, save="concave0")
plot_solution(r"Hanging chain (concave ground constraints, $y_i^{init},z_i^{init}=1$)", opti, y, z, init=1, save="concave1")
plot_solution(r"Hanging chain (concave ground constraints, $y_i^{init},z_i^{init}=-1$)", opti, y, z, init=-1, save="concave-1")
