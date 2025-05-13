import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt


# for people who cannot see the plots
if mpl.get_backend() == 'agg':
    mpl.use('WebAgg')
print(f"backend: {mpl.get_backend()}")


# create empty optimization problem
opti = ca.Opti()

N = 40

y = opti.variable(N)
z = opti.variable(N)

m = 4/N           # mass
Di = (70/40)*N    # spring constant
g0 = 9.81         # gravity

# defining the objective
Vchain = 0
for i in range(N):
    if i<N-1:
        Vchain += 1/2 * Di * ((y[i] - y[i+1])**2 + (z[i] - z[i+1])**2)
    Vchain += m*g0*z[i]

# passing the objective to opti
opti.minimize(Vchain)

# TODO: complete the (equality) constraints HERE
opti.subject_to( y[0] == -2 )
opti.subject_to( z[0] == 1 )
opti.subject_to( y[N-1] == 2 )
opti.subject_to( z[N-1] == 1 )

def plot_solution(title):
    # Setting solver to ipopt and solving the problem:
    opti.solver('ipopt')
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

# plot solution for free-hanging chain with no ground constraints
plot_solution(r"Optimal solution hanging chain (without extra constraints)")

# add additional ground constraints
opti.subject_to( z >= 0.5 )
for i in range(2,N):
    opti.subject_to( z[i] - 0.1*y[i] >= 0.5)
plot_solution(r"Optimal solution hanging chain (with extra constraints)")
