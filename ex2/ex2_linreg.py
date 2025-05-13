import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})

# setup
N = 30
np.random.seed(1312) # seed RNG for reproducible results

# generate noisy data
x = np.linspace(0.,5.,N)
y_ideal = 3*x+4
y = 3*x+4+np.random.randn(N)

# generate data with outliers
y_out = 3*x+4+np.random.randn(N)
y_out[0]=1e3
y_out[-1]=-1e3

# ordinary least squares regression
def reg(y):
    bar = lambda x: np.sum(x)/N
    a_hat = (bar(x*y) - bar(x)*bar(y)) / (bar(x**2) - bar(x)**2)
    b_hat = (bar(x**2)*bar(y) - bar(x) * bar(x*y)) / (bar(x**2) - bar(x)**2)
    return(a_hat, b_hat, a_hat*x+b_hat)

def reg_cas(y):
    opti = ca.Opti()
    s = opti.variable(N)
    a = opti.variable()
    b = opti.variable()
    # define objective function
    objective = 0
    for i in range(N):
        objective += s[i]
    opti.minimize(objective)
    # define constraints
    for i in range(N):
        opti.subject_to( s[i] + (a*x[i] + b - y[i])  >= 0)
        opti.subject_to( s[i] - (a*x[i] + b - y[i])  >= 0)
    # solve the problem
    opti.solver('ipopt')
    sol = opti.solve()
    a_hat = sol.value(a)
    b_hat = sol.value(b)
    return(a_hat, b_hat, a_hat*x+b_hat)
    

# plot results
def plot(title, filename, a_hat, b_hat, y):
    fig, ax = plt.subplots()
    ax.plot(x, y_ideal, label="original function")
    ax.plot(x, y, "rx", label="noisy data")
    ax.plot(x, y_reg, label="fitted function ($\\hat{{a}}\\approx{a:.3f}, \\hat{{b}}\\approx{b:.3f}$)".format(a=a_hat, b=b_hat))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    ## show plots
    plt.show()
    fig.savefig(filename, dpi=fig.dpi)


# L2 FITS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# perform regular regression
a_hat, b_hat, y_reg = reg(y)
plot(
    "$L_2$ fit of $y=3x+4+\\mathcal{{N}}(0,1), N={n}$".format(n=N),
    "linreg.pdf",
    a_hat,
    b_hat,
    y,
)

# introduce outliers
a_hat, b_hat, y_reg = reg(y_out)
plot(
    "$L_2$ fit of $y=3x+4+\\mathcal{{N}}(0,1), N={n}$ with outliers $y(0)=10^3=-y(5)$".format(n=N),
    "linregout.pdf",
    a_hat,
    b_hat,
    y_out,
)

# L1 FITS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
a_hat, b_hat, y_reg = reg_cas(y)
plot(
    "$L_1$ fit of $y=3x+4+\\mathcal{{N}}(0,1), N={n}$".format(n=N),
    "linregl1.pdf",
    a_hat,
    b_hat,
    y,
)

a_hat, b_hat, y_reg = reg_cas(y_out)
plot(
    "$L_1$ fit of $y=3x+4+\\mathcal{{N}}(0,1), N={n}$ with outliers $y(0)=10^3=-y(5)$".format(n=N),
    "linregoutl1.pdf",
    a_hat,
    b_hat,
    y_out,
)

