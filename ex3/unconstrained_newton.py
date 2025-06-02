import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True})

rho = 10 # set this parameter to 5. You can also play with this parameter.

## Define the objective function
def f(x, y) -> float:
    return 0.5*( (x-1)**2 + y**2 + rho*(y-np.cos(x))**2 )

## Define the gradient:
def gradient(x, y):
   return np.array([
       x + rho*y*np.sin(x) - rho/2*np.sin(2*x) - 1,
       (1+rho)*y - rho*np.cos(x)
    ])

## Define the Hessian Approximations
def Hessian(x, y, approximation):
    """
        Approximation is a string, equal to one of the following:
         - "exact" for exact Hessian approximation
         - "GN" for Gauss-Newton hessian approximation
         - "steeepest" for alpha I with alpha= 10
    """
    if approximation == "exact":
        return np.matrix(
            [[1 + rho*y*np.cos(x) - rho*np.cos(2*x), rho*np.sin(x)],
             [rho*np.sin(x), 1+rho]]
        )
    elif approximation == "GN":
        return np.matrix(
            [[1 + rho*np.sin(x)**2, rho*np.sin(x)],
             [rho*np.sin(x), 1+rho]]
        )
    elif approximation == "steepest":
        alpha = 10
        return np.matrix(
            [[alpha, 0],
             [0, alpha]]
        )
    else:
        raise ValueError("Unknown approximation type. Choose from 'exact', 'GN', or 'steepest'.")


def Newton_step(x, y, hessian_approximation):
    """
        Perform a Newton step using the specified approximation for the Hessian.
    """
    grad = gradient(x, y)
    H = Hessian(x, y, hessian_approximation)
    step = (-np.linalg.inv(H) @ grad)
    return x+step[0,0], y+step[0,1]

def stopping_condition(x, y) -> bool:
    """
        Check the stopping condition for the Newton method.
    """
    grad = gradient(x,y)
    return np.dot(grad,grad) <= 1e-6



# Run the algorithm (nothing to do here)
hessian_approximations = ["exact", "GN", "steepest"]
all_iterates = {}
N_max = 50
for hessian_approximation in hessian_approximations:
    iterates = []
    x, y = (0,10)
    for k in range(N_max):
        iterates.append((x, y))
        x, y = Newton_step(x, y, hessian_approximation)
        if stopping_condition(x, y):
            break
    all_iterates[hessian_approximation] = iterates



# Plot the solutions
plot = "3D" # either "3D" or "value"
if plot=="3D":
    N_grid = 500
    X = np.linspace(-10, 10, N_grid)
    Y = np.linspace(-10, 10, N_grid)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.grid()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$f(x, y)$') # type: ignore


    for hessian_approximation, iterates in all_iterates.items():
        iterates = np.array(iterates)
        x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
        ax.plot(x_iterates, y_iterates, f(x_iterates, y_iterates), "-o", markersize=3, label=hessian_approximation)
    ax.plot_surface(X, Y, Z) # type: ignore
    ax.legend()
    ax.set_title("Newton's method with different Hessian approximations")
    plt.show()
elif plot=="value":
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'number of iterations')
    ax.set_ylabel(r'$f(x_k, y_k)$')
    ax.grid()
    # ax.set_xscale('log')
    ax.set_yscale('log')
    for hessian_approximation, iterates in all_iterates.items():
        f_iterates = [f(x,y) for x,y in iterates]
        ax.plot(range(len(f_iterates)), f_iterates, "-o", markersize=3, label=hessian_approximation)
    ax.legend()
    ax.set_title("Newton's method with different Hessian approximations")
    plt.show()