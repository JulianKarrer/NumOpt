import matplotlib.pyplot as plt
import numpy as np

## plot the objective function

# Define the objective function
rho = 10 # set this parameter to 5. You can also play with this parameter.
def f(x, y):
    r1 = x-1
    r2 = y
    r3 = y - np.cos(x)
    return (r1**2 + r2**2 + rho * r3**2) / 2.
N = 500
X = np.linspace(-10, 10, N)
Y = np.linspace(-4, 4, N)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.grid()
ax.plot_surface(X, Y, Z)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f(x, y)$')
plt.show()