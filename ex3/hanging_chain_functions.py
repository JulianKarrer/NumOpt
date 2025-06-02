
import numpy as np


""""
    In this file, we create the functions for evaluating the potential energy and its gradient.
"""

# Define the parameters of the problem
N = 40
mi = 4/N
D = 70
g = 9.81
L  = 1 / N
y_begin = 0
z_begin = 0
y_end = 2
z_end = 1



# Define the potential energy function
def f(x):
    y, z = x[:N], x[N:]
    V_elastic = 0
    for i in range(-1, N):
        if i == -1:
            delta_y = y[0] - y_begin
            delta_z = z[0] - z_begin
        elif i+1==N:
            delta_y = y_end - y[-1] 
            delta_z = z_end - z[-1]
        else:
            delta_y = y[i+1] - y[i]
            delta_z = z[i+1] - z[i]
        distance = np.sqrt(delta_y**2 + delta_z**2)
        Vi= 0.5 * D * (distance - L)**2
        V_elastic += Vi
    V = V_elastic + mi * g * np.sum(z)
    return V

# Define the gradient of the potential energy function
def f_grad(x):
    y, z = x[:N], x[N:]

    grad_y = np.zeros(N)
    grad_z = np.zeros(N)
    for i in range(-1, N):
        if i == -1:
            delta_y = y[0] - y_begin
            delta_z = z[0] - z_begin
        elif i+1==N:
            delta_y = y_end - y[-1] 
            delta_z = z_end - z[-1]
        else:
            delta_y = y[i+1] - y[i]
            delta_z = z[i+1] - z[i]
        distance = np.sqrt(delta_y**2 + delta_z**2)
        if i >= 0:
            grad_y[i] = grad_y[i] + D * (-delta_y) / distance * (distance - L)
            grad_z[i] = grad_z[i] + D * (-delta_z) / distance * (distance - L)
        if i+1 < N:
            grad_y[i+1] = grad_y[i+1] + D * delta_y / distance * (distance - L)
            grad_z[i+1] = grad_z[i+1] + D * delta_z / distance * (distance - L)
    # Add the gradient of the gravity energy
    grad_z = grad_z + mi * g
    grad = np.concatenate([grad_y, grad_z])
    return grad
