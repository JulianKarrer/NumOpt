"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""




# This file is dedicated to the implementation of the Newton method for solving the problem:
#         w^{16} = 10

# Here, we reformulate the problem in the lifted form as follows:
#         w_2 - w_1^2 = 0
#         w_3 - w_2^2 = 0
#         w_4 - w_3^2 = 0
#         10 - w_4^2 = 0


import numpy as np
import matplotlib.pyplot as plt

def F(w):
    w_2 = w[1]
    w_3 = w[2]
    w_4 = w[3]
    return np.array([w_2, w_3, w_4, 10]) - w**2

def dF(w):
    w_1 = w[0]
    w_2 = w[1]
    w_3 = w[2]
    w_4 = w[3]
    dF = np.vstack([
        np.array([-2*w_1,   1,  0,  0]),
        np.array([0,    -2*w_2, 1,  0]),
        np.array([0,    0,  -2*w_3, 1]),
        np.array([0,    0,  0,  -2*w_4]),
    ])
    return dF

def newton_step(w):
    p_k = -np.linalg.inv(dF(w))@F(w)
    return w + p_k


def newton_method(w_bar, tol=1e-6, N_max=1000):
    i = 0
    w = np.array([w_bar, w_bar, w_bar, w_bar])
    for i in range(N_max):
        w = newton_step(w)
        if np.linalg.norm(F(w), ord=np.inf)<=tol:
            break
    return w[0], i # The method did not converge
        
number_of_iteration = []
initial_guesses = np.linspace(0.01, 5, 1000)
for w0 in initial_guesses:
    w_star, n = newton_method(w0)
    number_of_iteration.append(n)


# Plot the number of iterations for convergence
fig, ax = plt.subplots(figsize=(10, 5))
w_star = 10**(1/16.)
ax.grid()
ax.plot(initial_guesses, number_of_iteration, '-', label='Iterations')
ax.axvline(x=w_star, color='r', linestyle='--', label='w_star')
ax.set_xlabel(r'Initial guess $w_0$')
ax.set_ylabel('Number of iterations to converge')
ax.set_title(r"Lifted Newton's method for solving $w^{16} = 10$")
ax.legend()
plt.show()


