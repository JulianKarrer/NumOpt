"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""



# This file is dedicated to the implementation of the Newton method for solving the problem:
#       w^{16} = 10
# d/dw F(w) = d/dw w^{16} = 16w^{15}
# d^2/dw^2 F(w) = d/dw 16w^{15} = 240 w^{14}


import numpy as np
import matplotlib.pyplot as plt

def newton_step(w):
    Fw = w**16 - 10
    dF_dw = 16*w**15
    p_k = -(1.0/dF_dw)*Fw
    return w + p_k


def newton_method(w0, tol=1e-6, N_max=1000):
    w = w0
    i = 0
    for i in range(N_max):
        w = newton_step(w)
        if abs(w**16-10)<=tol:
            break
    return w, i
        
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
ax.set_title(r"Newton's method for solving $w^{16} = 10$")
ax.legend()
plt.show()


