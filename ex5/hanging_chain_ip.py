"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""

import numpy as np
import matplotlib.pyplot as plt

from hanging_chain_ip_matrices import create_matrices, N
from hanging_chain_ip_animation import make_animation


Q, c, A, b = create_matrices() # Create the matrices for the problem
M = b.shape[0]

def f(x, tau):
    # Objective function for a given tau
    s = A @ x + b # the constraint is s > 0
    if np.any(s <= 0):
        return np.inf
    else:
        # Objective function for the subproblem with log barrier
        return 0.5* np.dot(x, (Q @ x)) + np.dot(c, x) - tau*np.sum(np.log(s)) 

y_list, z_list = [], []
max_iter = 1000
x_k = np.ones(2*(N-1))
tol = 1e-1

for i in range(max_iter):
    print(f"Iteration {i+1}")
    tau_k = 1 / (i+1)**2
    
    # Compute the constraint residual
    s_k = A @ x_k + b # the constraint is s_k > 0
    assert np.all(s_k > 0), "Constraint violated !"

    # Compute the gradient
    grad_f = Q @ x_k + c
    grad = grad_f - tau_k * np.sum([1/s_k[j]*A[j].T for j in range(M)], axis=0) 

    # Check convergence
    inf_norm_grad = np.max(abs(grad))
    print(f"Iteration {i+1}, grad = {inf_norm_grad:.2e}")
    if inf_norm_grad < tol:
        print("Converged !")
        break

    # Compute the Hessian
    H_k = Q + tau_k * np.sum([1/s_k[j]**2 * np.outer(A[j], A[j]) for j in range(M)], axis=0) 

    # Newton step
    dx = -(np.linalg.inv(H_k)) @ grad_f

    # Globalization with backtracking line-search and Armijo criterion
    t = 1         # t_max
    gamma = 0.25  # gamma in (0, 1/2)
    beta = 0.8    # beta  in (0,  1 )
    while f(x_k + t * dx, tau_k) >= f(x_k, tau_k) + gamma*t*np.dot(grad, dx):
        t *= beta
    print(f"Armijo step length: t={t}")

   # Update the solution
    x_k = x_k + t * dx 
   
    # Save variables
    y_list.append(x_k[:N-1])
    z_list.append(x_k[N-1:])

anim = make_animation(y_list, z_list)
plt.show()