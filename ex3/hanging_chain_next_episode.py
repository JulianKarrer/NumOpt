
import numpy as np
import matplotlib.pyplot as plt

from hanging_chain_functions import f, f_grad, N
from hanging_chain_animation import make_animation

"""
    You need to implement the two following functions,
    one for the BFGS update, and the other one for the globalization.

"""
with_BFGS = True # Choose to perform the BFGS update or not

def my_globalization(x, dx, grad):
    t = 1.
    gamma = 0.1
    beta = 0.9
    for i in range(1000):
        x_candidate = x + t * dx
        # check the Armijo condition of sufficient descent
        if f(x_candidate) <= f(x) + gamma*t*np.dot(grad, dx): # TODO
            return t
        else:
            t *= beta
    raise ValueError("Globalization did not finish")

def my_update(x, grad, old_x, old_grad, old_Bk):
    Bk = old_Bk.copy()
    if with_BFGS and old_grad is not None:
        # BFGS update
        s = x - old_x 
        y = grad - old_grad
        Bk -= ((Bk @ s) @ (s.T @ Bk))/np.dot(s, (Bk @ s)) + np.dot(y,y)/np.dot(s,y)
    dx = -np.linalg.inv(Bk) @ grad
    t = my_globalization(x, dx, grad)
    new_x = x + t * dx
    return new_x, Bk




""""
    In this part of the file, we run the optimization algorithm using the two functions above.
"""
# initialize the optimization variables
y = np.linspace(-0.1, -0.5, N) 
z = np.zeros(N) 
x_opt = np.concatenate((y, z))
old_x, old_grad = None, None
Bk = 100*np.eye(2*N, 2*N)

y_list, z_list = [], []
N_max = 1_000
for i in range(N_max):
    # Compute the gradient
    grad = f_grad(x_opt)
    
    if np.dot(grad, grad) <= 1e-3: # type: ignore
        print("Converged !")
        break
    # Perform update
    new_x_opt, Bk = my_update(x_opt, grad, old_x, old_grad, Bk)
    
    # Update variables
    old_x = x_opt
    x_opt = new_x_opt
    old_grad = grad

    # Save variables
    y_list.append(x_opt[:N])
    z_list.append(x_opt[N:])

# Make animation
anim = make_animation(y_list, z_list)
plt.show()