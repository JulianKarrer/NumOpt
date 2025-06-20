"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""

# This file is for the Gauss-Newton method for the control problem defined in the exercise.



import numpy as np
import matplotlib.pyplot as plt
from control_phi import phi, nabla_s_phi, nabla_u_phi, rollout
from control_animation import make_animation

N = 100
s_terminal = 1
s_0 = 0

def s_u(x):
    s = x[:N+1]
    u = x[N+1:]
    assert len(s)==N+1
    assert len(u)==N+1
    return s, u

def F(x):
    # compute F
    _s,u = s_u(x)
    return u

def nabla_F(x):
    # compute nabla_F
    return np.vstack([
        np.zeros((N+1, N+1)),
        np.eye(N+1, N+1)
    ])

def g(x):
    # compute g
    s,u = s_u(x)
    # constraints should be column vectors
    c_0 = np.array([[s[0] - s_0]]).reshape(1,1)
    c_i = np.array([[s[i] - phi(s=s[i-1], u=u[i-1]) for i in range(1, N+1)]]).reshape(N,1)
    c_n_1 = np.array([[s[N] - s_terminal]]).reshape(1,1)
    res = np.vstack([c_0, c_i, c_n_1])
    assert c_0.shape == (1,1) and c_i.shape == (N, 1) and c_n_1.shape == (1, 1) and res.shape == (N+2, 1)
    return res

def nabla_g(x):
    # compute nabla_g
    # nabla_g is the jacobian transposed, i.e. a 2N+2 x N+2 matrix
    delta_t = nabla_u_phi(0,0)
    u_mat = np.array(
        [[0 for i in range(N+1)]]+[ # row of zeros
            [(-delta_t if j+1==i else 0) for j in range(N+1)] for i in range(1,N+1)    
        ]+[[0 for i in range(N+1)]] # row of zeros
    )
    s,_u = s_u(x)
    s_mat = np.eye(N+2, N+1)
    s_mat[-1,-1]= 1
    for i in range(1,N+1):
        s_mat[i, i-1] = (-1-3*delta_t*(s[i-1][0]+1)**2)
    J_g = np.hstack([s_mat, u_mat])
    return J_g.T 

def GaussNewton_step(x_k):
    # compute the Gauss-Newton step
    F_k = F(x_k)
    g_k = g(x_k)
    nab_F_k = nabla_F(x_k)
    nab_g_k = nabla_g(x_k)
    B = nab_F_k @ nab_F_k.T
    A = np.vstack([
            np.hstack([B,        -nab_g_k       ]), 
            np.hstack([nab_g_k.T, np.zeros(shape=(N+2, N+2)) ])])
    b =  np.vstack([
            -(nab_F_k @ F_k).reshape(2*N+2,1),
            -g_k
        ])
    dx_lambda = np.linalg.solve(A, b)
    dx = dx_lambda[:2*N+2]
    assert len(dx) == len(x_k)
    return x_k + dx

def GaussNewton(x0, tol=1e-3, maxiter=100):
    # The Gauss-Newton method (nothing to do here)
    x = x0
    iterates = [x]
    for i in range(maxiter):
        new_x = GaussNewton_step(x)
        step_length = abs(new_x - x).max()
        print(f"Iteration {i}, Step length: {step_length:.2e}")
        if step_length < tol:
            break
        x = new_x
        iterates.append(x)
    return iterates

n = 2*N+2
x0 = np.zeros(n).reshape(n,1) # initial guess
iterates = GaussNewton(x0)


list_s = [s_u(x)[0] for x in iterates]
list_u = [s_u(x)[1] for x in iterates]


anim = make_animation(list_s, list_u)
plt.show()

# this is for the last question
list_hat_s = [rollout(u, s_0) for u in list_u]
anim = make_animation(list_s, list_u, list_other_s=list_hat_s)
plt.show()