"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""

import numpy as np

# Define the parameters of the problem
N = 40
m = 0.1
D = 70
g = 9.81
y_begin = -2
z_begin = 1
y_end = 2
z_end = 1

# Constraints in the form z > a * y + b
constraints = [
    {"a": 0, "b": 0.5},
    {"a": 0.1, "b": 0.5},
    {"a": -1., "b": -1.},
]

def create_matrices():
    # write:
    # V(y, z) = 1/2 || J^Ty + d_y ||^2 + 1/2 || J^T z + d_z ||^2 + c_z^T z + cnst

    J = np.sqrt(D) * (np.diag(np.ones(N+1)) - np.diag(np.ones(N), -1))
    d_y = J[0] * y_begin + J[-1] * y_end
    d_z = J[0] * z_begin + J[-1] * z_end
    J = J[1:-1] # remove first and last columns because y_0 and y_N are fixed
    c_z = np.ones(N-1) * m * g 

    # Then, define the matrix Q and the vector c
    # Such that x.T Q x + c.T x + cnst = V(y, z) (with x = [y, z])


    Q = np.zeros((2*(N-1), 2*(N-1)))
    Q[:N-1, :N-1] = J @ J.T
    Q[N-1:, N-1:] = J @ J.T # Q_y and Q_z are the same

    c = np.zeros(2*(N-1))
    c[:N-1] = J @ d_y
    c[N-1:] = J @ d_z + c_z


    # Define the matrix A and the vector b
    # Such that A x + b > 0 corresponds to the constraints
    A_list = []
    b_list = []
    I = np.eye(N-1) # identity matrix 
    for dict_ab in constraints:
        A = np.concatenate([-dict_ab["a"]*I, I], axis=1)
        b = -dict_ab["b"]*np.ones(N-1)
        A_list.append(A)
        b_list.append(b)
    A = np.concatenate(A_list, axis=0)
    b = np.concatenate(b_list, axis=0)
    return Q, c, A, b






