"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""

import numpy as np
delta_t = 0.1
def phi(s, u):
    # function phi defined in the exercise
    return s + delta_t * (u + (s+1)**3)

def nabla_s_phi(s, u):
    # compute nabla_s phi
    return 1 + delta_t * 3 * (s + 1)**2

def nabla_u_phi(s, u):
    # compute nabla_u phi
    return delta_t

def rollout(seq_u, x0):
    N = len(seq_u) # length of the sequence
    seq_x = np.zeros(N)
    seq_x[0] = x0
    for i in range(N-1):
        seq_x[i+1] = phi(seq_x[i], seq_u[i])

    return seq_x