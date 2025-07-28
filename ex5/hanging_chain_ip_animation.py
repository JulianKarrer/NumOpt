"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from hanging_chain_ip_matrices import y_begin, z_begin, y_end, z_end, N, constraints


# This file is for making an animation
def make_animation(list_y, list_z):

    fig, ax = plt.subplots(figsize=(10, 10))

    
    ax.set_title(r'Position of chain at current iterate')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.grid()

    # retrieve the optimal solution
    y = np.concatenate(([y_begin], list_y[-1], [y_end]))
    z = np.concatenate(([z_begin], list_z[-1], [z_end])) 
    # plot the solution of the last iteration
    ax.plot(y, z, 'b--', alpha=0.2)
    ax.plot(y, z, 'ro', alpha=0.2)


    pad = 0.2
    y_extreme = np.array([y.min()-pad, y.max()+pad])
    z_extreme = np.array([z.min()-pad, z.max()+pad])
    ax.set_xlim(y_extreme)
    ax.set_ylim(z_extreme)
    # add the lines for the ground constraints
    for ab in constraints:
        ax.plot(y_extreme,  ab["a"] * y_extreme +  ab["b"], ':b')


    all_artist = []
    N_iter = len(list_y)
    for i in range(N_iter):

        y[1:-1] = list_y[i]
        z[1:-1] = list_z[i]

        # Update animation
        art0 = ax.text(0, 1, f"iter={i}")
        [art1] = ax.plot(y, z, 'b--')
        [art2] = ax.plot(y, z, 'ro')
        all_artist.append([art0, art1, art2])
    

    interval = int(5 * 1000 / N_iter) # animation lasts 5 seconds
    return animation.ArtistAnimation(fig, all_artist, interval=interval, repeat=True, repeat_delay=1500, blit=True)

