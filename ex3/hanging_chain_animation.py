
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from hanging_chain_functions import y_begin, z_begin, y_end, z_end, N

""""
    This file is for making an animation
"""

def make_animation(list_y, list_z):

    fig, ax = plt.subplots(figsize=(8, 6))

    
    ax.set_title(r'Position of chain at current iterate')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.grid()

    pad = 0.5
    ax.set_xlim([y_begin - pad, y_end + pad]) # type: ignore
    ax.set_ylim([-3, max(z_begin, z_end) + pad]) # type: ignore

    ax.plot(y_begin, z_begin, "rx", markersize=10)
    ax.plot(y_end, z_end, "rx", markersize=10)


    all_artist = []
    N_iter = len(list_y)

    y = np.zeros(N+2)
    z = np.zeros(N+2)
    y[0] = y_begin
    y[-1] = y_end
    z[0] = z_begin
    z[-1] = z_end


    for i in range(N_iter):

        y[1:-1] = list_y[i]
        z[1:-1] = list_z[i]

        # Update animation
        art0 = ax.text(1, -2, f"iter={i}")
        [art1] = ax.plot(y, z, 'b--')
        [art2] = ax.plot(y, z, 'ro')
        all_artist.append([art0, art1, art2])
    ax.plot(y, z, 'b--', alpha=0.2)
    ax.plot(y, z, 'ro', alpha=0.2)

    interval = int(5 * 1000 / N_iter) # animation lasts 5 seconds
    return animation.ArtistAnimation(fig, all_artist, interval=interval, repeat=True, repeat_delay=5000, blit=True)

