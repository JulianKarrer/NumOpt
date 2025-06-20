"""
    Leo Simpson, University of Freiburg (teacher assistant), 2025.

    This file is for an exercise for the course Numerical Optimization by Prof. Moritz Diehl.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# This file is for making an animation

def make_animation(list_s, list_u, list_other_s=None, title="Gauss-Newton method for a control problem"):

    fig, axs = plt.subplots(2, figsize=(8, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    ax_s = axs[0] # this is for the state trajectory
    ax_u = axs[1] # this is for the control trajectory
    ax_s.set_title(r'State trajectory')
    ax_u.set_title(r'Control trajectory')
    ax_u.set_xlabel(r'$t$')
    ax_u.set_ylabel(r'$u$')
    ax_s.set_ylabel(r'$s$')
    ax_u.grid()
    ax_s.grid()

    ax_s.set_ylim(-2, 2)
    ax_u.set_ylim(-1, 1)


    s_traj_opt = list_s[-1]
    u_traj_opt = list_u[-1]

    pad = 0.2
    ax_s.set_ylim( s_traj_opt.min() - pad, s_traj_opt.max() + pad)
    ax_u.set_ylim( u_traj_opt.min() - pad, u_traj_opt.max() + pad)

    T = len(s_traj_opt) # T = N+1
    t = np.arange(T)
    ax_s.plot(t, s_traj_opt, 'r--', alpha=0.2)
    ax_u.plot(t, u_traj_opt, 'r--', alpha=0.2)
    ax_s.plot(0, s_traj_opt[0], "gx", markersize=10)
    ax_s.plot(T, s_traj_opt[-1], "gx", markersize=10)


    all_artist = []
    N_iter = len(list_s)
    for k in range(N_iter):
        s_traj = list_s[k]
        u_traj = list_u[k]

        # Update animation
        art0= ax_s.text(1, 0.5, f"iter={k}")
        [art_s] = ax_s.plot(t, s_traj, 'b.')
        [art_u] = ax_u.plot(t, u_traj, 'b.')
        artists = [art0, art_s, art_u]
        if list_other_s is not None:
            s_traj_other = list_other_s[k]
            [art_s_other] = ax_s.plot(t, s_traj_other, '.', color="purple")
            artists.append(art_s_other)
        all_artist.append(artists)
        ax_s.legend()
    interval = int(5 * 1000 / N_iter) # animation lasts 5 seconds
    return animation.ArtistAnimation(fig, all_artist, interval=interval, repeat=True, repeat_delay=1500, blit=True)

