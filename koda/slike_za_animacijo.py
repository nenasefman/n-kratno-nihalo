import os, glob
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
from funkcije import *


def slike_za_animacijo_2x2(reseni_sistemi, l, radij, dt, shr_dir, fps, min_sv = 0, shrani=0):
    """
    Funkcija nariše sliko, na kateri so na na štirih podslikah narisana dvojna nihala,
    vsako z malo drugačnima začetnim pogojem.

    reseni  <- seznam štirih rešenih sistemov (za vsako podsliko eden)
    l1, l2  <- dolžini vrvic
    radij   <- radij kroglic
    dt      <- korak s katerim numerično rešujemo diferencialne enačbe
    fps     <- slike na sekundo
    min_sv  <- privzeta vrednost za svetlost barv
    shrani  <- 0 = prikaz v živo, 1 = shranjevanje frameov
    """

    assert len(reseni_sistemi) == 4, "Podaj točno 4 reŠEne sisteme."

    k = int((1/fps)/dt)

    # Ustvari direktorij in shrani slike
    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # Priprava figure
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    # Figure 1920x1080
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
    axs = axs.flatten()  # za lažjo obravnavo

    # Iz podatkov izračunamo max dolžino animacije
    max_len = min([r.shape[0] for r in reseni])

    for frame_i in range(0, max_len, k):

        for idx in range(4):
            ax = axs[idx]
            resen = reseni[idx]

            theta1, theta2 = resen[:, 0], resen[:, 2]
            omega1, omega2 = resen[:,1], resen[:,3]
            omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

            x1 = l1 * np.sin(theta1)
            y1 = -l1 * np.cos(theta1)
            x2 = x1 + l2 * np.sin(theta2)
            y2 = y1 - l2 * np.cos(theta2)

            # barve
            b1 = barva_kroglice(theta1[frame_i], omega1[frame_i], omega_max, min_sv)
            b2 = barva_kroglice(theta2[frame_i], omega2[frame_i], omega_max, min_sv)

            ax.clear()

            # palice
            ax.plot([0, x1[frame_i]], [0, y1[frame_i]], lw=2, c=b1)
            ax.plot([x1[frame_i], x2[frame_i]], [y1[frame_i], y2[frame_i]], lw=2, c=b2)

            # kroglice
            ax.add_patch(Circle((0, 0), radij, fc='k'))
            ax.add_patch(Circle((x1[frame_i], y1[frame_i]), radij, fc=b1, ec=b1))
            ax.add_patch(Circle((x2[frame_i], y2[frame_i]), radij, fc=b2, ec=b2))

            # meje
            lim = l1 + l2 + radij
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.axis("off")

        plt.tight_layout()

        if shrani == 1:
            plt.savefig(f"{shr_dir}/frame_{frame_i:05d}.png")
        else:
            plt.pause(1/fps)

    plt.close()



def narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, min_sv = 0, shrani=0):

    k = int((1/fps)/dt)

    theta1, theta2 = resen[:, 0], resen[:, 2]
    
    # kotni hitrosti (dtheta1, dtheta2):
    omega1, omega2 = resen[:,1], resen[:,3]
    omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    for t_i in range(0, resen.shape[0], k):
        # narišem kroglice
        c0 = Circle((0, 0), radij, fc='k', zorder=10)

        barva1 = barva_kroglice(theta1[t_i], omega1[t_i], omega_max, min_sv)
        barva2 = barva_kroglice(theta2[t_i], omega2[t_i], omega_max, min_sv)

        # narišem palčke od (0,0) do 1. in 2. kroglice
        ax.plot([0, x1[t_i]], [0, y1[t_i]], lw=1, c=barva1)
        ax.plot([x1[t_i], x2[t_i]], [y1[t_i], y2[t_i]], lw=1, c=barva2)

        # narišem kroglici
        c1 = Circle((x1[t_i], y1[t_i]), radij, fc=barva1, ec=barva1, zorder=10)
        c2 = Circle((x2[t_i], y2[t_i]), radij, fc=barva2, ec=barva2, zorder=10)

        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # Meje osi
        ax.set_xlim(-l1 - l2 - radij, l1 + l2 + radij)
        ax.set_ylim(-l1 - l2 - radij, l1 + l2 + radij)
        plt.axis("off")

        if shrani == 1:
            plt.savefig(f'{shr_dir}/frame_{t_i:05d}.png', dpi=72) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo
        
        plt.cla()






