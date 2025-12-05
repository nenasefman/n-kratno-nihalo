import os, glob
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from funkcije import resen_sistem_n


# NENA!

#naredi trojno nihalo narisi_sliko_3

# gravitacija
g = 9.81


def narisi_sliko_2(resen, l1, l2, l3, radij, dt, shr_dir, fps, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dpbimo iz resen_sistem_n
    - l1, l2, l3 <- dolžine vrvic
    - radij <- radij kroglice
    - dt <- dt, ki se uporabi v resen_sistem_n
    - shr_dir <- kam se naj shranjujejo slike - npr. "./output/dvojno_nihalo_frames"
    - fps = frames per second
    - shrani <- privzeta vrednost = 0. Če je 0, ti jih samo na zaslonu pokaže, če pa je 1 pa slike shranjuje v shr_dir
    """

    # vsakih k časovnih enot naredim eno sliko, tako da dobim želen fps (frames per second)
    # 1\fps = koliko sekund preteče med slikama, to delim z dt, da dobim na koliko izračunanih podatkov 
    # moram narediti eno sliko
    k = int((1/fps)/dt)

    theta1, theta2, theta3 = resen[:, 0], resen[:, 2], resen[:, 4]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    x3 = x2 + l3 * np.sin(theta3)
    y3 = y2 - l3 * np.cos(theta3)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    for t_i in range(0, resen.shape[0], k):
        # narišem palčke od (0,0) do 1. in 2. kroglice
        ax.plot([0, x1[t_i], x2[t_i], x3[t_i]], [0, y1[t_i], y2[t_i], y3[t_i]], lw=1, c='k')

        # narišem kroglice
        c0 = Circle((0, 0), radij, fc='k', zorder=10)
        c1 = Circle((x1[t_i], y1[t_i]), radij, fc='r', ec='r', zorder=10)
        c2 = Circle((x2[t_i], y2[t_i]), radij, fc='r', ec='r', zorder=10)
        c3 = Circle((x3[t_i], y3[t_i]), radij, fc='r', ec='r', zorder=10)
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c3)

        # Meje osi
        ax.set_xlim(-l1 - l2 -l3 - radij, l1 + l2 + l3 + radij)
        ax.set_ylim(-l1 - l2 -l3 - radij, l1 + l2 + l3 + radij)
        plt.axis("off")

        if shrani == 1:
            plt.savefig(f'{shr_dir}/frame_{t_i:05d}.png', dpi=72) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo
        
        plt.cla()


tmax, dt = 10, 0.01
zac_pog = np.array([np.pi/2, 0, 3*np.pi/4, 0, 3*np.pi/4, 0])
n = 3
l_val = [1, 1, 1]
m_val = [1 for _ in range(n)]
g_val = 9.81

resen = resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog)

radij = 0.03
shr_dir = "./output/frojno_nihalo_frames"
fps = 10

narisi_sliko_2(resen, 1, 1, radij, dt, shr_dir, fps, shrani=0)

