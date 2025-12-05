import os, glob
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
from funkcije import *


def slike_za_animacijo_2x2(reseni_sistemi, l_val, radij, dt, shr_dir, fps, min_sv = 0, shrani=0):
    """
    Funkcija nariše sliko, na kateri so na na štirih podslikah narisana dvojna nihala,
    vsako z malo drugačnima začetnim pogojem.

    reseni  <- seznam štirih rešenih sistemov (za vsako podsliko eden)
    l       <- dolžini vrvic 
    radij   <- radij kroglic
    dt      <- korak s katerim numerično rešujemo diferencialne enačbe
    fps     <- slike na sekundo
    min_sv  <- privzeta vrednost za svetlost barv
    shrani  <- 0 = prikaz v živo, 1 = shranjevanje frameov
    """

    assert len(reseni_sistemi) == 4, "Podaj točno 4 rešene sisteme."

    l1, l2 = l_val
    k = int((1/fps)/dt)

    # Ustvari direktorij in shrani slike
    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # Figure 1920x1080
    fig, axs = plt.subplots(2, 2, figsize=(1920/120, 1080/120), dpi=120) #ustvari mrežo 2x2, axs je tabela 2x2 ax objektov
    axs = axs.flatten()  # omogoča dostop do axs[0], axs[1] itd.

    # Iz podatkov izračunamo max dolžino animacije
    max_len = min([r.shape[0] for r in reseni_sistemi])

    frame_id = 0

    for frame_i in range(0, max_len, k):

        for idx in range(4):
            ax = axs[idx]
            resen = reseni_sistemi[idx]

            theta1, theta2 = resen[:, 0], resen[:, 2]
            omega1, omega2 = resen[:,1], resen[:,3]
            omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

            x1 = l1 * np.sin(theta1)
            y1 = -l1 * np.cos(theta1)
            x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
            y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

            # barve
            b1 = barva_kroglice(theta1[frame_i], omega1[frame_i], omega_max, min_sv)
            b2 = barva_kroglice(theta2[frame_i], omega2[frame_i], omega_max, min_sv)

            ax.clear()

            # palice
            ax.plot([0, x1[frame_i]], [0, y1[frame_i]], lw=5, c=b1)
            ax.plot([x1[frame_i], x2[frame_i]], [y1[frame_i], y2[frame_i]], lw=5, c=b2)

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
            plt.savefig(f"{shr_dir}/frame_{frame_id:05d}.png", dpi = 120)
        else:
            plt.pause(1/fps)

        frame_id += 1

    plt.close()


tmax, dt = 30, 0.01
zac_pog_1 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
zac_pog_2 = np.array([np.pi/2, 0, np.pi/2, 0])
zac_pog_3 = np.array([np.pi/2, 0, np.pi, 0])
zac_pog_4 = np.array([np.pi/2, 0, 3*np.pi/4, 0])
n = 2
l_val = [1 for _ in range(n)]
m_val = [1 for _ in range(n)]
g_val = 9.81

reseni_sistemi = [
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_1),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_2),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_3),
    resen_sistem_n(n, g_val, m_val, l_val, tmax, dt, zac_pog_4)
]

radij = 0.1
shr_dir = "./output/2x2_slikice"
fps = 30

slike_za_animacijo_2x2(reseni_sistemi, l_val, radij, dt, shr_dir, fps, min_sv = 0.5, shrani=0)





