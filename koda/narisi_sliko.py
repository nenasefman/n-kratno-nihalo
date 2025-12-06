import matplotlib.pyplot as plt
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from funkcije import *


def narisi_sliko_2(resen, l1, l2, radij, dt, shr_dir, fps, min_sv = 0, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dobimo iz resen_sistem_n
    - l1, l2 <- dolžini prve in druge vrvice
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

    theta1, theta2 = resen[:, 0], resen[:, 2]
    
    # kotni hitrosti (dtheta1, dtheta2):
    omega1, omega2 = resen[:,1], resen[:,3]
    omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')

    frame_id = 0

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
            plt.savefig(f'{shr_dir}/frame_{frame_id:05d}.png', dpi=100) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo

        frame_id += 1        
        plt.cla()

def narisi_sliko_2_enobarvna(resen, l1, l2, radij, dt, shr_dir, fps, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dobimo iz resen_sistem_n
    - l1, l2 <- dolžini prve in druge vrvice
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

    theta1, theta2 = resen[:, 0], resen[:, 2]
    
    # kotni hitrosti (dtheta1, dtheta2):
    omega1, omega2 = resen[:,1], resen[:,3]
    omega_max = max(np.max(np.abs(omega1)), np.max(np.abs(omega2)))

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    if shrani==1:
        os.makedirs(shr_dir, exist_ok=True)

        files = glob.glob(os.path.join(shr_dir, "*.png"))
        for f in files:
            os.remove(f)

    # --- Priprava figure ---
    fig = plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    fig.patch.set_facecolor("black")

    frame_id = 0

    for t_i in range(0, resen.shape[0], k):
        # narišem kroglice
        c0 = Circle((0, 0), radij, fc='k', zorder=10)

        barva = barva_sistema(theta1[t_i], theta2[t_i], omega1[t_i], omega2[t_i], omega_max)

        # narišem palčke od (0,0) do 1. in 2. kroglice
        ax.plot([0, x1[t_i]], [0, y1[t_i]], lw=1, c=barva)
        ax.plot([x1[t_i], x2[t_i]], [y1[t_i], y2[t_i]], lw=1, c=barva)

        # narišem kroglici
        c1 = Circle((x1[t_i], y1[t_i]), radij, fc=barva, ec=barva, zorder=10)
        c2 = Circle((x2[t_i], y2[t_i]), radij, fc=barva, ec=barva, zorder=10)

        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # Meje osi
        ax.set_xlim(-l1 - l2 - radij, l1 + l2 + radij)
        ax.set_ylim(-l1 - l2 - radij, l1 + l2 + radij)
        plt.axis("off")

        if shrani == 1:
            plt.savefig(f'{shr_dir}/frame_{frame_id:05d}.png', dpi=100) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo

        frame_id += 1        
        plt.cla()


def narisi_sliko_3(resen, l1, l2, l3, radij, dt, shr_dir, fps, shrani=0):
    """
    Na podlagi spremenljivke shrani ali sproti pokaže animacijo slik.
    Podatki:
    - resen <- dobimo iz resen_sistem_n
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
    fig = plt.figure(figsize=(1920/120, 1080/120), dpi=120)
    ax = fig.add_subplot(111)

    frame_id = 0

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
            plt.savefig(f'{shr_dir}/frame_{frame_id:05d}.png', dpi=120) #dpi=dots per inch, ločljivost slike
        else:
            plt.pause(1/fps)   # animira v živo
        
        frame_id += 1
        plt.cla()


def slike_za_animacijo_2x2(reseni_sistemi, l_val, radij, dt, shr_dir, fps, shrani=0):
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
    fig.patch.set_facecolor("black")   # črno ozadje

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
            b = barva_sistema(theta1[frame_i], theta2[frame_i], omega1[frame_i], omega2[frame_i], omega_max)

            ax.clear()

            # palice
            ax.plot([0, x1[frame_i]], [0, y1[frame_i]], lw=5, c=b)
            ax.plot([x1[frame_i], x2[frame_i]], [y1[frame_i], y2[frame_i]], lw=5, c=b)

            # kroglice
            ax.add_patch(Circle((0, 0), radij, fc='k'))
            ax.add_patch(Circle((x1[frame_i], y1[frame_i]), radij, fc=b, ec=b))
            ax.add_patch(Circle((x2[frame_i], y2[frame_i]), radij, fc=b, ec=b))

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